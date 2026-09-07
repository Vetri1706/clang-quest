# Mathematical specification and implementation derivations

All gradients below are vector–Jacobian products. Arrays use row vectors for token features. Computation is real-valued in the derivation; finite-precision execution is an approximation to that arithmetic. “Exact attention” means the full softmax attention function rather than an approximate sparse or kernelized substitute. It does not mean bitwise equivalence across reduction orders.

## 1. Reverse-mode tensor calculus

Let a scalar objective be \(\mathcal L\) and let \(\bar x=\partial\mathcal L/\partial x\). A graph node \(y=f(x_1,\ldots,x_k)\) stores its primal value and a closure computing

\[
(\bar x_1,\ldots,\bar x_k)
=\left(\left(\frac{\partial f}{\partial x_1}\right)^T\bar y,\ldots,
\left(\frac{\partial f}{\partial x_k}\right)^T\bar y\right).
\]

An iterative topological traversal visits shared ancestors once. The backward traversal sums cotangents from every outgoing use before differentiating the ancestor. Leaf gradients accumulate between backward calls until explicitly cleared. Intermediate cotangents are discarded unless retained. Broadcasting is inverted by summing inserted leading dimensions and every original singleton dimension.

For elementwise operations,

\[
\overline{a+b}=\begin{cases}\bar a=\bar y\\\bar b=\bar y\end{cases},\qquad
\bar a=\bar y\odot b,\quad \bar b=\bar y\odot a\quad(y=a\odot b),
\]
\[
\bar a=\bar y/b,\quad \bar b=-\bar y\odot a/b^2\quad(y=a/b),
\qquad \bar x=\bar y\odot p x^{p-1}\quad(y=x^p).
\]

These expressions precede the broadcast-gradient reductions. Addition notation above refers to the operation's two input cotangents, not a derivative of a sum without an output variable.

For \(C=AB\), \(A\in\mathbb R^{m\times k}\), \(B\in\mathbb R^{k\times n}\),

\[
C_{ij}=\sum_{r=1}^k A_{ir}B_{rj},\qquad
\bar A=\bar C B^T,\qquad \bar B=A^T\bar C.
\]

Vector operands are temporarily promoted to matrices; the corresponding singleton output axis is restored to its gradient. Batch dimensions follow NumPy broadcasting, followed by the same unbroadcast operation. The tiled reference kernel partitions \(m,n,k\) and accumulates \(A_{I,K}B_{K,J}\) into \(C_{I,J}\); its arithmetic is the same contraction. NumPy's native matrix product may use a vendor BLAS backend. Neither path uses an external autograd library.

Embedding is a gather \(Y_{bt:}=E_{i_{bt}:}\). Its adjoint is a scatter-add,

\[
\bar E_{v:}=\sum_{(b,t):i_{bt}=v}\bar Y_{bt:}.
\]

Repeated token IDs therefore require accumulation rather than indexed assignment. Views invert their shape/permutation transformations; indexed operations scatter cotangents into the source shape.

## 2. Stable softmax and cross-entropy

For row logits \(s\), define \(m=\max_j s_j\), \(z=\sum_j\exp(s_j-m)\). Then

\[
p_j=\frac{\exp(s_j-m)}{z},\qquad
\log p_j=s_j-m-\log z,
\qquad
\bar s_j=p_j\left(\bar p_j-\sum_kp_k\bar p_k\right).
\]

Subtracting the row maximum prevents exponential overflow for finite logits. A boolean attention mask sets disallowed probabilities to zero. A fully masked attention row is separately defined to return a zero output and zero gradients, avoiding the undefined numerical expression \(-\infty-(-\infty)\).

With target token \(y_{bt}\), nonnegative supervision weight \(w_{bt}\), and \(W=\sum_{bt}w_{bt}>0\),

\[
\mathcal L_{CE}=-\frac1W\sum_{bt}w_{bt}\log p_{bt,y_{bt}},\qquad
\frac{\partial\mathcal L_{CE}}{\partial s_{btv}}
=\frac{w_{bt}}W(p_{btv}-\mathbf1[v=y_{bt}]).
\]

Masks apply to shifted targets, not to their source tokens. Zero-supervision SFT windows must be skipped. Combining microbatches by their target counts recovers the global token mean; averaging unequal microbatch means does not.

## 3. Tiled online attention: the FlashAttention recurrence

The mathematical function is

\[
S=QK^T/\sqrt d+M,\qquad P=\operatorname{softmax}_{row}(S),\qquad O=PV,
\]

where \(M_{ij}=0\) for permitted pairs and \(-\infty\) otherwise. Causality permits \(j\le i+o\), with query offset \(o\) when a query block is positioned after an earlier key prefix.

For a fixed query tile, maintain row statistics \(m_i,\ell_i\) and unnormalized output accumulator \(a_i\). Initialize \(m_i=-\infty,\ell_i=0,a_i=0\). On key/value tile \(J\), compute \(s_{iJ}\) and update

\[
\widetilde m_i=\max(m_i,\max_{j\in J}s_{ij}),\qquad
\alpha_i=\exp(m_i-\widetilde m_i),
\]
\[
\widetilde p_{ij}=\exp(s_{ij}-\widetilde m_i),\qquad
\widetilde\ell_i=\alpha_i\ell_i+\sum_{j\in J}\widetilde p_{ij},
\]
\[
\widetilde a_i=\alpha_i a_i+\sum_{j\in J}\widetilde p_{ij}V_j.
\]

Only permitted finite entries contribute. After every tile, replace the running quantities with their tilded values. At termination,

\[
O_i=a_i/\ell_i,\qquad L_i=m_i+\log\ell_i.
\]

A zero denominator invokes the fully masked-row convention. The recurrence follows by expressing old and new exponential sums relative to their common maximum; the scaling factor converts the old normalization into the new one.

Backward retains \(Q,K,V,O,L\), not the full \(P\). For each reconstructed tile,

\[
P_{ij}=\exp(S_{ij}-L_i),\qquad
D_i=\sum_c \bar O_{ic}O_{ic},
\]
\[
\bar V_J\mathrel{+}=P_{I,J}^T\bar O_I,
\qquad
\bar P_{ij}=\bar O_i\cdot V_j,
\]
\[
\bar S_{ij}=P_{ij}(\bar P_{ij}-D_i),\qquad
\bar Q_I\mathrel{+}=\bar S_{I,J}K_J/\sqrt d,
\qquad
\bar K_J\mathrel{+}=\bar S_{I,J}^TQ_I/\sqrt d.
\]

The identity for \(D_i\) uses \(O_i=\sum_jP_{ij}V_j\). It removes the need to store a separate softmax-gradient row reduction. The retained attention state is linear in sequence length; score temporaries depend on tile sizes. The implementation reproduces this algorithm in NumPy. It is not a GPU fused SRAM kernel and carries no claim of FlashAttention hardware throughput. The original IO-aware attention method is described by [Dao et al.](https://arxiv.org/abs/2205.14135).

## 4. RMSNorm and LayerNorm

For a feature row \(x\in\mathbb R^d\), let

\[
r=\left(\frac1d\sum_jx_j^2+\epsilon\right)^{-1/2},\qquad y_j=\gamma_j x_j r.
\]

Writing \(h_j=\bar y_j\gamma_j\), differentiation gives

\[
\bar x_j=r h_j-x_jr^3\frac1d\sum_kh_kx_k,
\qquad
\bar\gamma_j=\sum_{\text{batch,time}}\bar y_jx_jr.
\]

The epsilon remains inside the root; deleting it from the derivative would change the implemented function. RMSNorm does not subtract the feature mean. See the [RMSNorm paper](https://arxiv.org/abs/1910.07467) for the normalization proposal.

LayerNorm additionally defines

\[
\mu=\frac1d\sum_jx_j,\quad v=\frac1d\sum_j(x_j-\mu)^2,\quad
\widehat x_j=(x_j-\mu)(v+\epsilon)^{-1/2},\quad y_j=\gamma_j\widehat x_j+\beta_j.
\]

With \(r=(v+\epsilon)^{-1/2}\) and \(h=\bar y\odot\gamma\),

\[
\bar x=r\left(h-\operatorname{mean}(h)-\widehat x\operatorname{mean}(h\odot\widehat x)\right),
\quad \bar\gamma=\sum\bar y\odot\widehat x,\quad \bar\beta=\sum\bar y.
\]

The last two sums span all non-feature dimensions. Reductions accumulate in stable working precision in the numerical implementation.

## 5. Rotary position embeddings

For even rotary width \(d_r\), frequency \(\omega_j=b^{-2j/d_r}\), position \(p\), and adjacent coordinate pair \((x_{2j},x_{2j+1})\), define

\[
\begin{pmatrix}x'_{2j}\\x'_{2j+1}\end{pmatrix}
=R(p\omega_j)\begin{pmatrix}x_{2j}\\x_{2j+1}\end{pmatrix},\qquad
R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}.
\]

Because \(R^TR=I\), the input gradient is \(\bar x=R^T\bar x'\). Moreover,

\[
(R(m\omega)q)^T(R(n\omega)k)=q^TR((n-m)\omega)k.
\]

Thus the dot product depends on relative displacement through the rotations. This implementation uses interleaved adjacent pairs consistently in training and decoding. Mixing interleaved and split-half conventions would silently corrupt cached inference. Position indices and the frequency base are fixed inputs, not trained parameters. [RoFormer](https://arxiv.org/abs/2104.09864) describes rotary position embeddings.

## 6. Multi-Head Latent Attention and paged decoding

For normalized input row \(u_t\), form compressed states

\[
c_t^{KV}=\operatorname{RMSNorm}(u_tW^{DKV}),\qquad
c_t^Q=\operatorname{RMSNorm}(u_tW^{DQ}).
\]

Per head \(h\),

\[
q_{th}^{C}=c_t^QW_h^{UQ,C},\quad q_{th}^{R}=R_t(c_t^QW_h^{UQ,R}),\quad
k_{sh}^{C}=c_s^{KV}W_h^{UK},\quad k_s^R=R_s(u_sW^{KR}),\quad
v_{sh}=c_s^{KV}W_h^{UV}.
\]

The rotary key is shared across heads. Attention scores combine content and rotary terms:

\[
s_{ths}=\frac{q_{th}^C\cdot k_{sh}^C+q_{th}^R\cdot k_s^R}{\sqrt{d_c+d_r}}.
\]

The projected head outputs are concatenated and multiplied by \(W^O\). Low-rank latent attention follows the construction motivating [DeepSeek-V2](https://arxiv.org/abs/2405.04434); this project uses a dense SwiGLU feed-forward block and is not a reproduction of its mixture-of-experts model.

At inference absorb the key up-projection into the query,

\[
\widetilde q_{th}=q_{th}^{C}(W_h^{UK})^T,
\qquad s_{ths}=(\widetilde q_{th}\cdot c_s^{KV}+q_{th}^{R}\cdot k_s^R)/\sqrt{d_c+d_r}.
\]

Accumulate latent context \(z_{th}=\sum_sP_{ths}c_s^{KV}\), then recover \(o_{th}=z_{th}W_h^{UV}\). Physical pages hold only \(c_s^{KV}\) and \(k_s^R\). The online recurrence above accumulates latent context directly across page boundaries. A logical token index \(s\) maps to page-table entry \(\lfloor s/B_p\rfloor\) and offset \(s\bmod B_p\). Reference counts permit shared prefix pages; mutation of a shared partial page first makes a private copy. This adapts the block-management principle described in [PagedAttention](https://arxiv.org/abs/2309.06180), without implementing its CUDA serving system.

## 7. SwiGLU, AdamW, sharding and alignment

The feed-forward function is

\[
F(x)=[\operatorname{SiLU}(xW_g)\odot(xW_u)]W_d,\quad
\operatorname{SiLU}(a)=a\sigma(a),\quad
\operatorname{SiLU}'(a)=\sigma(a)+a\sigma(a)(1-\sigma(a)).
\]

For each optimizer-owned parameter element,

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]
\[
\widehat m_t=m_t/(1-\beta_1^t),\quad \widehat v_t=v_t/(1-\beta_2^t),\quad
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\widehat m_t/(\sqrt{\widehat v_t}+\epsilon).
\]

ZeRO-style ownership partitions parameters, gradients, and moments; gathering a parameter for an operation and reduce-scattering its gradient leaves the persistent state sharded. Arithmetic is unchanged by ownership. The distinction between stages is defined in the [ZeRO paper](https://arxiv.org/abs/1910.02054). This project's coordinator and transient materialization buffers must still be included in a memory budget.

SFT sums only response-target log-probabilities and divides by the number of supervised tokens. For DPO let \(\ell_\theta^+\) and \(\ell_\theta^-\) be summed response log-probabilities of a preferred and rejected completion under identical prompt conditioning. With a frozen reference \(\pi_0\),

\[
z=\beta[(\ell_\theta^+-\ell_\theta^-)-(\ell_0^+-\ell_0^-)],\qquad
\mathcal L_{DPO}=\operatorname{softplus}(-z).
\]
\[
\frac{\partial\mathcal L}{\partial\ell_\theta^+}=-\beta\sigma(-z),\qquad
\frac{\partial\mathcal L}{\partial\ell_\theta^-}=\beta\sigma(-z).
\]

Reference scores are detached. Stable softplus avoids overflow. This is preference optimization as formulated by [Rafailov et al.](https://arxiv.org/abs/2305.18290); preference fit is not by itself evidence of factual reliability or safety.
