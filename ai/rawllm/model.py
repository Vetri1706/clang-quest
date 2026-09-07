"""Dense MLA/SwiGLU decoder with explicit parameter ownership and NumPy inference."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .tensor import Tensor, Parameter, concatenate, embedding, rms_norm, rope, attention


@dataclass(frozen=True)
class Config:
    vocab_size: int = 320
    dim: int = 32
    layers: int = 2
    heads: int = 4
    q_rank: int = 16
    kv_rank: int = 12
    content_dim: int = 8
    rope_dim: int = 8
    value_dim: int = 8
    hidden_dim: int = 64
    max_seq_len: int = 128
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self):
        dimensions = [self.vocab_size, self.dim, self.layers, self.heads, self.q_rank,
                      self.kv_rank, self.content_dim, self.rope_dim, self.value_dim,
                      self.hidden_dim, self.max_seq_len]
        if any(type(n) is not int or n < 1 for n in dimensions):
            raise ValueError("All architecture dimensions must be positive integers.")
        if type(self.tie_embeddings) is not bool:
            raise ValueError("tie_embeddings must be a boolean.")
        if self.rope_dim % 2 or not np.isfinite(self.rope_base) or not np.isfinite(self.norm_eps) or self.rope_base <= 0 or self.norm_eps <= 0:
            raise ValueError("RoPE width must be even, with positive base and norm epsilon.")

    @classmethod
    def seven_b(cls):
        return cls(vocab_size=32000, dim=4096, layers=32, heads=32, q_rank=1536,
                   kv_rank=512, content_dim=128, rope_dim=64, value_dim=128,
                   hidden_dim=14336, max_seq_len=32768)

    def shapes(self):
        d, h = self.dim, self.heads
        result = {"embedding": (self.vocab_size, d)}
        for i in range(self.layers):
            prefix = f"blocks.{i}."
            block = {"attn_norm": (d,), "ffn_norm": (d,),
                     "q_down": (d, self.q_rank), "q_norm": (self.q_rank,),
                     "q_up": (self.q_rank, h * (self.content_dim + self.rope_dim)),
                     "kv_down": (d, self.kv_rank), "kv_norm": (self.kv_rank,),
                     "k_up": (self.kv_rank, h * self.content_dim),
                     "v_up": (self.kv_rank, h * self.value_dim),
                     "k_rope": (d, self.rope_dim),
                     "o": (h * self.value_dim, d),
                     "gate": (d, self.hidden_dim), "up": (d, self.hidden_dim),
                     "down": (self.hidden_dim, d)}
            result.update({prefix + name: shape for name, shape in block.items()})
        result["final_norm"] = (d,)
        if not self.tie_embeddings:
            result["head"] = (d, self.vocab_size)
        return result

    @property
    def parameter_count(self):
        return sum(math.prod(shape) for shape in self.shapes().values())


class Transformer:
    def __init__(self, config=None, seed=0, dtype=np.float32, max_parameter_bytes=128 * 1024**2):
        self.config = config or Config()
        if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("Model storage must be float32 or float64; select reduced precision through emulation.")
        expected = self.config.parameter_count * np.dtype(dtype).itemsize
        if expected > max_parameter_bytes:
            raise MemoryError(f"Parameter allocation {expected:,} bytes exceeds limit {max_parameter_bytes:,}; use estimate.py for large configurations.")
        rng = np.random.default_rng(seed)
        self.params = {}
        for name, shape in self.config.shapes().items():
            value = np.ones(shape, dtype=dtype) if len(shape) == 1 else rng.normal(0, .02, shape).astype(dtype)
            self.params[name] = Parameter(value, name=name)
        self.precision = "fp32"

    def parameters(self):
        return self.params

    def weight(self, name):
        parameter = self.params[name]
        if self.precision == "fp32":
            return parameter
        from .optim import quantize_ste
        rounded = quantize_ste(parameter, self.precision)
        if not np.isfinite(rounded.data).all():
            raise FloatingPointError("Parameter overflow in emulated precision; loss scaling cannot repair a forward overflow.")
        return rounded

    def quant(self, value):
        if self.precision == "fp32":
            return value
        from .optim import quantize_ste
        rounded = quantize_ste(value, self.precision)
        if not np.isfinite(rounded.data).all():
            raise FloatingPointError("Activation overflow in emulated precision.")
        return rounded

    def block(self, x, index, mask=None, positions=None):
        c = self.config
        p = f"blocks.{index}."
        w = lambda key: self.weight(p + key)
        project = lambda a, key: self.quant(self.quant(a) @ w(key))
        batch, length, _ = x.shape
        u = rms_norm(x, w("attn_norm"), c.norm_eps)
        q_latent = rms_norm(project(u, "q_down"), w("q_norm"), c.norm_eps)
        q = project(q_latent, "q_up").reshape(batch, length, c.heads, c.content_dim + c.rope_dim).transpose(0, 2, 1, 3)
        q_content = q[..., :c.content_dim]
        q_rotary = rope(q[..., c.content_dim:], positions, c.rope_base)
        latent = rms_norm(project(u, "kv_down"), w("kv_norm"), c.norm_eps)
        k_content = project(latent, "k_up").reshape(batch, length, c.heads, c.content_dim).transpose(0, 2, 1, 3)
        k_rotary = rope(project(u, "k_rope").reshape(batch, 1, length, c.rope_dim), positions, c.rope_base)
        # One shared rotary key per token; multiplication broadcasts it across heads.
        k_rotary = k_rotary * np.ones((1, c.heads, 1, 1), dtype=x.data.dtype)
        values = project(latent, "v_up").reshape(batch, length, c.heads, c.value_dim).transpose(0, 2, 1, 3)
        attended = attention(concatenate((q_content, q_rotary), axis=-1),
                             concatenate((k_content, k_rotary), axis=-1), values,
                             causal=True, mask=mask, block_size=32)
        x = x + project(attended.transpose(0, 2, 1, 3).reshape(batch, length, c.heads * c.value_dim), "o")
        u = rms_norm(x, w("ffn_norm"), c.norm_eps)
        return x + project(project(u, "gate").silu() * project(u, "up"), "down")

    def __call__(self, token_ids, attention_mask=None):
        ids = np.asarray(token_ids)
        c = self.config
        if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Token ids must be an integer [batch, time] array.")
        if not 1 <= ids.shape[1] <= c.max_seq_len or np.any(ids < 0) or np.any(ids >= c.vocab_size):
            raise ValueError("Invalid sequence length or token id.")
        x = embedding(self.weight("embedding"), ids)
        positions = np.arange(ids.shape[1])
        for index in range(c.layers):
            x = self.block(x, index, attention_mask, positions)
        x = rms_norm(x, self.weight("final_norm"), c.norm_eps)
        head = self.weight("embedding").transpose(1, 0) if c.tie_embeddings else self.weight("head")
        return self.quant(x) @ head

    def decode(self, token_id, cache, sequence_id="default"):
        """One token, one sequence, using compressed latent pages without autograd."""
        from .cache import paged_latent_attention
        c = self.config
        if self.precision != "fp32":
            raise ValueError("Paged decoding currently requires fp32; emulated precision is available in full-sequence execution.")
        if cache.config != c:
            raise ValueError("Cache and model configurations differ.")
        if not 0 <= token_id < c.vocab_size:
            raise ValueError("Token id outside vocabulary.")
        position = cache.length(sequence_id)
        if position >= c.max_seq_len:
            raise ValueError("Maximum sequence length reached.")
        mark = cache.mark(sequence_id)
        def norm(a, weight):
            return a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + c.norm_eps) * weight
        def rotate(a):
            frequencies = c.rope_base ** (-np.arange(0, c.rope_dim, 2) / c.rope_dim)
            angle = position * frequencies
            even, odd = a[..., 0::2], a[..., 1::2]
            result = np.empty_like(a)
            result[..., 0::2] = even * np.cos(angle) - odd * np.sin(angle)
            result[..., 1::2] = even * np.sin(angle) + odd * np.cos(angle)
            return result
        try:
            slot = cache.reserve(sequence_id)
            x = self.params["embedding"].data[token_id].copy()
            for index in range(c.layers):
                p = f"blocks.{index}."
                w = lambda key: self.params[p + key].data
                u = norm(x, w("attn_norm"))
                q_latent = norm(u @ w("q_down"), w("q_norm"))
                q = (q_latent @ w("q_up")).reshape(c.heads, c.content_dim + c.rope_dim)
                q_content, q_rotary = q[:, :c.content_dim], rotate(q[:, c.content_dim:])
                latent = norm(u @ w("kv_down"), w("kv_norm"))
                k_rotary = rotate(u @ w("k_rope"))
                cache.write(index, slot, latent, k_rotary)
                result = paged_latent_attention(q_content, q_rotary, w("k_up"), w("v_up"), cache, sequence_id, index)
                x = x + result.reshape(c.heads * c.value_dim) @ w("o")
                u = norm(x, w("ffn_norm"))
                gate = u @ w("gate")
                sigmoid = np.exp(-np.logaddexp(0, -gate))
                x = x + ((gate * sigmoid) * (u @ w("up"))) @ w("down")
            x = norm(x, self.params["final_norm"].data)
            head = self.params["embedding"].data.T if c.tie_embeddings else self.params["head"].data
            return x @ head
        except BaseException:
            cache.rollback(sequence_id, mark)
            raise
