"""Fixed-size physical pages for latent KV state and online paged attention."""
import numpy as np


class PagedLatentCache:
    def __init__(self, config, page_size=8, max_pages=32, dtype=np.float32, max_bytes=64 * 1024**2):
        if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("Cache storage must be float32 or float64.")
        if not isinstance(page_size, int) or not isinstance(max_pages, int) or page_size < 1 or max_pages < 1:
            raise ValueError("Page size and page count must be positive.")
        size = config.layers * max_pages * page_size * (config.kv_rank + config.rope_dim) * np.dtype(dtype).itemsize
        if size > max_bytes:
            raise MemoryError("Cache allocation exceeds its explicit byte limit.")
        self.config, self.page_size, self.max_pages = config, page_size, max_pages
        self.latent = np.zeros((config.layers, max_pages, page_size, config.kv_rank), dtype=dtype)
        self.rotary = np.zeros((config.layers, max_pages, page_size, config.rope_dim), dtype=dtype)
        self.refs = np.zeros(max_pages, dtype=np.int64)
        self.free = list(range(max_pages - 1, -1, -1))
        self.tables = {}
        self.lengths = {}

    @property
    def allocated_bytes(self):
        return self.latent.nbytes + self.rotary.nbytes + self.refs.nbytes

    def length(self, sequence_id):
        return self.lengths.get(sequence_id, 0)

    def _allocate(self):
        if not self.free:
            raise MemoryError("Paged cache exhausted; release a sequence or allocate a larger pool.")
        page = self.free.pop()
        self.refs[page] = 1
        return page

    def _drop(self, page):
        self.refs[page] -= 1
        if self.refs[page] == 0:
            self.latent[:, page] = 0
            self.rotary[:, page] = 0
            self.free.append(page)

    def reserve(self, sequence_id):
        n = self.length(sequence_id)
        table = self.tables.setdefault(sequence_id, [])
        offset = n % self.page_size
        if offset == 0:
            page = self._allocate()
            table.append(page)
        else:
            page = table[-1]
            if self.refs[page] > 1:
                replacement = self._allocate()
                self.latent[:, replacement] = self.latent[:, page]
                self.rotary[:, replacement] = self.rotary[:, page]
                self._drop(page)
                table[-1] = page = replacement
        self.lengths[sequence_id] = n + 1
        return page, offset

    def mark(self, sequence_id):
        return (list(self.tables.get(sequence_id, [])), self.length(sequence_id), sequence_id in self.tables)

    def rollback(self, sequence_id, mark):
        previous, length, existed = mark
        current = self.tables.get(sequence_id, [])
        for page in set(current) - set(previous):
            self._drop(page)
        for page in set(previous) - set(current):
            self.refs[page] += 1
        if existed:
            self.tables[sequence_id] = previous
            self.lengths[sequence_id] = length
        else:
            self.tables.pop(sequence_id, None)
            self.lengths.pop(sequence_id, None)

    def write(self, layer, slot, latent, rotary):
        page, offset = slot
        self.latent[layer, page, offset] = latent
        self.rotary[layer, page, offset] = rotary

    def pages(self, sequence_id, layer):
        remaining = self.length(sequence_id)
        for page in self.tables.get(sequence_id, []):
            count = min(remaining, self.page_size)
            yield self.latent[layer, page, :count], self.rotary[layer, page, :count]
            remaining -= count

    def fork(self, source, destination):
        if source not in self.tables or destination in self.tables:
            raise ValueError("Fork requires an existing source and a new destination.")
        self.tables[destination] = list(self.tables[source])
        self.lengths[destination] = self.length(source)
        for page in self.tables[destination]:
            self.refs[page] += 1

    def pop(self, sequence_id):
        n = self.length(sequence_id)
        if n == 0:
            raise ValueError("Cannot pop an empty cache sequence.")
        if (n - 1) % self.page_size == 0:
            self._drop(self.tables[sequence_id].pop())
        self.lengths[sequence_id] = n - 1

    def release(self, sequence_id):
        for page in self.tables.pop(sequence_id, []):
            self._drop(page)
        self.lengths.pop(sequence_id, None)


def paged_latent_attention(q_content, q_rotary, k_up, v_up, cache, sequence_id, layer):
    """Read physical pages directly; no contiguous full-sequence KV reconstruction."""
    c = cache.config
    absorbed_query = np.einsum("hd,rhd->hr", q_content, k_up.reshape(c.kv_rank, c.heads, c.content_dim))
    max_score = np.full(c.heads, -np.inf)
    denominator = np.zeros(c.heads)
    accumulator = np.zeros((c.heads, c.kv_rank))
    scale = 1 / np.sqrt(c.content_dim + c.rope_dim)
    for latent, rotary in cache.pages(sequence_id, layer):
        scores = (absorbed_query @ latent.T + q_rotary @ rotary.T) * scale
        new_max = np.maximum(max_score, scores.max(axis=-1))
        correction = np.exp(max_score - new_max)
        probabilities = np.exp(scores - new_max[:, None])
        accumulator = correction[:, None] * accumulator + probabilities @ latent
        denominator = correction * denominator + probabilities.sum(axis=-1)
        max_score = new_max
    if np.any(denominator == 0):
        raise ValueError("Attention requires at least one cached token.")
    latent_context = accumulator / denominator[:, None]
    return np.einsum("hr,rhv->hv", latent_context, v_up.reshape(c.kv_rank, c.heads, c.value_dim))
