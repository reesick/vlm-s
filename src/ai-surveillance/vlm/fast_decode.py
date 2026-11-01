"""CUDA-graph greedy decoder over a static KV cache for Qwen2.5-VL.

Why: on this laptop (RTX 4050, Windows/WDDM, bitsandbytes NF4) HF `generate()`
is CPU-bound. One decode step is ~26 ms of GPU work but ~210 ms of Python and
~2000 individual kernel launches (measured with torch.profiler), i.e. ~5 tok/s.
Capturing one decode step in a CUDA graph and replaying it removes that host
overhead: measured 32.9 tok/s on the same model (6.3x), logits identical to
the HF path within fp16 noise. No torch.compile / triton needed.

How:
  * One `StaticCache` (fixed addresses, `max_cache_len` slots) is reused for
    every call; `reset()` only zeroes it in place.
  * Prefill (image + prompt) runs eagerly through the normal model forward,
    which also computes Qwen's M-RoPE `rope_deltas`.
  * The decode step (embed -> 36 layers -> lm_head -> repetition penalty ->
    argmax) reads/writes only static buffers: token id, 3-D position id
    (= cache index + rope_delta), additive attention mask row, a "seen
    tokens" mask for the penalty, and the cache itself. It is captured once.
  * `extend()` appends more text to the same cache (turn 2 of a conversation)
    without re-encoding the image: that is the cross-turn KV-cache reuse.
"""
from __future__ import annotations

import time
from typing import Iterable

import torch


class GraphDecoder:
    def __init__(self, model, eos_ids: Iterable[int], max_cache_len: int = 2048,
                 device: str | torch.device = "cuda:0"):
        from transformers import StaticCache

        self.model = model
        self.lm = model.model.language_model
        self.embed = self.lm.embed_tokens
        self.head = model.lm_head
        self.dev = torch.device(device)
        self.max_len = int(max_cache_len)
        self.eos = {int(e) for e in eos_ids}
        self.cache = StaticCache(config=model.config, max_cache_len=self.max_len)

        vocab = self.head.weight.shape[0] if hasattr(self.head, "weight") else model.config.vocab_size
        mdtype = self.embed.weight.dtype
        self._neg = torch.finfo(mdtype).min
        self.cur = torch.zeros((), dtype=torch.long, device=self.dev)      # cache index of the token fed next
        self.delta = torch.zeros((), dtype=torch.long, device=self.dev)    # M-RoPE delta from the image turn
        self.ids = torch.zeros((1, 1), dtype=torch.long, device=self.dev)  # token fed next / token just produced
        self.pos = torch.zeros((3, 1, 1), dtype=torch.long, device=self.dev)
        self.mask = torch.zeros((1, 1, 1, self.max_len), dtype=mdtype, device=self.dev)
        self.ar = torch.arange(self.max_len, device=self.dev)
        self.seen = torch.zeros(vocab, dtype=torch.bool, device=self.dev)
        self.penalty = torch.ones((), dtype=torch.float32, device=self.dev)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._evt = torch.cuda.Event()
        self.capture_s = 0.0

    # ------------------------------------------------------------------ core
    def _pick(self, logits_row: torch.Tensor) -> torch.Tensor:
        """Greedy pick with HF-style repetition penalty (penalty=1 -> plain argmax)."""
        lg = logits_row.float()
        p = self.penalty
        lg = torch.where(self.seen, torch.where(lg > 0, lg / p, lg * p), lg)
        nxt = lg.argmax(-1)
        # index_fill_ with a Python scalar: no host->device copy, so it is graph-capturable
        self.seen.index_fill_(0, nxt.view(1), True)
        return nxt

    def _step(self) -> None:
        """One decode step on static buffers only (this is what gets captured)."""
        self.mask.copy_(torch.where(self.ar <= self.cur, 0.0, self._neg).to(self.mask.dtype).view(1, 1, 1, -1))
        self.pos.copy_((self.cur + self.delta).view(1, 1, 1).expand(3, 1, 1))
        h = self.lm(inputs_embeds=self.embed(self.ids), position_ids=self.pos,
                    attention_mask={"full_attention": self.mask},
                    past_key_values=self.cache, use_cache=True).last_hidden_state
        self.ids.copy_(self._pick(self.head(h[:, -1, :])[0]).view(1, 1))
        self.cur.add_(1)

    def capture(self) -> None:
        """Capture the decode step. Needs an initialized cache (call prefill first)."""
        t0 = time.perf_counter()
        side = torch.cuda.Stream(device=self.dev)
        side.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(side):
            for _ in range(3):          # warm allocator + bnb kernels outside the graph
                self._step()
        torch.cuda.current_stream(self.dev).wait_stream(side)
        g = torch.cuda.CUDAGraph()
        # thread_local: the detector thread may launch CUDA work during capture
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            self._step()
        self.graph = g
        self.capture_s = time.perf_counter() - t0

    # ------------------------------------------------------------- public API
    def prefill(self, inputs: dict, penalty: float = 1.0) -> int:
        """Reset the cache and run image+prompt; returns the first new token id."""
        L = int(inputs["input_ids"].shape[1])
        if L >= self.max_len - 8:
            raise ValueError(f"prompt of {L} tokens does not fit static cache of {self.max_len}")
        self.cache.reset()
        self.model.model.rope_deltas = None     # force fresh M-RoPE index for this image
        out = self.model(**inputs, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        rd = self.model.model.rope_deltas
        self.delta.fill_(int(rd.view(-1)[0]) if rd is not None else 0)
        self.seen.zero_()
        self.seen[inputs["input_ids"][0]] = True
        self.penalty.fill_(float(penalty))
        self.cur.fill_(L)
        self.ids.copy_(self._pick(out.logits[0, -1]).view(1, 1))
        return int(self.ids)

    def extend(self, token_ids: torch.Tensor, penalty: float = 1.0) -> int:
        """Append text tokens to the cached conversation; returns the next token id.

        `token_ids` (1, n) must start with the last produced token (it was
        returned but never fed back into the cache).
        """
        n = int(token_ids.shape[1])
        start = int(self.cur)
        if start + n >= self.max_len - 8:
            raise ValueError(f"extension to {start + n} tokens exceeds static cache {self.max_len}")
        q = torch.arange(n, device=self.dev).view(n, 1) + start
        mask = torch.where(self.ar.view(1, -1) <= q, 0.0, self._neg).to(self.mask.dtype).view(1, 1, n, -1)
        pos = (torch.arange(n, device=self.dev) + start + self.delta).view(1, 1, n).expand(3, 1, n)
        h = self.lm(inputs_embeds=self.embed(token_ids.to(self.dev)), position_ids=pos,
                    attention_mask={"full_attention": mask},
                    past_key_values=self.cache, use_cache=True).last_hidden_state
        self.seen[token_ids[0].to(self.dev)] = True
        self.penalty.fill_(float(penalty))
        self.cur.fill_(start + n)
        self.ids.copy_(self._pick(self.head(h[:, -1, :])[0]).view(1, 1))
        return int(self.ids)

    def decode(self, first_token: int, max_new_tokens: int) -> list[int]:
        """Greedy-decode until EOS or max_new_tokens (first_token counts as #1)."""
        out = [int(first_token)]
        budget = min(int(max_new_tokens), self.max_len - 2 - int(self.cur))
        if self.graph is None and out[-1] not in self.eos and budget > 1:
            self.capture()
            # capture ran 4 steps on live state: rewind is not possible, so the
            # caller's state is now advanced. This only happens if warmup()
            # was skipped; signal it so the caller can redo the call.
            raise RuntimeError("graph captured mid-call; retry")
        while len(out) < budget and out[-1] not in self.eos:
            self.graph.replay()
            # Wait for the step with the GIL released (sleep), not with a
            # blocking .item(): the camera/detector thread must keep running.
            self._evt.record()
            while not self._evt.query():
                time.sleep(0.0005)
            out.append(int(self.ids))
        return out

    def warmup(self, text_inputs: dict) -> None:
        """Initialize the cache and capture the graph on a throwaway prompt."""
        self.prefill(text_inputs)
        self.capture()
        self.cache.reset()
