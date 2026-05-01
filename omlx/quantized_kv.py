# SPDX-License-Identifier: Apache-2.0
"""Batch-compatible affine quantized KV cache helpers."""

from __future__ import annotations

from typing import List

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce
from mlx_lm.models.cache import (
    KVCache,
    QuantizedKVCache,
    create_attention_mask,
    dynamic_roll,
)


def _time_len(state) -> int:
    return 0 if state is None else state[0].shape[2]


def _slice_time(state, start: int, end: int):
    return tree_map(lambda x: x[..., start:end, :], state)


def _pad_time(state, left: int, right: int = 0):
    if left == 0 and right == 0:
        return state
    return tree_map(lambda x: mx.pad(x, [(0, 0), (0, 0), (left, right), (0, 0)]), state)


def _zeros_like_time(state, batch: int, length: int):
    return tree_map(
        lambda x: mx.zeros((batch, x.shape[1], length, x.shape[3]), dtype=x.dtype),
        state,
    )


def _concat_batch(states):
    return tree_map(lambda *xs: mx.concatenate(xs, axis=0), *states)


class AffineQuantizedKVCache(QuantizedKVCache):
    """Affine q4 KV cache with mlx-lm batch merge support."""

    def __init__(self, group_size: int = 64, bits: int = 4):
        super().__init__(group_size=group_size, bits=bits)

    def size(self):
        return self.offset

    @classmethod
    def from_cache(
        cls,
        cache: KVCache,
        group_size: int = 64,
        bits: int = 4,
    ) -> "AffineQuantizedKVCache":
        quant_cache = cls(group_size=group_size, bits=bits)
        quant_cache.offset = cache.offset
        if cache.keys is not None:
            keys, values = cache.state
            quant_cache.keys = mx.quantize(keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(values, group_size=group_size, bits=bits)
        return quant_cache

    @classmethod
    def merge(cls, caches):
        return BatchQuantizedKVCache.merge(caches)


class BatchQuantizedKVCache(QuantizedKVCache):
    """Batched affine quantized KV cache matching mlx-lm BatchKVCache behavior."""

    step = 256

    def __init__(
        self,
        left_padding: List[int],
        group_size: int = 64,
        bits: int = 4,
    ):
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._right_padding = None
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys, values):
        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        num_steps = keys.shape[2]
        prev = self._idx

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step

            def init_quant(q):
                return tree_map(
                    lambda x: mx.zeros(
                        (*x.shape[:2], new_steps, x.shape[-1]), dtype=x.dtype
                    ),
                    q,
                )

            def expand_quant(x):
                new_x = mx.zeros((*x.shape[:2], new_steps, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )
                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys, self.values = init_quant(q_keys), init_quant(q_values)

        self.offset += num_steps
        self._idx += num_steps
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self._idx, :] = q_keys[i]
            self.values[i][..., prev : self._idx, :] = q_values[i]

        return tree_map(lambda x: x[..., : self._idx, :], (self.keys, self.values))

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchQuantizedKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self.keys = tree_map(
                lambda x: dynamic_roll(x, padding[:, None], axis=2), self.keys
            )
            self.values = tree_map(
                lambda x: dynamic_roll(x, padding[:, None], axis=2), self.values
            )
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        if self.keys is None:
            return self.keys, self.values
        if self._idx == self.keys[0].shape[2]:
            return self.keys, self.values
        return tree_map(lambda x: x[..., : self._idx, :], (self.keys, self.values))

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self._idx = _time_len(self.keys)
        self.offset += self._idx

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def filter(self, batch_indices):
        self.keys = tree_map(lambda x: x[batch_indices], self.keys)
        self.values = tree_map(lambda x: x[batch_indices], self.values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            self.keys = tree_map(lambda x: x[..., min_left_pad:, :], self.keys)
            self.values = tree_map(lambda x: x[..., min_left_pad:, :], self.values)
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        max_idx = max(self._idx, other._idx)
        max_size = max(self.keys[0].shape[2], other.keys[0].shape[2])

        def pad(c):
            left = max_idx - c._idx
            right = max_size - c.keys[0].shape[2] - left
            keys, values = c.keys, c.values
            if right < 0:
                keys, values = tree_map(lambda x: x[..., :right, :], (keys, values))
                right = 0
            keys, values = _pad_time((keys, values), left, right)
            return keys, values, c.offset, c.left_padding + left

        self_keys, self_values, self_offset, self_left_padding = pad(self)
        other_keys, other_values, other_offset, other_left_padding = pad(other)
        self.keys = _concat_batch([self_keys, other_keys])
        self.values = _concat_batch([self_values, other_values])
        self.offset = mx.concatenate([self_offset, other_offset])
        self.left_padding = mx.concatenate([self_left_padding, other_left_padding])
        self._idx = max_idx

    def extract(self, idx):
        cache = AffineQuantizedKVCache(group_size=self.group_size, bits=self.bits)
        padding = self.left_padding[idx].item()
        cache.keys = tree_map(
            lambda x: mx.contiguous(x[idx : idx + 1, :, padding : self._idx]),
            self.keys,
        )
        cache.values = tree_map(
            lambda x: mx.contiguous(x[idx : idx + 1, :, padding : self._idx]),
            self.values,
        )
        cache.offset = cache.keys[0].shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        group_size = getattr(caches[0], "group_size", 64)
        bits = getattr(caches[0], "bits", 4)
        lengths = [getattr(c, "offset", 0) for c in caches]
        max_length = max(lengths) if lengths else 0
        if max_length == 0:
            return cls([0] * len(caches), group_size=group_size, bits=bits)

        template_keys, template_values = next(c.state for c in caches if c.keys is not None)
        keys = []
        values = []
        padding = [max_length - int(l) for l in lengths]
        for left, length, cache in zip(padding, lengths, caches):
            if cache.keys is None:
                state = (
                    _zeros_like_time(template_keys, 1, max_length),
                    _zeros_like_time(template_values, 1, max_length),
                )
            else:
                state = _slice_time((cache.keys, cache.values), 0, int(length))
                state = _pad_time(state, left)
            keys.append(state[0])
            values.append(state[1])

        cache = cls(padding, group_size=group_size, bits=bits)
        cache.keys = _concat_batch(keys)
        cache.values = _concat_batch(values)
        cache.offset += max_length
        cache._idx = max_length
        return cache


def install_quantized_kv_merge_patch() -> None:
    QuantizedKVCache.merge = BatchQuantizedKVCache.merge
    QuantizedKVCache.size = lambda self: self.offset
