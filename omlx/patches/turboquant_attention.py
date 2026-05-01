# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to support TurboQuantKVCache.

When TurboQuantKVCache is detected, routes attention to:
  - Decode (L=1): cache.decode_attention() — Metal kernel, no dequant
  - Prefill (L>1): cache.prefill_attention() fast path, fallback to
    dequantize + mx.fast.scaled_dot_product_attention
"""

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False


def _is_quantized_tuple(value) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(v, mx.array) for v in value)
    )


def _is_affine_quantized_cache(cache) -> bool:
    try:
        from mlx_lm.models.cache import QuantizedKVCache
        from ..quantized_kv import AffineQuantizedKVCache, BatchQuantizedKVCache
    except ImportError:
        return False

    return isinstance(
        cache, (QuantizedKVCache, AffineQuantizedKVCache, BatchQuantizedKVCache)
    )


def _affine_quantized_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array],
):
    real_cache = cache
    if hasattr(cache, "_cache") and not _is_affine_quantized_cache(cache):
        real_cache = cache._cache

    if not _is_affine_quantized_cache(real_cache):
        return None
    if not (_is_quantized_tuple(keys) and _is_quantized_tuple(values)):
        return None
    if sinks is not None:
        raise ValueError("Affine quantized SDPA does not support attention sinks.")
    if not hasattr(mx.fast, "quantized_scaled_dot_product_attention"):
        raise RuntimeError(
            "Affine q4 KV cache requires mlx.fast.quantized_scaled_dot_product_attention"
        )
    if queries.shape[-2] > 8:
        if queries.shape[-1] != 256:
            raise RuntimeError(
                "Affine q4 long prefill currently requires head_dim=256 for the fused MLX path."
            )
        if mask is not None and not isinstance(mask, str):
            raise RuntimeError(
                "Affine q4 long prefill requires a causal or empty mask for the fused MLX path."
            )

    return mx.fast.quantized_scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
        group_size=real_cache.group_size,
        bits=real_cache.bits,
    )


def apply_turboquant_attention_patch() -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for TurboQuant."""
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    original_sdpa = mlx_base.scaled_dot_product_attention

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: Optional[mx.array],
        sinks: Optional[mx.array] = None,
    ) -> mx.array:
        from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache
        from ..turboquant_kv import BatchTurboQuantKVCache

        affine_result = _affine_quantized_attention(
            queries, keys, values, cache, scale, mask, sinks
        )
        if affine_result is not None:
            return affine_result

        # Detect underlying TQ cache (may be wrapped by proxy objects)
        real_cache = cache
        if hasattr(cache, "_cache") and not isinstance(
            cache, (_TQCache, BatchTurboQuantKVCache)
        ):
            real_cache = cache._cache

        if isinstance(real_cache, (_TQCache, BatchTurboQuantKVCache)):
            if queries.shape[-2] == 1:
                return real_cache.decode_attention(
                    queries,
                    keys_state=keys,
                    values_state=values,
                    scale=scale,
                    mask=mask,
                )
            # Prefill: try quantized fast path, fallback to dequantize+SDPA
            result = real_cache.prefill_attention(
                queries, scale=scale, mask=mask,
            )
            if result is not None:
                return result
            dequantized_keys, dequantized_values = real_cache.dequantize()
            return mx.fast.scaled_dot_product_attention(
                queries,
                dequantized_keys.astype(queries.dtype),
                dequantized_values.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )

        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Patch the module attribute
    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Also patch any model modules that already imported it locally
    # Covers both mlx_lm (LLM) and mlx_vlm (VLM) model modules
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models.")):
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            func = getattr(mod, "scaled_dot_product_attention")
            if func is original_sdpa or func is not patched_sdpa:
                setattr(mod, "scaled_dot_product_attention", patched_sdpa)

    # Also patch mlx_vlm.models.base if loaded
    try:
        from mlx_vlm.models import base as vlm_base
        if hasattr(vlm_base, "scaled_dot_product_attention"):
            vlm_base.scaled_dot_product_attention = patched_sdpa
    except ImportError:
        pass

    _PATCHED = True
    logger.info("TurboQuant attention patch applied")
    return True
