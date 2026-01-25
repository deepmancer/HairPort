"""
Paired with a good language model. Thanks!
"""

import torch
from typing import Optional, Tuple
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

from kernels import get_kernel
_k = get_kernel("kernels-community/vllm-flash-attn3")
_flash_attn_func = _k.flash_attn_func


def _ensure_fa3_available():
    if _flash_attn_func is None:
        raise ImportError(
            "FlashAttention-3 via Hugging Face `kernels` is required. "
            "Tried `get_kernel('kernels-community/vllm-flash-attn3')` and failed with:\n"
            f"{_kernels_err}"
        )

@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    outputs, lse = _flash_attn_func(q, k, v, causal=causal)
    return outputs

@flash_attn_func.register_fake
def _(q, k, v, **kwargs):
    # two outputs:
    # 1. output: (batch, seq_len, num_heads, head_dim)
    # 2. softmax_lse: (batch, num_heads, seq_len) with dtype=torch.float32
    meta_q = torch.empty_like(q).contiguous()
    return meta_q #, q.new_empty((q.size(0), q.size(2), q.size(1)), dtype=torch.float32)


class QwenDoubleStreamAttnProcessorFA3:
    """
    FA3-based attention processor for Qwen double-stream architecture.
    Computes joint attention over concatenated [text, image] streams using vLLM FlashAttention-3
    accessed via Hugging Face `kernels`.
    Notes / limitations:
    - General attention masks are not supported here (FA3 path). `is_causal=False` and no arbitrary mask.
    - Optional windowed attention / sink tokens / softcap can be plumbed through if you use those features.
    - Expects an available `apply_rotary_emb_qwen` in scope (same as your non-FA3 processor).
    """

    _attention_backend = "fa3"  # for parity with your other processors, not used internally

    def __init__(self):
        _ensure_fa3_available()

    @torch.no_grad()
    def __call__(
        self,
        attn,  # Attention module with to_q/to_k/to_v/add_*_proj, norms, to_out, to_add_out, and .heads
        hidden_states: torch.FloatTensor,                 # (B, S_img, D_model)  image stream
        encoder_hidden_states: torch.FloatTensor = None,  # (B, S_txt, D_model)  text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,  # unused in FA3 path
        attention_mask: Optional[torch.FloatTensor] = None,    # unused in FA3 path
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (img_freqs, txt_freqs)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessorFA3 requires encoder_hidden_states (text stream).")
        if attention_mask is not None:
            # FA3 kernel path here does not consume arbitrary masks; fail fast to avoid silent correctness issues.
            raise NotImplementedError("attention_mask is not supported in this FA3 implementation.")

        _ensure_fa3_available()

        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]

        # ---- QKV projections (image/sample stream) ----
        img_q = attn.to_q(hidden_states)   # (B, S_img, D)
        img_k = attn.to_k(hidden_states)
        img_v = attn.to_v(hidden_states)

        # ---- QKV projections (text/context stream) ----
        txt_q = attn.add_q_proj(encoder_hidden_states)  # (B, S_txt, D)
        txt_k = attn.add_k_proj(encoder_hidden_states)
        txt_v = attn.add_v_proj(encoder_hidden_states)

        # ---- Reshape to (B, S, H, D_h) ----
        H = attn.heads
        img_q = img_q.unflatten(-1, (H, -1))
        img_k = img_k.unflatten(-1, (H, -1))
        img_v = img_v.unflatten(-1, (H, -1))

        txt_q = txt_q.unflatten(-1, (H, -1))
        txt_k = txt_k.unflatten(-1, (H, -1))
        txt_v = txt_v.unflatten(-1, (H, -1))

        # ---- Q/K normalization (per your module contract) ----
        if getattr(attn, "norm_q", None) is not None:
            img_q = attn.norm_q(img_q)
        if getattr(attn, "norm_k", None) is not None:
            img_k = attn.norm_k(img_k)
        if getattr(attn, "norm_added_q", None) is not None:
            txt_q = attn.norm_added_q(txt_q)
        if getattr(attn, "norm_added_k", None) is not None:
            txt_k = attn.norm_added_k(txt_k)

        # ---- RoPE (Qwen variant) ----
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            # expects tensors shaped (B, S, H, D_h)
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, use_real=False)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, use_real=False)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, use_real=False)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, use_real=False)

        # ---- Joint attention over [text, image] along sequence axis ----
        # Shapes: (B, S_total, H, D_h)
        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)

        # FlashAttention-3 path expects (B, S, H, D_h) and returns (out, softmax_lse)
        out = flash_attn_func(q, k, v, causal=False)  # out: (B, S_total, H, D_h)

        # ---- Back to (B, S, D_model) ----
        out = out.flatten(2, 3).to(q.dtype)

        # Split back to text / image segments
        txt_attn_out = out[:, :S_txt, :]
        img_attn_out = out[:, S_txt:, :]

        # ---- Output projections ----
        img_attn_out = attn.to_out[0](img_attn_out)
        if len(attn.to_out) > 1:
            img_attn_out = attn.to_out[1](img_attn_out)  # dropout if present

        txt_attn_out = attn.to_add_out(txt_attn_out)

        return img_attn_out, txt_attn_out