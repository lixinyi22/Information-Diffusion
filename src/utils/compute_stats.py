"""Parameter counts and per-forward GFLOPs estimates for MCID (torch.profiler)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch


def parameter_stats(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def _sum_profiler_flops(prof: torch.profiler.profile) -> int:
    total = 0
    for evt in prof.key_averages():
        if evt.flops:
            total += int(evt.flops)
    return total


def measure_forward_gflops(
    model: torch.nn.Module,
    forward_fn: Callable[[], Any],
    device: torch.device,
    warmup: int = 2,
) -> Optional[float]:
    """
    One forward-pass GFLOPs (1e9 FLOPs) from PyTorch profiler.
    Sparse / scatter ops may be missing or partial; treat as approximate.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    model.zero_grad(set_to_none=True)
    for _ in range(max(0, warmup)):
        forward_fn()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
        record_shapes=False,
    ) as prof:
        forward_fn()
    torch.cuda.synchronize()
    flops = _sum_profiler_flops(prof)
    if flops <= 0:
        return None
    return flops / 1e9


def make_profile_dummy_batch(
    opt: Any,
    content_emb_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensors aligned with DataConstruct padding length opt.max_len + 1."""
    b = max(1, min(2, int(opt.batch_size)))
    seq_len = int(opt.max_len) + 1
    hi_user = max(3, int(opt.user_size))
    hi_info = max(1, int(opt.info_size))
    tgt = torch.randint(2, hi_user, (b, seq_len), dtype=torch.long, device=device)
    tgt_timestamp = torch.rand(b, seq_len, dtype=torch.float32, device=device)
    tgt_id = torch.randint(0, hi_info, (b,), dtype=torch.long, device=device)
    content_vecs = torch.randn(b, content_emb_dim, dtype=torch.float32, device=device)
    return tgt, tgt_timestamp, tgt_id, content_vecs


def log_parameter_and_flops(
    model: torch.nn.Module,
    opt: Any,
    diffusion_graph: torch.Tensor,
    content_emb_dim: int,
) -> Dict[str, Any]:
    """Log params + train/inference forward GFLOPs once; returns dict for run.py to merge into log summary."""
    stats = parameter_stats(model)
    logging.info(
        "[Profile] Parameters: total=%.4fM (%d), trainable=%.4fM (%d)",
        stats["total_params"] / 1e6,
        stats["total_params"],
        stats["trainable_params"] / 1e6,
        stats["trainable_params"],
    )

    device = next(model.parameters()).device
    if device.type != "cuda":
        logging.info("[Profile] Skip GFLOPs profiling (CUDA not used).")
        return {
            **stats,
            "gflops_train_forward": None,
            "gflops_inference_forward": None,
            "gflops_note": "Profiling skipped: model not on CUDA.",
        }

    tgt, tgt_timestamp, tgt_id, content_vecs = make_profile_dummy_batch(
        opt, content_emb_dim, device
    )
    dg = diffusion_graph
    if dg.device != device:
        dg = dg.to(device)

    def train_forward():
        model.train()
        pred, _ = model(tgt, tgt_timestamp, tgt_id, dg)
        return pred

    def infer_forward():
        model.eval()
        with torch.no_grad():
            pred, _ = model.inference(tgt, tgt_timestamp, content_vecs, dg)
        return pred

    g_train = None
    g_infer = None
    try:
        g_train = measure_forward_gflops(model, train_forward, device)
    except Exception as e:
        logging.warning("[Profile] Train-forward GFLOPs failed: %s", e)
    try:
        g_infer = measure_forward_gflops(model, infer_forward, device)
    except Exception as e:
        logging.warning("[Profile] Inference-forward GFLOPs failed: %s", e)

    if g_train is not None:
        logging.info("[Profile] Train forward (one pass, approximate): %.4f GFLOPs", g_train)
    else:
        logging.info(
            "[Profile] Train forward GFLOPs unavailable (profiler returned 0 or error)."
        )
    if g_infer is not None:
        logging.info(
            "[Profile] Inference forward (one pass, approximate): %.4f GFLOPs", g_infer
        )
    else:
        logging.info(
            "[Profile] Inference forward GFLOPs unavailable (profiler returned 0 or error)."
        )
    logging.info(
        "[Profile] GFLOPs are approximate; sparse matmul / custom ops may be under-counted."
    )

    return {
        **stats,
        "gflops_train_forward": g_train,
        "gflops_inference_forward": g_infer,
        "gflops_note": "PyTorch profiler forward pass; sparse ops may be incomplete.",
    }
