from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


def save_old_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Save trainable parameters of old model snapshot on CPU."""
    model = unwrap_model(model)
    params = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    LOGGER.info(f"[EWC] saved old_params: {len(params)} tensors")
    return params


def compute_fisher(
    model: torch.nn.Module,
    dataloader,
    trainer,
    device: torch.device | str,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    """Estimate diagonal Fisher matrix by averaging squared gradients on old-task batches."""
    model = unwrap_model(model)
    model.train()
    dev = torch.device(device)

    fisher = {
        n: torch.zeros_like(p, dtype=torch.float32, device="cpu")
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    processed_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and processed_batches >= max_batches:
            break

        batch = trainer.preprocess_batch(batch)
        if batch["img"].device != dev:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(dev, non_blocking=True)

        model.zero_grad(set_to_none=True)
        loss, _ = model(batch)
        loss_scalar = loss.sum() if loss.ndim else loss
        if not torch.isfinite(loss_scalar).all():
            LOGGER.warning(f"[EWC] skip fisher batch {batch_idx}: non-finite loss={float(loss_scalar.detach())}")
            continue

        loss_scalar.backward()
        for n, p in model.named_parameters():
            if n in fisher and p.grad is not None:
                fisher[n].add_(p.grad.detach().float().cpu().pow(2))

        processed_batches += 1
        if processed_batches == 1 or processed_batches % 10 == 0:
            LOGGER.info(f"[EWC] fisher batch={processed_batches} loss={float(loss_scalar.detach()):.6f}")

    if processed_batches == 0:
        raise RuntimeError("No valid batch was processed in compute_fisher().")

    for n in fisher:
        fisher[n].div_(float(processed_batches))
    LOGGER.info(f"[EWC] fisher done: batches={processed_batches}, tensors={len(fisher)}")
    return fisher


def select_topk_fisher(fisher: dict[str, torch.Tensor], topk_ratio: float = 1.0) -> dict[str, torch.Tensor]:
    """Keep top-k Fisher values globally and return mask tensors."""
    if not 0 < topk_ratio <= 1.0:
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
    if topk_ratio == 1.0:
        return {n: torch.ones_like(v, dtype=torch.float32) for n, v in fisher.items()}

    all_fisher = torch.cat([v.reshape(-1) for v in fisher.values()])
    k = max(int(all_fisher.numel() * (1 - topk_ratio)), 1)
    threshold = torch.kthvalue(all_fisher, k).values
    LOGGER.info(f"[EWC] topk fisher threshold={float(threshold):.6f}")

    fisher_mask = {}
    total = 0
    kept = 0.0
    for n, v in fisher.items():
        mask = (v >= threshold).float()
        fisher_mask[n] = mask
        total += mask.numel()
        kept += float(mask.sum())
    LOGGER.info(f"[EWC] topk fisher kept ratio={kept / max(total, 1):.6f}")
    return fisher_mask


def save_ewc_artifacts(
    cache_dir: str | Path,
    old_params: dict[str, torch.Tensor],
    fisher: dict[str, torch.Tensor],
) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    old_params_path = cache_dir / "old_params.pth"
    fisher_path = cache_dir / "fisher.pth"
    torch.save(old_params, old_params_path)
    torch.save(fisher, fisher_path)
    LOGGER.info(f"[EWC] saved old params -> {old_params_path}")
    LOGGER.info(f"[EWC] saved fisher -> {fisher_path}")
    return old_params_path, fisher_path


def load_ewc_artifacts(cache_dir: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cache_dir = Path(cache_dir)
    old_params_path = cache_dir / "old_params.pth"
    fisher_path = cache_dir / "fisher.pth"
    if not old_params_path.exists() or not fisher_path.exists():
        raise FileNotFoundError(f"EWC cache files not found under {cache_dir}")
    old_params = torch.load(old_params_path, map_location="cpu")
    fisher = torch.load(fisher_path, map_location="cpu")
    LOGGER.info(f"[EWC] loaded old params <- {old_params_path}")
    LOGGER.info(f"[EWC] loaded fisher <- {fisher_path}")
    return old_params, fisher
