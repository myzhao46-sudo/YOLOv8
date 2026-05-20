import torch
import os

path = r"C:\Users\DOCTOR\Desktop\yoloe-v8-s_distill_noreplay_300\exp_b_clean_freeze22_distill.pt"

print("file size:", os.path.getsize(path) / 1024 / 1024, "MB")

ckpt = torch.load(path, map_location="cpu", weights_only=False)

print("\n[1] checkpoint type:")
print(type(ckpt))

print("\n[2] checkpoint keys:")
if isinstance(ckpt, dict):
    for k in ckpt.keys():
        print(" -", k, type(ckpt[k]))
else:
    print("not a dict checkpoint")

print("\n[3] choose model:")
model = None
if isinstance(ckpt, dict):
    if "model" in ckpt and ckpt["model"] is not None:
        model = ckpt["model"]
        print("using ckpt['model']")
    elif "ema" in ckpt and ckpt["ema"] is not None:
        model = ckpt["ema"]
        print("using ckpt['ema']")
else:
    model = ckpt
    print("using ckpt itself")

if model is None:
    raise RuntimeError("No model found in checkpoint")

print("\n[4] model type:")
print(type(model))

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n[5] params:")
print(f"total params:     {total / 1e6:.3f} M")
print(f"trainable params: {trainable / 1e6:.3f} M")
print(f"frozen params:    {(total - trainable) / 1e6:.3f} M")

print("\n[6] dtype count:")
dtype_count = {}
for p in model.parameters():
    dtype_count[str(p.dtype)] = dtype_count.get(str(p.dtype), 0) + p.numel()
for dtype, n in dtype_count.items():
    print(dtype, n / 1e6, "M params")

print("\n[7] first 20 parameter requires_grad:")
for i, (name, p) in enumerate(model.named_parameters()):
    if i >= 20:
        break
    print(i, name, p.shape, p.dtype, "requires_grad=", p.requires_grad)