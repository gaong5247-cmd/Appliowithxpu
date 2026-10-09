"""Intel Arc hardware diagnostic. This intentionally cannot fall back to CPU."""
import sys
import time
import traceback
import torch


def check(name, fn):
    t0 = time.perf_counter()
    try:
        value = fn()
        torch.xpu.synchronize()
        print(f"[PASS] {name}: {time.perf_counter() - t0:.3f}s {value or ''}")
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        raise


def main():
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), (
        "No Intel XPU found; this version has NO CUDA/CPU fallback"
    )
    dev = torch.device("xpu:0")
    info = torch.xpu.get_device_properties(0)
    print("PyTorch:", torch.__version__)
    print("Device:", torch.xpu.get_device_name(0))
    print("Memory reported:", round(info.total_memory / 1024**3, 2), "GiB (shared on iGPU)")
    print("Driver:", getattr(info, "driver_version", "unknown"))
    print("FP64:", getattr(info, "has_fp64", "unknown"))
    print("BF16 conversions:", getattr(info, "has_bfloat16_conversions", "unknown"))

    def matmul():
        a = torch.randn(512, 512, device=dev, requires_grad=True)
        with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
            loss = (a @ a).square().mean()
        loss.backward()
        return f"loss={loss.item():.2f}; grad={a.grad.dtype}"

    def conv():
        conv = torch.nn.Conv1d(64, 64, 3, padding=1).to(dev)
        x = torch.randn(2, 64, 1024, device=dev, requires_grad=True)
        loss = conv(x).square().mean()
        loss.backward()
        assert conv.weight.grad.device.type == "xpu"
        return "Conv1d + backward on XPU"

    def transposed():
        conv = torch.nn.ConvTranspose1d(64, 32, 4, stride=2).to(dev)
        x = torch.randn(2, 64, 512, device=dev, requires_grad=True)
        loss = conv(x).abs().mean()
        loss.backward()
        return "ConvTranspose1d + backward on XPU"

    def stft():
        wav = torch.randn(2, 16000, device=dev, requires_grad=True)
        window = torch.hann_window(1024, device=dev)
        spec = torch.stft(wav, n_fft=1024, hop_length=160,
                          window=window, return_complex=True)
        spec.abs().mean().backward()
        assert wav.grad.device.type == "xpu"
        return "STFT + complex magnitude + backward on XPU"

    check("BF16 AMP matmul backward", matmul)
    check("Generator/discriminator convolution", conv)
    check("Vocoder transposed convolution", transposed)
    check("Mel spectrogram FFT", stft)
    print("\nXPU COMPUTE SMOKE OK. End-to-end RVC training needs a real dataset/checkpoints.")


if __name__ == "__main__":
    main()
