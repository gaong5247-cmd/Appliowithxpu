# Applio with Intel XPU

**Standalone source fork of [IAHispano/Applio](https://github.com/IAHispano/Applio)**, with an experimental PyTorch Intel XPU port for Intel Arc Graphics / Core Ultra on Windows 11. Upstream source is copied into THIS repository, not downloaded as an overlay when you launch it. Based on upstream commit `324f4d89c3e0e8dc0e555a3d53784e3282c16e09`; source license is preserved in `LICENSE`.

**IMPORTANT STATUS:** Windows GitHub Actions checks **Python syntax and static XPU guards only**. Real Intel Arc RVC 200-epoch training, HuBERT, RMVPE and realtime conversion have **not been demonstrated on an Intel GPU by CI**. This is a port under validation, not a certified fast/stable release. When hardware is unavailable, the application refuses to run neural compute on CUDA or CPU.

## Windows installation

Requirements: Windows 11, updated Intel Arc graphics driver, Python **3.12** (or **3.11**) with the `py` launcher, Git, internet for PyTorch and model downloads.

```powershell
git clone https://github.com/gaong5247-cmd/Appliowithxpu.git
cd Appliowithxpu
.\run-install.bat
.\run-applio.bat
```

Run without administrator rights. Installer creates **`.venv-xpu`**, installs PyTorch + torchaudio from the official Intel XPU wheel index, installs the other Applio dependencies, checks Intel GPU availability, and runs `scripts/xpu_smoke.py`. It will **fail** rather than silently installing CPU/CUDA PyTorch. Do not use the original CUDA/Miniconda installer. You may install Python from https://www.python.org/downloads/.

Manual driver/framework check:
```powershell
.\.venv-xpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0))"
.\.venv-xpu\Scripts\python.exe scripts\xpu_smoke.py
```

## Training setup (start here)

1. Back up any existing `logs/` (checkpoints), `rvc/models/`, dataset audio and indexes. Do **not** delete them.
2. Run preprocessing and feature extraction in the original **Train** tab; pitch extractor `rmvpe` is the initial path to test.
3. For an Intel integrated Arc GPU, start at **batch size 2** (4 if memory permits), use the `bf16` precision set in `assets/config_template.json` and use a small dataset for an initial 1-epoch trial.
4. Confirm GPU utilization and actual epoch time. Try `fp32` if BF16 triggers unsupported kernels or NaNs.
5. Then run your target 200 epochs. Compare against the CPU baseline (**4 minutes/epoch** for your setup); speedup is not guaranteed.

Performance controls:

- `APPLIO_XPU_WORKERS=2` (default): PyTorch DataLoader worker processes; tune 0–4 on Windows.
- `APPLIO_XPU_CACHE=0` (default): no full-dataset GPU caching on integrated graphics. Set 1 only if the whole dataset fits and the GUI caching checkbox is also enabled.
- `torch.xpu.empty_cache()` is not called per batch or epoch; avoiding allocator churn generally helps.

## Ported code areas

- `rvc/configs/config.py`: strict XPU device, GPU names and iGPU-specific inference buffer sizes.
- `rvc/train/train.py`: single Intel GPU training, optimizer and XPU BF16/FP16 AMP, XPU tensors, no NCCL or CUDA calls, conservative data loading.
- `rvc/train/extract/extract.py`: XPU F0 / HuBERT feature extraction worker and exception reporting.
- `rvc/infer/infer.py`, `rvc/infer/pipeline.py`: torch XPU RVC `.pth` model inference, CUDA cleanup removed.
- `rvc/lib/predictors/RMVPE.py`, `FCPE.py`: XPU device handling and memory management; `f0.py` uses DirectML (not XPU) for optional Swift ONNX inference.
- `rvc/realtime/`: reuses the central XPU Config and removes CUDA cleanup calls.
- Windows/Linux installer + `requirements.txt`: remove CUDA wheel index and CUDA ONNX runtime.
- `scripts/verify_xpu_port.py`: source/guard tests in Windows CI.
- `scripts/xpu_smoke.py`: real GPU AMP, Conv1d, transposed conv and STFT backward diagnostics.

**Important limitation:** Sound file loading, resampling, FAISS index search and checkpoint reading legitimately use CPU-host work; "XPU-only" refers to **neural GPU compute** and no CPU/CUDA compute fallback, not an impossible complete ban on CPU tasks. ONNX DirectML is a separate optional backend. Some operators or dependencies may still be incompatible with XPU on particular Intel drivers.

## Test a real Intel Arc training epoch (no personal dataset)

`run-xpu-check.bat` first tests Intel XPU primitives plus the real generator/discriminator synthetic forward/backward. After that, `run-xpu-one-epoch.bat` runs **the unchanged real rvc/train/train.py command-line entrypoint** for one epoch on eight generated 24 kHz synthetic clips and verifies that both G/D checkpoints were written. It takes nontrivial GPU time and memory, so run it separately:

```powershell
.\run-xpu-check.bat
.\run-xpu-one-epoch.bat
```

This does **not** need to download or reuse anybody's voice model. The generated experiment has a unique name under `logs/__xpu_one_epoch_*`. On success it removes only its own test directory; on failure it **preserves it** for diagnosis. Pass `--keep` to retain it on success. Never remove an unrelated experiment folder.

If successful, you have validated actual G/D RVC training and checkpoint writing for the 24 kHz synthetic case. It still does **not** certify 40/48 kHz, feature extraction (HuBERT/RMVPE), 200 epochs, live microphone conversion, or final audio quality. Those are separate tests on the physical Arc device.

GPU training prints `[XPU BENCH] epoch=... seconds=... steps_per_second=...` for measurements; compare per-epoch numbers with your baseline only using the **same dataset, batch size and settings**.

## CI artifacts

[Actions: Validate full Applio XPU source](https://github.com/gaong5247-cmd/Appliowithxpu/actions/workflows/package-xpu.yml) checks the full source on Windows and creates `Applio-XPU-Full-Source-Windows`. This is **source**, not an installer or proven GPU-trained model. The runner has no Intel Arc GPU and does not perform an RVC epoch.

## Diagnostics and reporting

If installation or model training fails, collect:

```powershell
.\.venv-xpu\Scripts\python.exe scripts\xpu_smoke.py
.\.venv-xpu\Scripts\python.exe -m pip check
```

Include the first traceback, Python version, torch version, Intel driver version, dataset length, batch size and precision. This helps distinguish unsupported XPU operations from CUDA-specific dependencies.

Upstream README: [UPSTREAM_README.md](UPSTREAM_README.md). Licensing: [LICENSE](LICENSE).
