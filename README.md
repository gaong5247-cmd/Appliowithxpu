# Applio with Intel XPU

**Standalone source fork of [IAHispano/Applio](https://github.com/IAHispano/Applio)**, with an experimental PyTorch Intel XPU port for Intel Arc Graphics / Core Ultra on Windows 11. Upstream source is copied into THIS repository, not downloaded as an overlay when you launch it. Based on upstream commit `324f4d89c3e0e8dc0e555a3d53784e3282c16e09`; source license is preserved in `LICENSE`.

**IMPORTANT STATUS:** Windows GitHub Actions checks **Python syntax and static XPU guards only**. Real Intel Arc RVC 200-epoch training, HuBERT, RMVPE and realtime conversion have **not been demonstrated on an Intel GPU by CI**. This is a port under validation, not a certified fast/stable release. When hardware is unavailable, the application refuses to run neural compute on CUDA or CPU.

## Windows installation

Requirements: Windows 11, Intel Arc graphics driver **32.0.101.8801 or newer** per [Intel's PyTorch 2.14 requirements](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-14.html), Python **3.12** (or **3.11**) with the `py` launcher, Git, internet for PyTorch and model downloads.

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
- `APPLIO_XPU_GRAD_LOG_INTERVAL=10` (default): only collect expensive gradient-norm diagnostics every 10 steps, plus each TensorBoard 50-step boundary and epoch end. Use 1 to restore logging every step.
- `APPLIO_XPU_ADAMW_FOREACH=0` (default): keep memory-conservative AdamW. On an Intel GPU, `.\.venv-xpu\Scripts\python.exe scripts\benchmark_xpu_adamw.py` compares the regular and foreach optimizer implementations. Set 1 only after verifying speed **and** memory headroom using a real RVC epoch.
- `torch.xpu.empty_cache()` is not called per batch or epoch; avoiding allocator churn generally helps.

## Legacy RVC .pth compatibility and safe precision

- Public RVC `.pth` exports often use legacy `weight_g/weight_v` tensors, while recent PyTorch parametrized weight normalization uses `parametrizations.weight.original0/original1`. Both offline and realtime loaders now normalize these keys and **reject missing/unexpected neural weights** instead of silently loading only a subset (`rvc/lib/weights.py`).
- Fast Windows CI verifies both root-level and nested conversion. The full GPU-independent integration workflow also checks actual generator export/reload tensor parity with `scripts/test_rvc_checkpoint_roundtrip.py`.
- Headless `rvc/train/train.py` reads the same BF16 default from `assets/config_template.json` when the GUI has not created `assets/config.json`. Override per run using `APPLIO_XPU_PRECISION=fp32|bf16|fp16`.
- PyTorch's Intel XPU documentation notes that FP16 GradScaler needs FP64 hardware support. The trainer refuses unsupported FP16 setups and directs the user toward BF16/FP32, rather than quietly continuing with unsafe optimizer behavior.

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
# Optional: test 40 kHz RVC and retain synthetic wave/checkpoints
.\run-xpu-one-epoch.bat --sample-rate 40000 --keep
```

This does **not** need to download or reuse anybody's voice model. The generated experiment has a unique name under `logs/__xpu_one_epoch_*`. On success it removes only its own test directory; on failure it **preserves it** for diagnosis. Pass `--keep` to retain it on success. Never remove an unrelated experiment folder.

If successful, you have validated a real G/D RVC training epoch, checkpoint export, and generated-waveform inference on Intel XPU **for the sample rate you tested**. The launcher supports `--sample-rate 24000|32000|40000|48000`. It still does **not** certify 200 epochs, real feature extraction (HuBERT/RMVPE), live microphone conversion, or final audio quality. Those are separate tests on the physical Arc device.

GPU training prints `[XPU BENCH] epoch=... seconds=... steps_per_second=...` for measurements; compare per-epoch numbers with your baseline only using the **same dataset, batch size and settings**.

## One-click Intel Arc end-to-end diagnostics

Instead of running each hardware test manually, after `run-install.bat` you can invoke:

```powershell
.\run-xpu-validate.bat --sample-rate 40000 --download --keep
```

This sequentially checks **BF16/Conv/STFT GPU kernels**, real RVC generator and discriminator update, RMVPE+HuBERT feature extraction, **40 kHz full one-epoch training**, G/D checkpoints, public `.pth` export and **XPU audio waveform generation**. `--download` fetches missing upstream predictor/ContentVec weights; omit it to disallow downloads. Every stage runs in a clean subprocess so an earlier model cannot contaminate the result. The script stops at the first failure and writes the exact traceback and timing to `logs/xpu-diagnostics/xpu_full_*.log`. Those logs can be used to identify precisely which Intel Arc operation failed.

For a short smoke test without network or model weights: `.\run-xpu-validate.bat --no-features --no-epoch`.

**Important:** These are real Intel GPU tests **only when run on a physical Arc GPU**. Passing a Windows GitHub Actions source-check does not establish GPU compatibility or a speedup. Synthetic sample durations differ from your actual voice dataset, so compare epoch times only using identical real datasets and settings.

## Verify real RMVPE and HuBERT on Intel Arc

The one-epoch test above uses generated F0/HuBERT arrays and therefore **does not validate feature extraction**. This separate diagnostic erases the generated labels from its own temporary dataset, then runs the **actual Applio RMVPE and ContentVec/HuBERT extraction scripts** on eight synthetic clips using the Intel XPU:

```powershell
.\run-xpu-feature-check.bat --sample-rate 40000
# If RMVPE or ContentVec weights are not yet downloaded:
.\run-xpu-feature-check.bat --sample-rate 40000 --download --keep
```

It fails if any real F0, voiced-F0 or HuBERT output is missing/corrupt. On success it cleans up its unique temporary directory unless `--keep` is given. On failure the diagnostic directory and original Python traceback are retained. `--download` fetches upstream weights from IAHispano/Applio and may consume significant internet bandwidth; omit it when you already have the models.

**Recommended physical-GPU validation sequence:** `run-xpu-check.bat` → `run-xpu-feature-check.bat` → `run-xpu-one-epoch.bat --sample-rate 40000` → a real one-epoch voice dataset trial → the intended 200 epochs. Neither these scripts nor hosted GitHub CI claim the Arc device has passed until you actually run them.

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
