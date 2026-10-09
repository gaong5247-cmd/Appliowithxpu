"""One command to diagnose the physical Intel Arc RVC end-to-end path.

Runs existing real GPU diagnostics in fresh subprocesses, writes one log file,
and stops at the FIRST error instead of misreporting overall success.
"""
import argparse
import datetime
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "xpu-diagnostics"


def run_step(label, args, report):
    headline = f"\n========== {label} ==========\n"
    print(headline, flush=True)
    report.write(headline)
    report.flush()
    start = time.monotonic()
    command = [sys.executable, *args]
    p = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    for line in p.stdout:
        print(line, end="", flush=True)
        report.write(line)
        report.flush()
    exit_code = p.wait()
    summary = f"\n[{label}] exit={exit_code}, elapsed_seconds={time.monotonic()-start:.1f}\n"
    print(summary, flush=True)
    report.write(summary)
    report.flush()
    if exit_code:
        raise RuntimeError(f"{label} failed: exit={exit_code}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, choices=(24000, 32000, 40000, 48000), default=40000)
    parser.add_argument("--download", action="store_true", help="Allow downloading upstream HuBERT and RMVPE weights")
    parser.add_argument("--keep", action="store_true", help="Keep generated test audio/checkpoints after a pass")
    parser.add_argument("--no-features", action="store_true", help="Skip HuBERT and RMVPE extraction stage")
    parser.add_argument("--no-epoch", action="store_true", help="Skip the real training-epoch and inference stage")
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = LOGS / f"xpu_full_{stamp}.log"
    try:
        with output.open("w", encoding="utf-8", buffering=1) as report:
            report.write(
                f"Applio XPU physical-device validation - {stamp}\n"
                f"sample_rate={args.sample_rate} download={args.download}\n"
            )
            run_step("GPU kernel and BF16 autodiff", ["scripts/xpu_smoke.py"], report)
            run_step("Real RVC Generator/Discriminator step", ["scripts/xpu_model_step.py"], report)
            if not args.no_features:
                extra = ["--sample-rate", str(args.sample_rate)]
                if args.download:
                    extra.append("--download")
                if args.keep:
                    extra.append("--keep")
                run_step("Real RMVPE + HuBERT extraction", ["scripts/xpu_feature_check.py", *extra], report)
            if not args.no_epoch:
                extra = ["--sample-rate", str(args.sample_rate)]
                if args.keep:
                    extra.append("--keep")
                run_step(
                    "Real RVC epoch + G/D checkpoint + exported .pth + XPU waveform",
                    ["scripts/xpu_one_epoch.py", *extra],
                    report,
                )
            report.write("\nPASS: all requested physical-device tests completed.\n")
    except Exception as error:
        print(f"\n[FAIL] {error}\nDiagnostic log retained: {output}", flush=True)
        raise SystemExit(1)

    print(f"\n[PASS] Intel GPU diagnostics completed. Full log: {output}", flush=True)


if __name__ == "__main__":
    main()
