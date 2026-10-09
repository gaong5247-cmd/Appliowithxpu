"""Subprocess completion and metadata lifecycle for Intel XPU training.

No torch or GPU dependency: errors can be tested on ordinary GitHub runners.
"""
import json
import os


def write_process_ids(config_save_path, pids):
    """Atomically update process IDs without exposing a truncated JSON file.

    On Windows spawned RVC workers re-import train.py, and its module-level
    config loading must never race a parent's open(..., "w") truncation.
    """
    with open(config_save_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["process_pids"] = list(pids)
    temporary = config_save_path + ".parent-tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    os.replace(temporary, config_save_path)


def wait_for_training_workers(children, config_save_path):
    failures = []
    for child in children:
        child.join()
        if child.exitcode != 0:
            failures.append((child.pid, child.exitcode))

    # Parent is responsible for cleanup whether a GPU worker succeeds or fails.
    with open(config_save_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata.pop("process_pids", None)
    temporary = config_save_path + ".parent-tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    os.replace(temporary, config_save_path)

    if failures:
        raise RuntimeError(
            "Intel XPU RVC worker failed: "
            + ", ".join(f"pid={pid} exitcode={code}" for pid, code in failures)
            + "; see the original traceback above. The model is NOT trained."
        )
    return True
