"""Crash-safe RVC checkpoint serialization.

Never replace a valid G/D or inference checkpoint until the replacement has
been completely serialized to a temporary file in the same directory.
"""
import os
import tempfile

import torch


def atomic_torch_save(payload, destination):
    destination = os.fspath(destination)
    folder = os.path.dirname(os.path.abspath(destination))
    os.makedirs(folder, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".rvc-writing-", suffix=".tmp", dir=folder)
    os.close(fd)
    try:
        torch.save(payload, temp)
        # This is atomic for files on the same filesystem, including Windows.
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
