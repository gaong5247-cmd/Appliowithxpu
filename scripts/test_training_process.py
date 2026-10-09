"""Regression tests for failed/successful training-process results."""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rvc.train.process.worker_status import wait_for_training_workers, write_process_ids


class Child:
    def __init__(self, pid, exitcode):
        self.pid, self.exitcode = pid, exitcode
        self.joined = False

    def join(self):
        self.joined = True


def case(exitcodes):
    with tempfile.TemporaryDirectory() as temp:
        path = pathlib.Path(temp) / "config.json"
        path.write_text(json.dumps({"name": "sample", "process_pids": [10, 20]}), encoding="utf-8")
        children = [Child(n + 100, exitcode) for n, exitcode in enumerate(exitcodes)]
        write_process_ids(str(path), [child.pid for child in children])
        assert json.loads(path.read_text(encoding="utf-8"))["process_pids"] == [child.pid for child in children]
        assert not (pathlib.Path(temp) / "config.json.parent-tmp").exists()
        try:
            wait_for_training_workers(children, str(path))
            failed = False
        except RuntimeError as exc:
            failed = True
            assert "NOT trained" in str(exc)
        assert failed == any(exitcode != 0 for exitcode in exitcodes)
        assert all(child.joined for child in children)
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved == {"name": "sample"}, saved


if __name__ == "__main__":
    case([0])
    case([0, 0])
    case([1])
    case([0, -1073741819])
    print("PASS: training worker success, failure propagation, cleanup, and joining.")
