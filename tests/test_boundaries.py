import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "agent_elapsed"

FORBIDDEN = (
    "import scipy", "from scipy", "import requests", "import http",
    "import sqlite3", "from offline", "import offline", ".jsonl",
)


class RuntimeDependencyBoundaryTest(unittest.TestCase):
    def test_runtime_never_imports_offline_or_heavy_deps(self):
        for path in sorted(RUNTIME.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN:
                self.assertNotIn(
                    token, text, f"{path.name} 含被禁止的运行时依赖线索：{token}")

    def test_math_modules_do_no_file_io(self):
        for name in ("model.py", "estimate.py", "features.py"):
            path = RUNTIME / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("open(", text, f"{name} 不该做文件 I/O")
            self.assertNotIn("read_text", text, f"{name} 不该做文件 I/O")


if __name__ == "__main__":
    unittest.main()
