from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lab_2_env_config_import_does_not_preload_pxr():
    code = """
import sys
import environments.tasks.lab_2_classical_control.classical_control_env_cfg
print(len([name for name in sys.modules if name.startswith("pxr")]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "0"
