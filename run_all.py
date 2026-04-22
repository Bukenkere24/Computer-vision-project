"""
Run the full project: experiment, all figures, enhancement demo, and quick tests.

Usage:
  py run_all.py
  py run_all.py --skip-install   (if dependencies are already installed)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    sys.stdout.flush()
    subprocess.run(cmd, cwd=str(_ROOT), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not run pip install -r requirements.txt",
    )
    args = p.parse_args()

    py = sys.executable

    if not args.skip_install:
        run([py, "-m", "pip", "install", "-r", "requirements.txt"], "1/5 Installing dependencies")

    run([py, "run_experiment.py", "--out", "output/summary.png"], "2/5 Main experiment + output/summary.png")
    run([py, "generate_figures.py"], "3/5 Figures 01-05 in figures/")
    run([py, "demo_enhancement.py"], "4/5 Enhancement figure (figures/06_...) + output/enhancement_demo.png")
    run([py, "tests/test_chain_code.py"], "5/5 Chain-code sanity tests")

    print()
    print("Done. Outputs:")
    print("  - Console: experiment metrics above")
    print("  - output/summary.png")
    print("  - output/enhancement_demo.png")
    print("  - figures/01_*.png through figures/06_*.png")
    print()


if __name__ == "__main__":
    main()
