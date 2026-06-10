from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
FINAL_DIR = PAPER_DIR / "final"
FINAL_PDF = FINAL_DIR / "iclr_submission.pdf"
DOWNLOAD_PDF = Path.home() / "Downloads" / "best-of-n-3d-scene-world-models.pdf"
AUDIT = ROOT / "docs" / "final_audit.md"


def _replace_audit_line(prefix: str, value: str) -> None:
    if not AUDIT.exists():
        return
    text = AUDIT.read_text(encoding="utf-8")
    if prefix in text:
        text = text.replace(prefix, value)
    elif value not in text:
        text += "\n" + value + "\n"
    AUDIT.write_text(text, encoding="utf-8")


def main() -> int:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    try:
        for _ in range(2):
            completed = subprocess.run(
                command,
                cwd=PAPER_DIR,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            if completed.returncode != 0:
                failure = (
                    "PDF build failed.\n\n"
                    f"Command: {' '.join(command)}\n\n"
                    f"Error output:\n{completed.stdout[-4000:]}\n\n"
                    "Next fix: inspect paper/main.tex and the LaTeX log, then rerun scripts/build_paper.py."
                )
                (FINAL_DIR / "build_failure.txt").write_text(failure, encoding="utf-8")
                _replace_audit_line(
                    "Pending build: `C:\\Users\\wangz\\Downloads\\best-of-n-3d-scene-world-models.pdf`",
                    "Build failed; see `paper/final/build_failure.txt`.",
                )
                print(failure)
                return completed.returncode
    except FileNotFoundError:
        failure = (
            "PDF build failed.\n\n"
            "Command: pdflatex -interaction=nonstopmode -halt-on-error main.tex\n\n"
            "Error output:\npdflatex was not found on PATH.\n\n"
            "Next fix: install a LaTeX distribution or place pdflatex on PATH."
        )
        (FINAL_DIR / "build_failure.txt").write_text(failure, encoding="utf-8")
        _replace_audit_line(
            "Pending build: `C:\\Users\\wangz\\Downloads\\best-of-n-3d-scene-world-models.pdf`",
            "Build failed; see `paper/final/build_failure.txt`.",
        )
        print(failure)
        return 1

    built = PAPER_DIR / "main.pdf"
    if not built.exists():
        failure = "PDF build failed: pdflatex exited successfully but paper/main.pdf was not produced."
        (FINAL_DIR / "build_failure.txt").write_text(failure, encoding="utf-8")
        print(failure)
        return 1

    shutil.copyfile(built, FINAL_PDF)
    shutil.copyfile(built, DOWNLOAD_PDF)
    _replace_audit_line(
        "Pending build: `C:\\Users\\wangz\\Downloads\\best-of-n-3d-scene-world-models.pdf`",
        "`C:\\Users\\wangz\\Downloads\\best-of-n-3d-scene-world-models.pdf`",
    )
    print(f"wrote {FINAL_PDF}")
    print(f"wrote {DOWNLOAD_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
