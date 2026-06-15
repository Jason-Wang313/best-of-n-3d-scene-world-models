from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "results" / "expansion" / "claims.json"
MODELNET_CLAIMS = ROOT / "results" / "modelnet10_benchmark" / "claims.json"
PAPER = ROOT / "paper" / "main.tex"
FINAL_PDF = ROOT / "paper" / "final" / "best-of-n-3d-scene-world-models-v4.pdf"
DESKTOP_PDF = Path.home() / "OneDrive" / "Desktop" / "best-of-n-3d-scene-world-models-v4.pdf"

STALE_PATTERNS = [
    "best-of-n-3d-scene-world-models-" + "v" + "2",
    "best-of-n-3d-scene-world-models-" + "v" + "3",
    "submission-ready " + "v" + "3",
    "desktop artifact named " + "v" + "3",
    "iclr" + "_submission.pdf",
    "best of n " + "llm",
    "inference " + "value " + "theorem",
]


def _pdf_page_count(path: Path) -> int:
    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page\b", data))


def main() -> int:
    failures: list[str] = []
    if not CLAIMS.exists():
        failures.append(f"missing claims file: {CLAIMS}")
    else:
        claims = json.loads(CLAIMS.read_text(encoding="utf-8"))
        if not claims.get("claim_pass"):
            failures.append("claim_pass is false")
        for name, value in claims.get("checks", {}).items():
            if value is not True:
                failures.append(f"claim check failed: {name}={value}")

    if not MODELNET_CLAIMS.exists():
        failures.append(f"missing ModelNet10 claims file: {MODELNET_CLAIMS}")
    else:
        modelnet = json.loads(MODELNET_CLAIMS.read_text(encoding="utf-8"))
        if not modelnet.get("all_passed"):
            failures.append("ModelNet10 benchmark all_passed is false")
        for name, payload in modelnet.get("checks", {}).items():
            if payload.get("passed") is not True:
                failures.append(f"ModelNet10 claim check failed: {name}")

    paper_text = PAPER.read_text(encoding="utf-8")
    lower_text = paper_text.lower()
    for pattern in STALE_PATTERNS:
        if pattern.lower() in lower_text:
            failures.append(f"stale text found in paper/main.tex: {pattern}")

    for pdf in [FINAL_PDF, DESKTOP_PDF]:
        if not pdf.exists():
            failures.append(f"missing PDF: {pdf}")
            continue
        pages = _pdf_page_count(pdf)
        if pages < 25:
            failures.append(f"PDF has only {pages} pages: {pdf}")

    if failures:
        print("claim audit failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("claim audit complete: submission-ready v4")
    print(f"claims: {CLAIMS}")
    print(f"modelnet: {MODELNET_CLAIMS}")
    print(f"final: {FINAL_PDF}")
    print(f"desktop: {DESKTOP_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
