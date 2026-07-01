"""
01_extract_transcripts.py
-------------------------
Segment raw .docx podcast transcripts into per-sentence CSV records.

Input : a directory of transcript .docx files (one per episode)
Output: a CSV with columns [episode_id, sentence_index, sentence_text]

Usage:
    python scripts/01_extract_transcripts.py \\
        --input transcripts/ \\
        --output outputs/sentences.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import docx  # from python-docx

from utils import sentence_segment


def read_docx_text(path: Path) -> str:
    """Return the concatenated visible text of a .docx file."""
    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())


def episode_id_from_path(path: Path) -> str:
    """Derive a stable episode id from the file stem. Users may adapt this."""
    stem = path.stem
    # Strip common numeric prefixes like "001 - " or "EP001_"
    stem = re.sub(r"^(EP)?0*(\d+)\s*[-_.]?\s*", r"EP\2 - ", stem)
    return stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Directory of .docx transcripts")
    ap.add_argument("--output", required=True, help="Output CSV path")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.docx"))
    if not files:
        print(f"[01_extract] No .docx files found in {in_dir}", file=sys.stderr)
        return 1

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "sentence_index", "sentence_text"])
        total = 0
        for path in files:
            text = read_docx_text(path)
            sents = sentence_segment(text)
            ep_id = episode_id_from_path(path)
            for i, s in enumerate(sents):
                w.writerow([ep_id, i, s])
                total += 1
            print(f"[01_extract] {path.name}: {len(sents)} sentences")

    print(f"[01_extract] wrote {total:,} sentences from {len(files)} episodes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
