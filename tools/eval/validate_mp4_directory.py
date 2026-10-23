#!/usr/bin/env python3
"""Validate generated benchmark MP4 files without decoding them on a GPU."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-frames", type=int, default=81)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--delete-invalid", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def probe(path, expected_frames):
    if path.stat().st_size < 100_000:
        return path, "file_too_small"
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        return path, result.stderr.strip() or "ffprobe_failed"
    try:
        stream = json.loads(result.stdout)["streams"][0]
        if not stream.get("codec_name"):
            return path, "missing_codec"
        if int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0:
            return path, "invalid_dimensions"
        if int(stream.get("nb_frames", 0)) != expected_frames:
            return path, f"frames={stream.get('nb_frames')}"
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return path, f"invalid_metadata:{exc}"
    return path, None


def main():
    args = parse_args()
    files = sorted(args.directory.glob("*.mp4"))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        checked = list(pool.map(lambda path: probe(path, args.expected_frames), files))
    invalid = [{"path": str(path), "reason": reason} for path, reason in checked if reason]
    if args.delete_invalid:
        for item in invalid:
            Path(item["path"]).unlink(missing_ok=True)
    report = {
        "directory": str(args.directory),
        "count_before": len(files),
        "expected_count": args.expected_count,
        "invalid_count": len(invalid),
        "invalid": invalid,
        "deleted_invalid": bool(args.delete_invalid),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    count_ok = args.expected_count is None or len(files) == args.expected_count
    raise SystemExit(0 if count_ok and not invalid else 1)


if __name__ == "__main__":
    main()
