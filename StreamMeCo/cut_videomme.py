#!/usr/bin/env python3
import json
import math
import subprocess
from pathlib import Path

# Inputs
JSONL_PATH = Path("/data/annotations/videomme.json")
DEFAULT_VIDEO_ROOT = Path("/data/videos/videomme")
CLIPS_ROOT = Path("/data/videos/clips/videomme")
SEGMENT_SECONDS = 30


def _normalize_path(path_str: str) -> Path:
    # Accept both Windows and POSIX separators.
    cleaned = path_str.replace("\\", "/")
    return Path(cleaned)


def _resolve_video_path(record: dict) -> Path:
    if "video_path" in record and record["video_path"]:
        return _normalize_path(record["video_path"])
    vid_name = record.get("video_youtube_id") or record.get("video_id")
    if not vid_name:
        raise ValueError("Missing video identifier and video_path")
    return DEFAULT_VIDEO_ROOT / f"{vid_name}.mp4"


def _derive_clip_dir(video_path: Path) -> Path:
    # Preserve subdirectories if the path sits under the known video root.
    try:
        relative = video_path.relative_to(DEFAULT_VIDEO_ROOT)
        return (CLIPS_ROOT / relative).with_suffix("")
    except ValueError:
        return CLIPS_ROOT / video_path.stem


def _probe_duration_seconds(video_path: Path) -> int:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())
    return max(1, math.ceil(duration))


def _cut_video(video_path: Path, clip_dir: Path) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    duration_seconds = _probe_duration_seconds(video_path)
    segments = max(1, math.ceil(duration_seconds / SEGMENT_SECONDS))
    _print_progress(video_path.name, 0, segments)
    for idx in range(segments):
        start = idx * SEGMENT_SECONDS
        output = clip_dir / f"{idx}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(SEGMENT_SECONDS),
            "-c",
            "copy",
            str(output),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _print_progress(video_path.name, idx + 1, segments)
    print()


def _load_records():
    with JSONL_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _write_records(records):
    tmp_path = JSONL_PATH.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        for record in records:
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")
    tmp_path.replace(JSONL_PATH)


def _print_progress(label: str, current: int, total: int, width: int = 30) -> None:
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(f"{label[:24]:<24} [{bar}] {current}/{total}", end="\r", flush=True)


def main():
    records = list(_load_records())
    processed = {}
    for record in records:
        video_path = _resolve_video_path(record)
        clip_dir = processed.get(video_path)
        if clip_dir is None:
            clip_dir = _derive_clip_dir(video_path)
            _cut_video(video_path, clip_dir)
            processed[video_path] = clip_dir
        record["clip_path"] = clip_dir.as_posix()
    _write_records(records)


if __name__ == "__main__":
    main()
