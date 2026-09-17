import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_ROOT = ROOT / "datasets/MELD"
OUTPUT = ROOT / "datasets/MELD/MELD.Raw/audio"

SPLITS = {
    "train": "train",
    "dev": "dev",
    "test": "test",
}

for tool in ("ffmpeg", "ffprobe"):
    if shutil.which(tool) is None:
        raise SystemExit(f"{tool} is missing. Install FFmpeg first.")

for folder in SPLITS.values():
    if not (VIDEO_ROOT / folder).is_dir():
        raise SystemExit(f"Missing video folder: {VIDEO_ROOT / folder}")

OUTPUT.mkdir(parents=True, exist_ok=True)
counts = Counter()

with (OUTPUT / "extraction_report.csv").open(
    "w", newline="", encoding="utf-8"
) as report:
    writer = csv.writer(report)
    writer.writerow(["video", "audio", "status", "details"])

    for split, folder in SPLITS.items():
        (OUTPUT / split).mkdir(parents=True, exist_ok=True)
        for video in sorted((VIDEO_ROOT / folder).glob("*.mp4")):
            # Ignore macOS metadata files.
            if video.name.startswith("._"):
                continue

            audio = OUTPUT / split / f"{video.stem}.wav"
            temporary = audio.with_suffix(".partial.wav")

            try:
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-select_streams", "a:0",
                        "-show_entries", "stream=index",
                        "-of", "json", str(video),
                    ],
                    capture_output=True, text=True, check=True,
                    timeout=60,
                )

                if not json.loads(probe.stdout).get("streams"):
                    status, details = "no_audio", "No audio track found"
                else:
                    subprocess.run(
                        [
                            "ffmpeg", "-nostdin", "-v", "error",
                            "-xerror", "-y", "-i", str(video),
                            "-map", "0:a:0", "-vn",
                            "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", str(temporary),
                        ],
                        capture_output=True, text=True, check=True,
                        timeout=120,
                    )

                    # Confirm that extraction produced audio samples.
                    import wave

                    with wave.open(str(temporary), "rb") as wav:
                        if wav.getnframes() == 0:
                            raise ValueError("Decoded audio contains no samples")

                    temporary.replace(audio)
                    status, details = "converted", ""

            except subprocess.CalledProcessError as error:
                status = "failed"
                details = (error.stderr or str(error)).strip()[-1000:]
            except Exception as error:
                status, details = "failed", str(error)
            finally:
                temporary.unlink(missing_ok=True)

            counts[status] += 1
            writer.writerow([
                str(video),
                str(audio) if status == "converted" else "",
                status,
                details,
            ])
            print(f"{status}: {split}/{video.name}")

print("\nResults:", dict(counts))
print("Audio folder:", OUTPUT)
print("Report:", OUTPUT / "extraction_report.csv")