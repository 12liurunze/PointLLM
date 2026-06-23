import os
import json
import argparse
from glob import glob
import subprocess
def get_video_duration(path):
    """
    使用 ffprobe 获取视频时长（秒）
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[WARN] Could not get duration for {path}: {e}")
        return 0.0
def build_metadata(video_dir, output_file):
    """
    扫描视频目录，生成 metadata.jsonl 文件
    每行一个样本，包含 video_id、path 和 duration（秒）
    """
    video_paths = glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
    video_paths.sort()  # 保证顺序一致

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, path in enumerate(video_paths):
            duration = get_video_duration(path)
            item = {
                "video_id": f"{idx:05d}",
                "path": os.path.abspath(path),
                "duration": duration
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved metadata with {len(video_paths)} entries to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir", type=str, required=True,
        help="LongVideoBench 视频目录"
    )
    parser.add_argument(
        "--output_file", type=str,
        default="/root/autodl-tmp/eagle-eye/EAGLE_EYE/eagle_eye/metadata.jsonl",
        help="输出 JSONL 文件路径"
    )
    args = parser.parse_args()

    build_metadata(args.video_dir, args.output_file)
