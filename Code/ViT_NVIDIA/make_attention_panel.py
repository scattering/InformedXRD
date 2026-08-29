#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pattern", default="attention_*.png")
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--summary-json")
    p.add_argument("--crop-top-frac", type=float, default=0.10)
    p.add_argument("--cell-width", type=int, default=1000)
    p.add_argument("--cell-height", type=int, default=360)
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit("no input PNGs found")
    summary_by_name = {}
    if args.summary_json:
        with open(args.summary_json) as f:
            summary = json.load(f)
        for row in summary:
            png_name = Path(row["png"]).name
            mineral = row["case_id"].split("__")[0]
            pred = row["topk_pred_egs"][0]
            status = "correct" if pred == row["true_eg"] else "descendant error"
            summary_by_name[png_name] = (
                f"{mineral} ({status})\n"
                f"true EG {row['true_eg']} | top-1 EG {pred}"
            )
    cols = max(1, args.cols)
    rows = (len(files) + cols - 1) // cols
    panel = Image.new("RGB", (cols * args.cell_width, rows * args.cell_height), "white")
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
        small_font = ImageFont.truetype("Arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for i, path in enumerate(files):
        row, col = divmod(i, cols)
        x0 = col * args.cell_width
        y0 = row * args.cell_height
        img = Image.open(path).convert("RGB")
        crop_px = int(img.height * max(0.0, min(args.crop_top_frac, 0.4)))
        if crop_px > 0:
            img = img.crop((0, crop_px, img.width, img.height))
        title = summary_by_name.get(path.name, "")
        title_lines = title.splitlines() if title else []
        title_h = 52 if title_lines else 8
        avail_w = args.cell_width - 20
        avail_h = args.cell_height - title_h - 12
        img.thumbnail((avail_w, avail_h))
        paste_x = x0 + (args.cell_width - img.width) // 2
        paste_y = y0 + title_h + (avail_h - img.height) // 2
        panel.paste(img, (paste_x, paste_y))
        if title_lines:
            draw.text((x0 + 10, y0 + 6), title_lines[0], fill="black", font=font)
            if len(title_lines) > 1:
                draw.text((x0 + 10, y0 + 30), title_lines[1], fill="black", font=small_font)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    panel.save(args.output)


if __name__ == "__main__":
    main()
