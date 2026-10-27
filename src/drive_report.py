"""
Turn the frame_metrics.csv written by segment.py into a short drive report:
how often the path was clear, which obstacles appeared, lane-line visibility and
drivable area over time.

    python src/drive_report.py --csv results/segmented/frame_metrics.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STATUS_COLOURS = {"clear": "#2e9e44", "obstacle_ahead": "#f08c00", "obstacle_near": "#d6336c"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/segmented/frame_metrics.csv")
    args = p.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    n = len(df)
    shares = df["status"].value_counts(normalize=True).reindex(STATUS_COLOURS, fill_value=0) * 100
    print(f"Frames analysed: {n}")
    for status, pct in shares.items():
        print(f"  {status:15s} {pct:5.1f}%")
    obstacles = df.loc[df["obstacle"].notna() & (df["obstacle"] != ""), "obstacle"].value_counts()
    if len(obstacles):
        print("Nearest obstacle in path (frames):\n" + obstacles.to_string())
    print(f"Frames with at least one lane line: {100 * (df['lane_lines'] > 0).mean():.0f}%")
    print(f"Average drivable area in path: {df['drivable_pct'].mean():.0f}%")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    a.plot(df["frame"], df["drivable_pct"], color="#5b5b8f", lw=1.6, label="drivable area in path (%)")
    a.scatter(df["frame"], [103] * n, c=df["status"].map(STATUS_COLOURS), s=14, marker="s", label=None)
    a.set(ylim=(0, 108), xlabel="frame", ylabel="%", title="Drive timeline (top strip = path status)")
    a.legend(frameon=False, loc="lower left")
    b.barh([s.replace("_", " ") for s in shares.index], shares.values, color=list(STATUS_COLOURS.values()))
    b.invert_yaxis()
    b.set(xlabel="% of frames", title="Path status", xlim=(0, 100))
    for y, v in enumerate(shares.values):
        b.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=9)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = csv_path.parent / "drive_report.png"
    fig.savefig(out, dpi=150)
    print(f"\nChart saved to {out}")


if __name__ == "__main__":
    main()
