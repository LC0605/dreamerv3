#!/usr/bin/env python3
"""Render dependency-light trajectory/action diagnostics from eval CSV."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def scale(values, low, high, start, end):
    span = max(high - low, 1e-6)
    return [start + (float(value) - low) / span * (end - start) for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaldir", type=Path)
    args = parser.parse_args()
    with (args.evaldir / "trajectory_actions.csv").open() as stream:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    rows = [row for row in rows if not row["is_first"]]
    image = Image.new("RGB", (1500, 460), "white")
    draw = ImageDraw.Draw(image)
    panels = [(30, 45, 470, 425), (530, 45, 970, 425), (1030, 45, 1470, 425)]
    for box, title in zip(panels, ("Trajectories", "Actions", "Obstacle response")):
        draw.rectangle(box, outline=(30, 30, 30), width=2)
        draw.text((box[0], 15), title, fill=(0, 0, 0))

    # XY trajectories, with episode boundaries left disconnected.
    xs = [row["position_x"] for row in rows]
    ys = [row["position_y"] for row in rows]
    sx = scale(xs, min(xs), max(xs), panels[0][0] + 8, panels[0][2] - 8)
    sy = scale(ys, min(ys), max(ys), panels[0][3] - 8, panels[0][1] + 8)
    for index in range(1, len(rows)):
        if rows[index]["episode"] == rows[index - 1]["episode"]:
            draw.line((sx[index - 1], sy[index - 1], sx[index], sy[index]),
                      fill=(40, 100, 210), width=1)

    # World vx/vy and route-relative lateral action time series.
    limit = max(1.0, max(abs(row[key]) for row in rows for key in
                         ("action_forward", "action_lateral", "action_route_lateral")))
    ax = scale(range(len(rows)), 0, max(len(rows) - 1, 1),
               panels[1][0] + 8, panels[1][2] - 8)
    colors = {"action_forward": (30, 150, 60), "action_lateral": (220, 60, 40),
              "action_route_lateral": (70, 70, 220)}
    for key, color in colors.items():
        ay = scale([row[key] for row in rows], -limit, limit,
                   panels[1][3] - 8, panels[1][1] + 8)
        draw.line(list(zip(ax, ay)), fill=color, width=1)
    draw.text((540, 55), "green=vx  red=vy  blue=route lateral", fill=(0, 0, 0))

    # Bin obstacle-relative lateral location and show mean route lateral action.
    obs = [row["obstacle_lateral_relative"] for row in rows]
    acts = [row["action_route_lateral"] for row in rows]
    low, high = min(obs), max(obs)
    bins = [[] for _ in range(20)]
    for value, action in zip(obs, acts):
        index = min(19, int((value - low) / max(high - low, 1e-6) * 20))
        bins[index].append(action)
    bx = [low + (i + 0.5) / 20 * (high - low) for i, values in enumerate(bins) if values]
    by = [sum(values) / len(values) for values in bins if values]
    px = scale(bx, low, high, panels[2][0] + 8, panels[2][2] - 8)
    py = scale(by, -limit, limit, panels[2][3] - 8, panels[2][1] + 8)
    if len(px) > 1:
        draw.line(list(zip(px, py)), fill=(120, 30, 180), width=3)
    for x, y in zip(px, py):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(120, 30, 180))
    output = args.evaldir / "trajectory_action_diagnostics.png"
    image.save(output)
    print(output)


if __name__ == "__main__":
    main()
