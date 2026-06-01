from __future__ import annotations

import csv
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GeoFilterParams:
    name: str
    max_axis_ratio: float
    min_area_ratio: float
    max_area_ratio: float
    center_margin_ratio: float
    bbox_margin_ratio: float


@dataclass
class Candidate:
    raw_line: str
    values: list[float]

    @property
    def row(self) -> float:
        return self.values[1]

    @property
    def col(self) -> float:
        return self.values[2]

    @property
    def cx(self) -> float:
        return self.col + 1.0

    @property
    def cy(self) -> float:
        return self.row + 1.0

    @property
    def axis_a(self) -> float:
        return self.values[3] / 2.0

    @property
    def axis_b(self) -> float:
        return self.values[4] / 2.0

    @property
    def angle_rad(self) -> float:
        return -self.values[5] / 180.0 * math.pi


@dataclass
class ImageStats:
    image_name: str
    width: int
    height: int
    before: int
    after: int
    removed_nonpositive_axis: int = 0
    removed_axis_ratio: int = 0
    removed_area_small: int = 0
    removed_area_large: int = 0
    removed_center: int = 0
    removed_bbox: int = 0
    removed_malformed: int = 0


def read_image_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(32)
        handle.seek(0)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            handle.seek(16)
            width, height = struct.unpack(">II", handle.read(8))
            return width, height

        if header.startswith(b"\xff\xd8"):
            handle.seek(2)
            while True:
                marker_start = handle.read(1)
                if not marker_start:
                    break
                if marker_start != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in {b"\xd8", b"\xd9"}:
                    continue
                length_bytes = handle.read(2)
                if len(length_bytes) != 2:
                    break
                segment_length = struct.unpack(">H", length_bytes)[0]
                if marker in {
                    b"\xc0",
                    b"\xc1",
                    b"\xc2",
                    b"\xc3",
                    b"\xc5",
                    b"\xc6",
                    b"\xc7",
                    b"\xc9",
                    b"\xca",
                    b"\xcb",
                    b"\xcd",
                    b"\xce",
                    b"\xcf",
                }:
                    data = handle.read(5)
                    if len(data) != 5:
                        break
                    height, width = struct.unpack(">HH", data[1:5])
                    return width, height
                handle.seek(segment_length - 2, 1)

    raise ValueError(f"Unsupported or unreadable image size: {path}")


def parse_fled(path: Path) -> tuple[str, list[Candidate], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return "0", [], []

    time_line = lines[0].strip() or "0"
    candidates: list[Candidate] = []
    malformed: list[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values = [float(value) for value in stripped.split()]
        except ValueError:
            malformed.append(stripped)
            continue
        if len(values) < 6:
            malformed.append(stripped)
            continue
        candidates.append(Candidate(raw_line=stripped, values=values[:6]))
    return time_line, candidates, malformed


def candidate_reject_reason(candidate: Candidate, width: int, height: int, params: GeoFilterParams) -> str | None:
    axis_a = candidate.axis_a
    axis_b = candidate.axis_b
    if axis_a <= 0.0 or axis_b <= 0.0:
        return "nonpositive_axis"

    long_axis = max(axis_a, axis_b)
    short_axis = min(axis_a, axis_b)
    axis_ratio = long_axis / short_axis
    if axis_ratio > params.max_axis_ratio:
        return "axis_ratio"

    image_area = float(width * height)
    ellipse_area_ratio = math.pi * axis_a * axis_b / image_area
    if ellipse_area_ratio < params.min_area_ratio:
        return "area_small"
    if ellipse_area_ratio > params.max_area_ratio:
        return "area_large"

    center_margin = params.center_margin_ratio * max(width, height)
    if (
        candidate.cx < -center_margin
        or candidate.cx > width + center_margin
        or candidate.cy < -center_margin
        or candidate.cy > height + center_margin
    ):
        return "center"

    cos_t = math.cos(candidate.angle_rad)
    sin_t = math.sin(candidate.angle_rad)
    extent_x = math.sqrt((axis_a * cos_t) ** 2 + (axis_b * sin_t) ** 2)
    extent_y = math.sqrt((axis_a * sin_t) ** 2 + (axis_b * cos_t) ** 2)
    bbox_margin = params.bbox_margin_ratio * max(width, height)
    if (
        candidate.cx - extent_x < -bbox_margin
        or candidate.cx + extent_x > width + bbox_margin
        or candidate.cy - extent_y < -bbox_margin
        or candidate.cy + extent_y > height + bbox_margin
    ):
        return "bbox"

    return None


def filter_one_file(
    image_name: str,
    input_dir: Path,
    image_dir: Path,
    output_dir: Path,
    params: GeoFilterParams,
) -> ImageStats:
    input_path = input_dir / f"{image_name}.fled.txt"
    output_path = output_dir / f"{image_name}.fled.txt"
    image_path = image_dir / image_name

    width, height = read_image_size(image_path)
    time_line, candidates, malformed = parse_fled(input_path)

    accepted: list[Candidate] = []
    stats = ImageStats(
        image_name=image_name,
        width=width,
        height=height,
        before=len(candidates),
        after=0,
        removed_malformed=len(malformed),
    )

    for candidate in candidates:
        reason = candidate_reject_reason(candidate, width, height, params)
        if reason is None:
            accepted.append(candidate)
            continue
        attr = f"removed_{reason}"
        setattr(stats, attr, getattr(stats, attr) + 1)

    stats.after = len(accepted)
    output_path.write_text(
        "\n".join([time_line, *[candidate.raw_line for candidate in accepted]]) + "\n",
        encoding="utf-8",
    )
    return stats


def summarize(stats: Iterable[ImageStats], params: GeoFilterParams) -> dict[str, object]:
    rows = list(stats)
    totals = {
        "before": sum(row.before for row in rows),
        "after": sum(row.after for row in rows),
        "removed": sum(row.before - row.after for row in rows),
        "removed_nonpositive_axis": sum(row.removed_nonpositive_axis for row in rows),
        "removed_axis_ratio": sum(row.removed_axis_ratio for row in rows),
        "removed_area_small": sum(row.removed_area_small for row in rows),
        "removed_area_large": sum(row.removed_area_large for row in rows),
        "removed_center": sum(row.removed_center for row in rows),
        "removed_bbox": sum(row.removed_bbox for row in rows),
        "removed_malformed": sum(row.removed_malformed for row in rows),
    }
    return {
        "params": asdict(params),
        "images": len(rows),
        "totals": totals,
        "retained_ratio": totals["after"] / totals["before"] if totals["before"] else 0.0,
    }


def write_stats(output_dir: Path, params: GeoFilterParams, stats: list[ImageStats]) -> None:
    csv_path = output_dir / "scc_geofilter_counts.csv"
    json_path = output_dir / "scc_geofilter_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(stats[0]).keys()) if stats else list(ImageStats.__annotations__))
        writer.writeheader()
        for row in stats:
            writer.writerow(asdict(row))

    json_path.write_text(json.dumps(summarize(stats, params), indent=2), encoding="utf-8")


def run_geofilter(
    image_names: list[str],
    input_dir: Path,
    image_dir: Path,
    output_dir: Path,
    params: GeoFilterParams,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = [
        filter_one_file(image_name, input_dir, image_dir, output_dir, params)
        for image_name in image_names
    ]
    write_stats(output_dir, params, stats)
    return summarize(stats, params)
