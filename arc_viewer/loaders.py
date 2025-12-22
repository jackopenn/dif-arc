from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class LoadStats:
    path: str
    lines: int
    seconds: float


def _iter_records(path: str | os.PathLike[str]):
    """
    Supports:
      - JSONL: one JSON object per line
      - JSON: either a single object or a list of objects
    """
    path = str(path)
    p = Path(path)
    if p.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for item in obj:
                if item is not None:
                    yield item
        else:
            yield obj
        return

    # default: JSONL
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_test_jsonl_by_aug_id(path: str | os.PathLike[str], *, verbose: bool = True) -> tuple[dict[int, dict[str, Any]], LoadStats]:
    """
    Loads a JSONL of records that include at least:
      - aug_puzzle_idx (int)
      - x (2d list)
      - y (2d list)

    Returns dict: aug_puzzle_idx -> record (full metadata preserved).
    """
    t0 = time.time()
    path = str(path)
    by_id: dict[int, dict[str, Any]] = {}
    n = 0
    if verbose:
        print(f"[arc_viewer] loading test jsonl: {path}")
    for obj in _iter_records(path):
        n += 1
        aug_id = int(obj["aug_puzzle_idx"])
        by_id[aug_id] = obj
    dt = time.time() - t0
    if verbose:
        print(f"[arc_viewer] loaded {len(by_id):,} test records ({n:,} lines) in {dt:.1f}s")
    return by_id, LoadStats(path=path, lines=n, seconds=dt)


def load_test_records(path: str | os.PathLike[str], *, verbose: bool = True) -> tuple[list[dict[str, Any]], LoadStats]:
    """
    Loads all records from the test file (JSONL or JSON).
    Unlike `load_test_jsonl_by_aug_id`, this preserves duplicates (multiple example_idx per aug_puzzle_idx).
    """
    t0 = time.time()
    path = str(path)
    records: list[dict[str, Any]] = []
    n = 0
    if verbose:
        print(f"[arc_viewer] loading test records: {path}")
    for obj in _iter_records(path):
        n += 1
        records.append(obj)
    dt = time.time() - t0
    if verbose:
        print(f"[arc_viewer] loaded {len(records):,} records ({n:,} items) in {dt:.1f}s")
    return records, LoadStats(path=path, lines=n, seconds=dt)


def _extract_pred_grid(obj: dict[str, Any]) -> list[list[int]]:
    if "preds" in obj:
        return obj["preds"]
    if "y_pred" in obj:
        return obj["y_pred"]
    if "pred" in obj:
        return obj["pred"]
    raise KeyError("Prediction grid not found; expected one of: preds, y_pred, pred")


def load_preds_jsonl(
    path: str | os.PathLike[str], *, verbose: bool = True
) -> tuple[dict[int, list[list[int]]], dict[tuple[int, int], list[list[int]]], LoadStats]:
    """
    Loads a preds JSONL of objects that include:
      - aug_puzzle_idx (int)
      - optional example_idx (int)
      - preds (2d list) or y_pred (2d list)

    Returns:
      - by_aug: aug_puzzle_idx -> pred_grid (used as fallback)
      - by_aug_ex: (aug_puzzle_idx, example_idx) -> pred_grid (preferred when present)
    """
    t0 = time.time()
    path = str(path)
    by_aug: dict[int, list[list[int]]] = {}
    by_aug_ex: dict[tuple[int, int], list[list[int]]] = {}
    n = 0
    if verbose:
        size = None
        try:
            size = Path(path).stat().st_size
        except Exception:
            pass
        extra = f" ({size/1e9:.2f}GB)" if size is not None else ""
        print(f"[arc_viewer] loading preds jsonl: {path}{extra}")
    for obj in _iter_records(path):
        n += 1
        aug_id = int(obj["aug_puzzle_idx"])
        grid = _extract_pred_grid(obj)
        if "example_idx" in obj and obj["example_idx"] is not None:
            ex_id = int(obj["example_idx"])
            by_aug_ex[(aug_id, ex_id)] = grid
        else:
            by_aug[aug_id] = grid
    dt = time.time() - t0
    if verbose:
        extra2 = f", {len(by_aug_ex):,} with example_idx" if by_aug_ex else ""
        print(f"[arc_viewer] loaded {len(by_aug):,} aug-level preds ({n:,} lines){extra2} in {dt:.1f}s")
    return by_aug, by_aug_ex, LoadStats(path=path, lines=n, seconds=dt)


def grid_shape(grid: Optional[list[list[int]]]) -> tuple[int, int]:
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]) if grid and grid[0] is not None else 0)


def dataloader_translate_and_pad(
    x: Optional[list[list[int]]],
    y: Optional[list[list[int]]],
    *,
    max_grid_size: int = 30,
    translate: bool | str = False,
    pad_value: int = 11,
    border_value: int = 10,
    aug_puzzle_idx: Optional[int] = None,
) -> tuple[Optional[list[list[int]]], Optional[list[list[int]]]]:
    """
    Reimplements `dataset.TranslateAndPad` for the viewer.

    - Pads x/y into max_grid_size x max_grid_size with `pad_value` (default 11)
    - Adds right/bottom border with `border_value` (default 10)
    - Uses shared (pad_r, pad_c) offsets for translation
    - Computes border coordinates using each grid's own shape (matches updated dataset.py)
    """
    if x is None or y is None:
        return None, None

    xh, xw = grid_shape(x)
    yh, yw = grid_shape(y)
    if xh == 0 or xw == 0 or yh == 0 or yw == 0:
        return None, None
    if xh > max_grid_size or xw > max_grid_size or yh > max_grid_size or yw > max_grid_size:
        raise ValueError(f"grid larger than max_grid_size={max_grid_size}: x={xh}x{xw}, y={yh}x{yw}")

    max_rows = max(xh, yh)
    max_cols = max(xw, yw)

    # Determine translation offsets (pad_r, pad_c)
    if translate is False or translate == "none":
        pad_r = 0
        pad_c = 0
    elif translate == "fixed":
        if aug_puzzle_idx is None:
            raise ValueError("translate='fixed' requires aug_puzzle_idx")
        # Mirror dataset.py: np.random.default_rng(seed=aug_puzzle_idx)
        import numpy as _np

        local_rng = _np.random.default_rng(seed=int(aug_puzzle_idx))
        pad_r = int(local_rng.integers(0, max_grid_size - max_rows + 1))
        pad_c = int(local_rng.integers(0, max_grid_size - max_cols + 1))
    elif translate is True or translate == "random":
        # Viewer doesn't have a stable RNG stream; fall back to deterministic zero offset.
        # If you want true random translation in the viewer, we can add a seed parameter.
        pad_r = 0
        pad_c = 0
    else:
        raise ValueError(f"unknown translate={translate!r}")

    def blank():
        return [[pad_value for _ in range(max_grid_size)] for _ in range(max_grid_size)]

    def translate_pad_and_add_border(grid: list[list[int]]) -> list[list[int]]:
        gh, gw = grid_shape(grid)
        padded = blank()
        for r in range(gh):
            row = grid[r]
            for c in range(gw):
                padded[pad_r + r][pad_c + c] = int(row[c])

        eos_row, eos_col = pad_r + gh, pad_c + gw
        if eos_row < max_grid_size:
            for c in range(pad_c, eos_col):
                if 0 <= c < max_grid_size:
                    padded[eos_row][c] = border_value
        if eos_col < max_grid_size:
            for r in range(pad_r, eos_row):
                if 0 <= r < max_grid_size:
                    padded[r][eos_col] = border_value
        return padded

    padded_x = translate_pad_and_add_border(x)
    padded_y = translate_pad_and_add_border(y)
    return padded_x, padded_y


def align_pred_to_y(
    pred: Optional[list[list[int]]],
    y: Optional[list[list[int]]],
    *,
    # kept for backwards compatibility; no longer used in alignment
    border_value: int = 10,
    pad_value: int = 11,
) -> tuple[Optional[list[list[int]]], dict[str, Any]]:
    """
    Align prediction to ground truth by taking the top-left crop with the same
    shape as `y` (when possible). This is intentionally simple.

    If pred is smaller than y in either dimension, we do not pad; stats will treat
    that as a shape mismatch.
    """
    info: dict[str, Any] = {
        "method": "top_left_to_y",
        "y_shape": list(grid_shape(y)),
        "pred_raw_shape": list(grid_shape(pred)),
        "pred_used_shape": list(grid_shape(pred)),
        "cropped": False,
        "can_align": False,
    }
    if pred is None:
        return None, info
    if y is None:
        return pred, info

    yh, yw = grid_shape(y)
    ph, pw = grid_shape(pred)
    if yh == 0 or yw == 0 or ph == 0 or pw == 0:
        return pred, info

    if ph >= yh and pw >= yw:
        used = [row[:yw] for row in pred[:yh]]
        uh, uw = grid_shape(used)
        info["can_align"] = True
        info["pred_used_shape"] = [uh, uw]
        info["cropped"] = (uh, uw) != (ph, pw)
        return used, info

    # cannot align (pred smaller)
    return pred, info


def compute_grid_match(a: Optional[list[list[int]]], b: Optional[list[list[int]]]) -> Optional[dict[str, Any]]:
    if a is None or b is None:
        return None
    if len(a) == 0 or len(b) == 0:
        return {"equal": a == b, "same_shape": a == b, "matches": 0, "total": 0, "accuracy": None}
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return {"equal": False, "same_shape": False, "matches": 0, "total": 0, "accuracy": None}

    matches = 0
    total = 0
    for r in range(len(a)):
        row_a = a[r]
        row_b = b[r]
        for c in range(len(row_a)):
            total += 1
            if int(row_a[c]) == int(row_b[c]):
                matches += 1
    return {
        "equal": matches == total,
        "same_shape": True,
        "matches": matches,
        "total": total,
        "accuracy": (matches / total) if total else None,
    }

