from __future__ import annotations

import argparse
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from .loaders import (
    align_pred_to_y,
    compute_grid_match,
    dataloader_translate_and_pad,
    load_preds_jsonl,
    load_test_records,
)


def _json_response(handler: BaseHTTPRequestHandler, obj: Any, *, status: int = 200) -> None:
    data = json.dumps(obj).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _text_response(handler: BaseHTTPRequestHandler, text: str, *, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _read_static(path: Path) -> bytes:
    return path.read_bytes()


def _guess_content_type(path: Path) -> str:
    if path.suffix == ".html":
        return "text/html; charset=utf-8"
    if path.suffix == ".js":
        return "text/javascript; charset=utf-8"
    if path.suffix == ".css":
        return "text/css; charset=utf-8"
    if path.suffix == ".svg":
        return "image/svg+xml"
    return "application/octet-stream"


class ArcViewerApp:
    def __init__(
        self,
        *,
        test_records: list[dict[str, Any]],
        preds_by_aug: dict[int, list[list[int]]],
        preds_by_aug_ex: dict[tuple[int, int], list[list[int]]],
        static_dir: Path,
        max_grid_size: int,
    ):
        # Grouped test records:
        #   puzzle_id -> aug_puzzle_idx -> example_idx -> record
        self.puzzle_to_augs: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
        # Convenience indexes
        self.aug_to_puzzle: dict[int, str] = {}
        self.aug_to_examples: dict[int, list[int]] = {}
        self.test_item: dict[tuple[int, int], dict[str, Any]] = {}

        for rec in test_records:
            pid = str(rec.get("puzzle_id", ""))
            aug_id = int(rec["aug_puzzle_idx"])
            ex_id = int(rec.get("example_idx", 0))
            self.puzzle_to_augs.setdefault(pid, {}).setdefault(aug_id, {})[ex_id] = rec
            self.aug_to_puzzle[aug_id] = pid
            self.test_item[(aug_id, ex_id)] = rec

        for pid, augs in self.puzzle_to_augs.items():
            for aug_id, ex_map in augs.items():
                self.aug_to_examples[aug_id] = sorted(ex_map.keys())

        self.preds_by_aug = preds_by_aug
        self.preds_by_aug_ex = preds_by_aug_ex
        self.static_dir = static_dir
        self.ids_sorted = sorted(self.aug_to_puzzle.keys())  # aug_puzzle_idx list
        self.ids_set = set(self.ids_sorted)
        self.max_grid_size = int(max_grid_size)

        # Precompute per-aug stats and per-puzzle aggregates for fast filtering + metrics.
        self.aug_stats: dict[int, dict[str, Any]] = {}
        self.puzzle_to_aug_ids: dict[str, list[int]] = {}

        total_matches = 0
        total_cells = 0
        total_with_pred = 0
        total_correct = 0
        total_shape_mismatch = 0
        total_cropped = 0

        for aug_id in self.ids_sorted:
            pid = self.aug_to_puzzle.get(aug_id, "")
            self.puzzle_to_aug_ids.setdefault(pid, []).append(aug_id)

            has_pred = (aug_id in self.preds_by_aug) or any(
                (aug_id, ex_id) in self.preds_by_aug_ex for ex_id in self.aug_to_examples.get(aug_id, [])
            )

            # Aggregate over all example_idx for this aug_id.
            ex_ids = self.aug_to_examples.get(aug_id, [])
            aug_equal = True
            aug_same_shape = True
            aug_matches = 0
            aug_cells = 0
            aug_any_cropped = False
            for ex_id in ex_ids:
                rec = self.test_item.get((aug_id, ex_id)) or {}
                y = rec.get("y")
                pred_raw = self.preds_by_aug_ex.get((aug_id, int(ex_id))) or self.preds_by_aug.get(aug_id)
                pred_used, pred_align = align_pred_to_y(pred_raw, y, border_value=10, pad_value=11)
                match = compute_grid_match(pred_used, y)
                if match is None:
                    aug_equal = False
                    aug_same_shape = False
                    continue
                aug_same_shape = aug_same_shape and bool(match["same_shape"])
                aug_equal = aug_equal and bool(match["equal"])
                if match["same_shape"]:
                    aug_matches += int(match["matches"] or 0)
                    aug_cells += int(match["total"] or 0)
                if pred_align.get("cropped"):
                    aug_any_cropped = True

            match = (
                {"equal": aug_equal, "same_shape": aug_same_shape, "matches": aug_matches, "total": aug_cells, "accuracy": (aug_matches / aug_cells) if aug_cells else None}
                if ex_ids
                else {"equal": False, "same_shape": False, "matches": 0, "total": 0, "accuracy": None}
            )
            same_shape = bool(match["same_shape"]) if match is not None else False
            equal = bool(match["equal"]) if match is not None else False
            matches = int(match["matches"]) if match is not None and match["matches"] is not None else 0
            cells = int(match["total"]) if match is not None and match["total"] is not None else 0
            acc = match["accuracy"] if match is not None else None

            if has_pred:
                total_with_pred += 1
                if not same_shape:
                    total_shape_mismatch += 1
                else:
                    total_matches += matches
                    total_cells += cells
                    if equal:
                        total_correct += 1
                if aug_any_cropped:
                    total_cropped += 1

            self.aug_stats[aug_id] = {
                "puzzle_id": pid,
                "has_pred": has_pred,
                "same_shape": same_shape,
                "equal": equal,
                "matches": matches,
                "total": cells,
                "accuracy": acc,
                "cropped": bool(aug_any_cropped),
            }

        self.global_summary = {
            "num_test": len(self.ids_sorted),
            "num_with_pred": total_with_pred,
            "num_shape_mismatch": total_shape_mismatch,
            "num_correct": total_correct,
            "num_cropped": total_cropped,
            "puzzle_accuracy": (total_correct / (total_with_pred - total_shape_mismatch)) if (total_with_pred - total_shape_mismatch) > 0 else None,
            "cell_accuracy": (total_matches / total_cells) if total_cells > 0 else None,
            "total_cells_compared": total_cells,
        }

        self.puzzle_summaries: dict[str, dict[str, Any]] = {}
        for pid, aug_ids in self.puzzle_to_aug_ids.items():
            pmatches = 0
            pcells = 0
            pwith = 0
            pcorrect = 0
            pshape_mismatch = 0
            pcropped = 0
            for aug_id in aug_ids:
                st = self.aug_stats[aug_id]
                if st["has_pred"]:
                    pwith += 1
                    if not st["same_shape"]:
                        pshape_mismatch += 1
                    else:
                        pmatches += int(st["matches"])
                        pcells += int(st["total"])
                        if st["equal"]:
                            pcorrect += 1
                    if st.get("cropped"):
                        pcropped += 1
            denom = pwith - pshape_mismatch
            self.puzzle_summaries[pid] = {
                "puzzle_id": pid,
                "count": len(aug_ids),
                "num_with_pred": pwith,
                "num_shape_mismatch": pshape_mismatch,
                "num_correct": pcorrect,
                "num_cropped": pcropped,
                "puzzle_accuracy": (pcorrect / denom) if denom > 0 else None,
                "cell_accuracy": (pmatches / pcells) if pcells > 0 else None,
                # accuracy over all augs under this puzzle_id (missing preds count as incorrect)
                "aug_accuracy": (pcorrect / len(aug_ids)) if len(aug_ids) > 0 else None,
            }

    def list_aug_ids(
        self,
        *,
        q: Optional[str] = None,
        limit: int = 500,
        only_correct: bool = False,
        only_with_pred: bool = False,
        puzzle_id: Optional[str] = None,
    ) -> list[int]:
        limit = max(1, min(int(limit), 5000))
        q = (q or "").strip()
        q_is_num = q.isdigit()
        ql = q.lower()
        out: list[int] = []

        base_iter = self.puzzle_to_aug_ids.get(puzzle_id, []) if puzzle_id else self.ids_sorted
        for aug_id in base_iter:
            st = self.aug_stats.get(aug_id)
            if st is None:
                continue
            if only_with_pred and not st["has_pred"]:
                continue
            if only_correct and not st["equal"]:
                continue

            if q:
                if q_is_num:
                    if not str(aug_id).startswith(q):
                        continue
                else:
                    if ql not in str(st.get("puzzle_id", "")).lower():
                        continue

            out.append(aug_id)
            if len(out) >= limit:
                break
        return out

    def list_puzzles(
        self,
        *,
        q: Optional[str] = None,
        limit: int = 500,
        only_correct: bool = False,
        only_with_pred: bool = False,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 5000))
        q = (q or "").strip().lower()
        out: list[dict[str, Any]] = []
        for pid in sorted(self.puzzle_to_augs.keys()):
            if q and q not in pid.lower():
                continue
            summ = self.puzzle_summaries[pid]
            # Filter at puzzle-level:
            # - only_correct: keep puzzles containing at least one correct aug
            # - only_with_pred: keep puzzles containing at least one predicted aug
            if only_correct and int(summ.get("num_correct", 0)) <= 0:
                continue
            if only_with_pred and int(summ.get("num_with_pred", 0)) <= 0:
                continue
            out.append(summ)
            if len(out) >= limit:
                break
        return out

    def list_augs_for_puzzle(self, puzzle_id: str, *, limit: int = 2000) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 5000))
        augs = self.puzzle_to_augs.get(puzzle_id, {})
        out: list[dict[str, Any]] = []
        for aug_id in sorted(augs.keys()):
            st = self.aug_stats.get(aug_id) or {}
            out.append(
                {
                    "aug_puzzle_idx": aug_id,
                    "example_idxs": self.aug_to_examples.get(aug_id, []),
                    **st,
                }
            )
            if len(out) >= limit:
                break
        return out

    def list_examples_for_aug(self, aug_puzzle_idx: int) -> list[dict[str, Any]]:
        ex_ids = self.aug_to_examples.get(int(aug_puzzle_idx), [])
        pid = self.aug_to_puzzle.get(int(aug_puzzle_idx), "")
        out: list[dict[str, Any]] = []
        for ex_id in ex_ids:
            rec = self.test_item.get((int(aug_puzzle_idx), int(ex_id))) or {}
            meta = {k: v for (k, v) in rec.items() if k not in ("x", "y")}
            out.append({"example_idx": int(ex_id), "meta": meta, "puzzle_id": pid})
        return out

    def get_aug_bundle(self, aug_id: int) -> dict[str, Any]:
        """
        Return all examples for an aug_puzzle_idx in one payload so the UI can render
        every example on the same page.
        """
        aug_id = int(aug_id)
        pid = self.aug_to_puzzle.get(aug_id, "")
        ex_ids = self.aug_to_examples.get(aug_id, [])
        if not ex_ids:
            raise KeyError(aug_id)

        pred_aug = self.preds_by_aug.get(aug_id)
        aug_stat = self.aug_stats.get(aug_id) or {}

        examples: list[dict[str, Any]] = []
        for ex_id in ex_ids:
            rec = self.test_item.get((aug_id, int(ex_id)))
            if rec is None:
                continue
            meta = {k: v for (k, v) in rec.items() if k not in ("x", "y")}
            x_raw = rec.get("x")
            y_raw = rec.get("y")
            x_dl, y_dl = dataloader_translate_and_pad(
                x_raw,
                y_raw,
                max_grid_size=self.max_grid_size,
                translate=False,
                pad_value=11,
                border_value=10,
                aug_puzzle_idx=int(aug_id),
            )
            pred_raw = self.preds_by_aug_ex.get((aug_id, int(ex_id))) or pred_aug
            pred_used, pred_alignment = align_pred_to_y(pred_raw, y_raw, border_value=10, pad_value=11)
            match = compute_grid_match(pred_used, y_raw)
            examples.append(
                {
                    "example_idx": int(ex_id),
                    "meta": meta,
                    "x_raw": x_raw,
                    "y_raw": y_raw,
                    "x_dl": x_dl,
                    "y_dl": y_dl,
                    "pred_raw": pred_raw,
                    "pred_used": pred_used,
                    "pred_alignment": pred_alignment,
                    "match": match,
                }
            )

        return {
            "puzzle_id": pid,
            "aug_puzzle_idx": aug_id,
            "aug_stats": aug_stat,
            "pred_raw": pred_aug,
            "example_idxs": ex_ids,
            "examples": examples,
            "has_pred": (pred_aug is not None) or any((aug_id, ex_id) in self.preds_by_aug_ex for ex_id in ex_ids),
        }

    def summarize_aug_ids(self, aug_ids: list[int]) -> dict[str, Any]:
        matches = 0
        cells = 0
        with_pred = 0
        correct = 0
        shape_mismatch = 0
        cropped = 0
        for aug_id in aug_ids:
            st = self.aug_stats.get(aug_id)
            if not st:
                continue
            if st["has_pred"]:
                with_pred += 1
                if not st["same_shape"]:
                    shape_mismatch += 1
                else:
                    matches += int(st["matches"])
                    cells += int(st["total"])
                    if st["equal"]:
                        correct += 1
                if st.get("cropped"):
                    cropped += 1
        denom = with_pred - shape_mismatch

        return {
            "num_items": len(aug_ids),
            "num_with_pred": with_pred,
            "num_shape_mismatch": shape_mismatch,
            "num_correct": correct,
            "num_cropped": cropped,
            "puzzle_accuracy": (correct / denom) if denom > 0 else None,
            "cell_accuracy": (matches / cells) if cells > 0 else None,
            "total_cells_compared": cells,
        }

    def get_bundle(self, aug_id: int, example_idx: Optional[int] = None) -> dict[str, Any]:
        aug_id = int(aug_id)
        if example_idx is None:
            exs = self.aug_to_examples.get(aug_id, [])
            example_idx = exs[0] if exs else 0
        example_idx = int(example_idx)
        rec = self.test_item.get((aug_id, example_idx))
        if rec is None:
            raise KeyError((aug_id, example_idx))
        pred_raw = self.preds_by_aug_ex.get((aug_id, example_idx)) or self.preds_by_aug.get(aug_id)

        # Send metadata without the potentially-large x/y duplication in meta.
        meta = {k: v for (k, v) in rec.items() if k not in ("x", "y")}
        x_raw = rec.get("x")
        y_raw = rec.get("y")

        # viewer-side reconstruction of "dataloader output" (TranslateAndPad, translate=False)
        x_dl, y_dl = dataloader_translate_and_pad(
            x_raw,
            y_raw,
            max_grid_size=self.max_grid_size,
            translate=False,
            pad_value=11,
            border_value=10,
            aug_puzzle_idx=int(aug_id),
        )

        pred, pred_alignment = align_pred_to_y(pred_raw, y_raw, border_value=10, pad_value=11)
        match = compute_grid_match(pred, y_raw)
        return {
            "aug_puzzle_idx": aug_id,
            "example_idx": example_idx,
            "example_idxs": self.aug_to_examples.get(aug_id, []),
            "meta": meta,
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x": x_raw,  # backwards compat
            "y": y_raw,  # backwards compat
            "x_dl": x_dl,
            "y_dl": y_dl,
            "pred_raw": pred_raw,
            "pred": pred,  # kept for backwards compatibility (this is the "used"/cropped pred)
            "pred_used": pred,
            "pred_alignment": pred_alignment,
            "pred_key": ("preds" if "preds" in rec else ("y_pred" if "y_pred" in rec else None)),
            "match": match,
            "has_pred": pred_raw is not None,
        }


def make_handler(app: ArcViewerApp):
    static_dir = app.static_dir

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            # keep server output quiet (still logs errors)
            return

        def do_GET(self) -> None:
            try:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                qs = urllib.parse.parse_qs(parsed.query)

                if path == "/api/health":
                    return _json_response(
                        self,
                        {
                            "ok": True,
                            "num_test": len(app.ids_sorted),
                            "num_preds": len(app.preds_by_aug) + len(app.preds_by_aug_ex),
                            "summary": app.global_summary,
                        },
                    )

                if path == "/api/summary":
                    # optional filters matching /api/ids
                    q = (qs.get("q", [""])[0] or "").strip()
                    only_correct = (qs.get("only_correct", ["0"])[0] == "1")
                    only_with_pred = (qs.get("only_with_pred", ["0"])[0] == "1")
                    puzzle_id = (qs.get("puzzle_id", [""])[0] or "").strip() or None
                    aug_ids = app.list_aug_ids(
                        q=q,
                        limit=5000,
                        only_correct=only_correct,
                        only_with_pred=only_with_pred,
                        puzzle_id=puzzle_id,
                    )
                    return _json_response(
                        self,
                        {"global": app.global_summary, "filtered": app.summarize_aug_ids(aug_ids)},
                    )

                if path == "/api/ids":
                    q = (qs.get("q", [""])[0] or "").strip()
                    limit = int(qs.get("limit", ["500"])[0])
                    only_correct = (qs.get("only_correct", ["0"])[0] == "1")
                    only_with_pred = (qs.get("only_with_pred", ["0"])[0] == "1")
                    puzzle_id = (qs.get("puzzle_id", [""])[0] or "").strip() or None

                    ids = app.list_aug_ids(
                        q=q,
                        limit=limit,
                        only_correct=only_correct,
                        only_with_pred=only_with_pred,
                        puzzle_id=puzzle_id,
                    )
                    items = [
                        {
                            "aug_puzzle_idx": aug_id,
                            **(app.aug_stats.get(aug_id) or {}),
                        }
                        for aug_id in ids
                    ]
                    return _json_response(
                        self,
                        {
                            "ids": ids,
                            "items": items,
                            "count": len(ids),
                            "total": len(app.ids_sorted),
                            "summary": {"global": app.global_summary, "filtered": app.summarize_aug_ids(ids)},
                        },
                    )

                if path == "/api/puzzles":
                    q = (qs.get("q", [""])[0] or "").strip()
                    limit = int(qs.get("limit", ["500"])[0])
                    only_correct = (qs.get("only_correct", ["0"])[0] == "1")
                    only_with_pred = (qs.get("only_with_pred", ["0"])[0] == "1")
                    puzzles = app.list_puzzles(q=q, limit=limit, only_correct=only_correct, only_with_pred=only_with_pred)
                    return _json_response(
                        self,
                        {
                            "puzzles": puzzles,
                            "count": len(puzzles),
                            "total": len(app.puzzle_to_augs),
                            "summary": app.global_summary,
                        },
                    )

                if path == "/api/augs":
                    puzzle_id = (qs.get("puzzle_id", [""])[0] or "").strip()
                    if not puzzle_id:
                        return _json_response(self, {"error": "missing puzzle_id"}, status=400)
                    limit = int(qs.get("limit", ["2000"])[0])
                    augs = app.list_augs_for_puzzle(puzzle_id, limit=limit)
                    return _json_response(self, {"puzzle_id": puzzle_id, "augs": augs, "count": len(augs)})

                if path == "/api/examples":
                    aug_s = (qs.get("aug_puzzle_idx", [""])[0] or "").strip()
                    if not aug_s.isdigit():
                        return _json_response(self, {"error": "missing/invalid aug_puzzle_idx"}, status=400)
                    examples = app.list_examples_for_aug(int(aug_s))
                    return _json_response(self, {"aug_puzzle_idx": int(aug_s), "examples": examples, "count": len(examples)})

                if path.startswith("/api/puzzle/"):
                    aug_s = path.split("/api/puzzle/", 1)[1]
                    if not aug_s or not aug_s.isdigit():
                        return _json_response(self, {"error": "invalid aug_puzzle_idx"}, status=400)
                    aug_id = int(aug_s)
                    example_idx_q = (qs.get("example_idx", [""])[0] or "").strip()
                    example_idx = int(example_idx_q) if example_idx_q.isdigit() else None
                    try:
                        bundle = app.get_bundle(aug_id, example_idx=example_idx)
                    except KeyError:
                        return _json_response(
                            self,
                            {
                                "error": "not found",
                                "aug_puzzle_idx": aug_id,
                                "example_idx": example_idx,
                            },
                            status=404,
                        )
                    return _json_response(self, bundle)

                if path.startswith("/api/aug/"):
                    aug_s = path.split("/api/aug/", 1)[1]
                    if not aug_s or not aug_s.isdigit():
                        return _json_response(self, {"error": "invalid aug_puzzle_idx"}, status=400)
                    aug_id = int(aug_s)
                    try:
                        bundle = app.get_aug_bundle(aug_id)
                    except KeyError:
                        return _json_response(self, {"error": "not found", "aug_puzzle_idx": aug_id}, status=404)
                    return _json_response(self, bundle)

                # Static
                if path == "/":
                    file_path = static_dir / "index.html"
                else:
                    # prevent path traversal
                    rel = path.lstrip("/")
                    if ".." in rel or rel.startswith("/"):
                        return _text_response(self, "bad path", status=400)
                    file_path = static_dir / rel

                if not file_path.exists() or not file_path.is_file():
                    return _text_response(self, "not found", status=404)
                data = _read_static(file_path)
                self.send_response(200)
                self.send_header("Content-Type", _guess_content_type(file_path))
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                return _json_response(self, {"error": "server_error", "detail": str(e)}, status=500)

    return Handler


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Local web app to visualize ARC-AGI puzzles + predictions.")
    parser.add_argument("--preds", default="preds.jsonl", help="Path to preds JSONL (aug_puzzle_idx + preds/y_pred 30x30).")
    parser.add_argument("--test", required=True, help="Path to test JSONL (records with aug_puzzle_idx + x + y + metadata).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--max_grid_size", default=30, type=int, help="Max grid size used by dataloader padding (default: 30).")
    args = parser.parse_args(argv)

    test_records, _ = load_test_records(args.test, verbose=True)
    preds_by_aug, preds_by_aug_ex, _ = load_preds_jsonl(args.preds, verbose=True)

    static_dir = Path(__file__).resolve().parent / "static"
    app = ArcViewerApp(
        test_records=test_records,
        preds_by_aug=preds_by_aug,
        preds_by_aug_ex=preds_by_aug_ex,
        static_dir=static_dir,
        max_grid_size=args.max_grid_size,
    )

    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(app))
    url = f"http://{args.host}:{args.port}/"
    print(f"[arc_viewer] ready: {url}")
    print("[arc_viewer] tip: Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

