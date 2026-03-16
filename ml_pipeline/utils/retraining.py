from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


Direction = Literal["max", "min"]


@dataclass(frozen=True)
class MetricSpec:
    path: str
    direction: Direction
    label: str
    min_delta: float = 1e-6


class TrainingLock:
    def __init__(self, lock_path: Path):
        self.lock_path = Path(lock_path)
        self._fd: int | None = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            payload = {
                "pid": os.getpid(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            os.write(self._fd, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        except FileExistsError as exc:
            raise RuntimeError(f"Ya existe un entrenamiento en curso: '{self.lock_path.name}'") from exc

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    def __enter__(self) -> "TrainingLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def build_run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{suffix}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested_value(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_better(candidate_value: float, incumbent_value: float, direction: Direction, min_delta: float) -> bool:
    if direction == "max":
        return candidate_value > incumbent_value + min_delta
    return candidate_value < incumbent_value - min_delta


def _is_worse(candidate_value: float, incumbent_value: float, direction: Direction, min_delta: float) -> bool:
    if direction == "max":
        return candidate_value < incumbent_value - min_delta
    return candidate_value > incumbent_value + min_delta


def compare_metric_reports(
    candidate_report: dict[str, Any],
    incumbent_report: dict[str, Any] | None,
    metric_specs: list[MetricSpec],
    force_replace: bool = False,
) -> dict[str, Any]:
    if incumbent_report is None:
        return {
            "promoted": True,
            "reason": "no_hay_modelo_anterior",
            "force_replace": force_replace,
            "metrics": [asdict(s) for s in metric_specs],
            "comparisons": [],
        }

    comparisons: list[dict[str, Any]] = []
    first_decisive_reason = "sin_diferencia_relevante"
    promoted = False

    for spec in metric_specs:
        cand_raw = get_nested_value(candidate_report, spec.path)
        inc_raw = get_nested_value(incumbent_report, spec.path)
        row = {
            "path": spec.path,
            "label": spec.label,
            "direction": spec.direction,
            "min_delta": spec.min_delta,
            "candidate": cand_raw,
            "incumbent": inc_raw,
            "decision": "skip",
        }

        if cand_raw is None or inc_raw is None:
            comparisons.append(row)
            continue

        try:
            cand = float(cand_raw)
            inc = float(inc_raw)
        except Exception:
            comparisons.append(row)
            continue

        if _is_better(cand, inc, spec.direction, spec.min_delta):
            row["decision"] = "better"
            comparisons.append(row)
            promoted = True
            first_decisive_reason = f"mejora_en_{spec.label}"
            break

        if _is_worse(cand, inc, spec.direction, spec.min_delta):
            row["decision"] = "worse"
            comparisons.append(row)
            promoted = False
            first_decisive_reason = f"empeora_en_{spec.label}"
            break

        row["decision"] = "tie"
        comparisons.append(row)

    if force_replace:
        promoted = True
        if first_decisive_reason == "sin_diferencia_relevante":
            first_decisive_reason = "force_replace"
        else:
            first_decisive_reason = f"force_replace__{first_decisive_reason}"

    return {
        "promoted": promoted,
        "reason": first_decisive_reason,
        "force_replace": force_replace,
        "metrics": [asdict(s) for s in metric_specs],
        "comparisons": comparisons,
    }


def save_candidate_artifacts(
    *,
    artifacts_dir: Path,
    model_name: str,
    run_id: str,
    save_model_fn,
    report: dict[str, Any],
) -> Path:
    candidate_dir = artifacts_dir / "_candidates" / model_name / run_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    save_model_fn(candidate_dir)
    write_json(candidate_dir / "train_report.json", report)
    return candidate_dir


def promote_candidate_if_needed(
    *,
    artifacts_dir: Path,
    model_name: str,
    candidate_dir: Path,
    decision: dict[str, Any],
) -> dict[str, Any]:
    champion_dir = artifacts_dir / model_name
    archive_root = artifacts_dir / "_archive" / model_name
    rejected_root = artifacts_dir / "_rejected" / model_name

    promotion_payload = {
        **decision,
        "candidate_dir": str(candidate_dir),
        "champion_dir": str(champion_dir),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(candidate_dir / "promotion_decision.json", promotion_payload)

    final_model_dir = candidate_dir
    archived_previous_dir = None

    if decision.get("promoted"):
        if champion_dir.exists():
            archive_root.mkdir(parents=True, exist_ok=True)
            archived_previous_dir = archive_root / f"replaced_{build_run_id(model_name)}"
            shutil.move(str(champion_dir), str(archived_previous_dir))

        champion_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidate_dir), str(champion_dir))
        final_model_dir = champion_dir
        write_json(
            final_model_dir / "promotion_decision.json",
            promotion_payload | {
                "archived_previous_dir": str(archived_previous_dir) if archived_previous_dir else None
            },
        )
    else:
        rejected_root.mkdir(parents=True, exist_ok=True)
        rejected_dir = rejected_root / candidate_dir.name
        if rejected_dir.exists():
            shutil.rmtree(rejected_dir)
        shutil.move(str(candidate_dir), str(rejected_dir))
        final_model_dir = rejected_dir

    return {
        "promoted": bool(decision.get("promoted")),
        "reason": decision.get("reason", "sin_diferencia_relevante"),
        "candidate_dir": str(final_model_dir),
        "champion_dir": str(champion_dir),
        "archived_previous_dir": str(archived_previous_dir) if archived_previous_dir else None,
    }