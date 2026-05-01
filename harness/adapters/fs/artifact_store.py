"""``FilesystemArtifactStore`` -- on-disk ``ArtifactStorePort`` adapter.

Layout
------

Given a ``root_dir``, artifacts are written to::

    <root>/model_card.json
    <root>/thresholds.json
    <root>/metric_report.json
    <root>/predictions/<name>.csv

All write methods are idempotent: re-writing the same logical artifact
overwrites the file in place and returns the same absolute path string.
The root (and any required subdirectories) are created on first write.

Serialization
~~~~~~~~~~~~~

* ``ModelCard`` / ``ThresholdSet`` / ``MetricReport`` are written as JSON
  via explicit ``dataclasses.asdict``-style traversal. ``MetricInterval``
  fields are emitted with keys ``{point, low, high}`` (not ``lower``/``upper``)
  to match the on-disk schema. ``ThresholdSet`` flattens to a ``per_label``
  mapping rather than parallel arrays. Datetimes serialize via ``isoformat``.

* ``Predictions`` are written as CSV with a header row whose first column
  is ``sample_id`` followed by the label names; data rows contain the
  sample id followed by integer 0/1 values.

Only stdlib + numpy are used.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from harness.domain.types import (
    MetricInterval,
    MetricReport,
    ModelCard,
    PerClassMetric,
    Predictions,
    ThresholdSet,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# JSON-serializable value alias used for the typed payload returned by the
# ``_*_to_dict`` helpers. Container types (dict/list) are spelled with
# ``JSONValue`` recursively so that invariance does not force callers to
# upcast nested dictionaries manually.
type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]


def _interval_to_dict(interval: MetricInterval) -> dict[str, JSONValue]:
    return {
        "point": interval.point,
        "low": interval.lower,
        "high": interval.upper,
    }


def _per_class_to_dict(metric: PerClassMetric) -> dict[str, JSONValue]:
    return {
        "label": metric.label,
        "f1": _interval_to_dict(metric.f1),
        "auroc": _interval_to_dict(metric.auroc),
        "auprc": _interval_to_dict(metric.auprc),
        "support": metric.support,
    }


def _metric_report_to_dict(report: MetricReport) -> dict[str, JSONValue]:
    per_class: list[JSONValue] = [_per_class_to_dict(c) for c in report.per_class]
    return {
        "macro_f1": _interval_to_dict(report.macro_f1),
        "macro_auroc": _interval_to_dict(report.macro_auroc),
        "macro_auprc": _interval_to_dict(report.macro_auprc),
        "per_class": per_class,
        "n_bootstrap": report.n_bootstrap,
        "seed": report.seed,
    }


def _model_card_to_dict(card: ModelCard) -> dict[str, JSONValue]:
    label_names: list[JSONValue] = list(card.label_names)
    return {
        "name": card.name,
        "version": card.version,
        "created_at": card.created_at.isoformat(),
        "backbone": card.backbone,
        "head": card.head,
        "calibrator": card.calibrator,
        "threshold_method": card.threshold_method,
        "label_names": label_names,
        "train_size": card.train_size,
        "val_size": card.val_size,
        "test_size": card.test_size,
        "config_hash": card.config_hash,
        "metrics": _metric_report_to_dict(card.metrics),
        "notes": card.notes,
    }


def _thresholds_to_dict(ts: ThresholdSet) -> dict[str, JSONValue]:
    per_label: dict[str, JSONValue] = {
        name: float(t)
        for name, t in zip(ts.label_names, ts.thresholds, strict=True)
    }
    label_names: list[JSONValue] = list(ts.label_names)
    return {
        "per_label": per_label,
        "label_names": label_names,
        "method": ts.method,
        "shrinkage": ts.shrinkage,
        "clamp_lo": ts.clamp_lo,
        "clamp_hi": ts.clamp_hi,
    }


def _json_default(obj: object) -> str:
    """Last-resort JSON encoder for unexpected non-stdlib types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    msg = f"object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


class FilesystemArtifactStore:
    """On-disk artifact store rooted at ``root_dir``."""

    def __init__(self, root_dir: Path | str) -> None:
        self._root: Path = Path(root_dir).resolve()

    # --- helpers -----------------------------------------------------------

    def _ensure_root(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    def _write_json(self, relative: str, payload: Mapping[str, JSONValue]) -> str:
        self._ensure_root()
        path = self._root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            dict(payload),
            indent=2,
            sort_keys=False,
            default=_json_default,
        )
        path.write_text(text, encoding="utf-8")
        return str(path.resolve())

    # --- writes ------------------------------------------------------------

    def write_model_card(self, card: ModelCard) -> str:
        """Persist ``card`` to ``<root>/model_card.json`` and return its path."""
        return self._write_json("model_card.json", _model_card_to_dict(card))

    def write_thresholds(self, thresholds: ThresholdSet) -> str:
        """Persist ``thresholds`` to ``<root>/thresholds.json``."""
        return self._write_json("thresholds.json", _thresholds_to_dict(thresholds))

    def write_metric_report(self, report: MetricReport) -> str:
        """Persist ``report`` to ``<root>/metric_report.json``."""
        return self._write_json("metric_report.json", _metric_report_to_dict(report))

    def write_predictions(self, preds: Predictions, name: str) -> str:
        """Persist ``preds`` as CSV under ``<root>/predictions/<name>.csv``."""
        self._ensure_root()
        pred_dir = self._root / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        path = pred_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["sample_id", *preds.label_names])
            values = preds.values
            for row_idx, sid in enumerate(preds.sample_ids):
                row = values[row_idx].tolist()
                writer.writerow([sid, *(int(v) for v in row)])
        return str(path.resolve())
