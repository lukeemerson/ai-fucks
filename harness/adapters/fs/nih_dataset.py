"""Composition adapter wiring CSV + image I/O for NIH ChestX-ray14.

This is the Wave 3 layer of the NIH adapter stack. It composes
:class:`harness.adapters.fs.nih_csv.NIHCsvIngestor` with
:class:`harness.adapters.fs.nih_images.NIHImageLoader` to satisfy the
:class:`harness.ports.dataset.DatasetPort` protocol.

Design references:

* ``harness/adapters/fs/docs/NIH_DATASET_SPEC.md`` sections 4 (module
  layout), 5 (image processing), 6 (patient metadata), 7 (Sample mapping),
  8 (config), 10 (contract integration), 11 (smoke test), 13
  (edge-case policies).
* ``harness/ports/dataset.py`` -- exact port surface.

Public surface:

* :class:`NIHDatasetConfig` -- frozen configuration dataclass.
* :class:`NIHDataset` -- adapter implementing :class:`DatasetPort`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from harness.adapters.fs.nih_csv import NIH14_LABELS, NIHCsvIngestor, NIHRecord
from harness.adapters.fs.nih_images import ImageLoaderConfig, NIHImageLoader
from harness.domain.errors import AdapterError, ConfigError, DataError
from harness.domain.types import Dataset, Sample

__all__ = ["NIHDataset", "NIHDatasetConfig"]


_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NIHDatasetConfig:
    """Configuration for :class:`NIHDataset`.

    See spec section 8 for full semantics. ``__post_init__`` resolves
    ``csv_path`` and ``images_dir`` to absolute paths (defensive against
    working-directory drift, per spec §13.7) and creates ``disk_cache_dir``
    if requested.
    """

    csv_path: Path
    images_dir: Path
    image_size: tuple[int, int] = (224, 224)
    cache_size: int = 1024
    disk_cache_dir: Path | None = None
    strict_missing_images: bool = True
    name: str = "nih-cxr14"

    def __post_init__(self) -> None:
        if not self.csv_path.is_file():
            raise ConfigError(f"NIH CSV not found: {self.csv_path}")
        if not self.images_dir.is_dir():
            raise ConfigError(f"NIH images dir not found: {self.images_dir}")
        h, w = self.image_size
        if h <= 0 or w <= 0:
            raise ConfigError(
                f"image_size dimensions must be positive, got {self.image_size}"
            )
        if self.cache_size < 0:
            raise ConfigError(
                f"cache_size must be >= 0, got {self.cache_size}"
            )

        # Resolve to absolute paths so downstream behaviour is independent
        # of the calling thread's CWD (see spec §13.7).
        object.__setattr__(self, "csv_path", self.csv_path.resolve())
        object.__setattr__(self, "images_dir", self.images_dir.resolve())

        if self.disk_cache_dir is not None:
            resolved_cache = self.disk_cache_dir.resolve()
            try:
                resolved_cache.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise AdapterError(
                    f"cannot create disk_cache_dir {resolved_cache}: {exc}"
                ) from exc
            object.__setattr__(self, "disk_cache_dir", resolved_cache)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class NIHDataset:
    """:class:`DatasetPort` adapter for the NIH ChestX-ray14 corpus.

    The adapter ingests the CSV at construction time, validates each
    referenced PNG against ``strict_missing_images``, and builds an
    immutable :class:`Dataset`. Image bytes are read on demand via the
    composed :class:`NIHImageLoader`.
    """

    __slots__ = (
        "_config",
        "_dataset",
        "_image_loader",
        "_index_by_id",
        "_known_refs",
        "_ref_to_path",
    )

    def __init__(self, config: NIHDatasetConfig) -> None:
        self._config = config

        ingestor = NIHCsvIngestor()
        records = ingestor.ingest(config.csv_path)

        loader_config = ImageLoaderConfig(
            images_dir=config.images_dir,
            image_size=config.image_size,
            cache_size=config.cache_size,
            disk_cache_dir=config.disk_cache_dir,
        )
        self._image_loader = NIHImageLoader(loader_config)

        kept_records, kept_paths = self._filter_records(records)

        samples = tuple(
            self._build_sample(rec, path)
            for rec, path in zip(kept_records, kept_paths, strict=True)
        )

        self._dataset = Dataset(
            name=config.name,
            label_names=NIH14_LABELS,
            samples=samples,
        )

        self._index_by_id: dict[str, int] = {
            s.sample_id: i for i, s in enumerate(samples)
        }
        # Map absolute-path image_ref -> resolved Path. Stored once at
        # construction so :meth:`get_image_bytes` is independent of the
        # caller's working directory (spec §13.7, review C1).
        self._ref_to_path: dict[str, Path] = {
            s.image_ref: p for s, p in zip(samples, kept_paths, strict=True)
        }
        self._known_refs: frozenset[str] = frozenset(self._ref_to_path)

    # ------------------------------------------------------------------ port
    def load(self) -> Dataset:
        """Return the immutable :class:`Dataset` built at construction."""
        return self._dataset

    def get_image_bytes(self, image_ref: str) -> bytes:
        """Return raw on-disk PNG bytes for ``image_ref``.

        ``image_ref`` must be one of the absolute paths produced at
        construction time (i.e. the value of some
        :attr:`Sample.image_ref`). Reading is performed directly against
        the stored absolute path so the call is independent of the
        process's current working directory.

        Raises:
            DataError: If ``image_ref`` is not a known sample reference or
                the underlying file cannot be read.
        """
        path = self._ref_to_path.get(image_ref)
        if path is None:
            raise DataError(f"unknown image_ref: {image_ref!r}")
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise DataError(
                f"NIH image not found: {image_ref} (resolved {path})"
            ) from exc
        except OSError as exc:
            raise DataError(
                f"NIH image unreadable: {image_ref} (resolved {path}): {exc}"
            ) from exc

    # -------------------------------------------------------------- helpers
    def __len__(self) -> int:
        return len(self._dataset.samples)

    def sample_ids(self) -> tuple[str, ...]:
        """Return the tuple of sample ids in CSV row order."""
        return tuple(s.sample_id for s in self._dataset.samples)

    def get_sample(self, sample_id: str) -> Sample:
        """Return the :class:`Sample` for ``sample_id``.

        Raises:
            DataError: If ``sample_id`` is not present in the dataset.
        """
        idx = self._index_by_id.get(sample_id)
        if idx is None:
            raise DataError(f"unknown sample_id: {sample_id!r}")
        return self._dataset.samples[idx]

    @property
    def label_names(self) -> tuple[str, ...]:
        """Return the canonical NIH-14 label vocabulary."""
        return self._dataset.label_names

    # -------------------------------------------------------------- private
    def _filter_records(
        self, records: tuple[NIHRecord, ...]
    ) -> tuple[tuple[NIHRecord, ...], tuple[Path, ...]]:
        """Resolve each record's PNG path; drop or fail on missing files."""
        images_dir = self._config.images_dir
        kept_records: list[NIHRecord] = []
        kept_paths: list[Path] = []
        missing: list[str] = []

        for record in records:
            path = images_dir / record.image_index
            if path.is_file():
                kept_records.append(record)
                kept_paths.append(path)
            else:
                missing.append(record.image_index)

        if missing:
            if self._config.strict_missing_images:
                preview = missing[:20]
                raise DataError(
                    f"{len(missing)} CSV rows reference missing PNGs "
                    f"under {images_dir}; first {len(preview)}: {preview!r}"
                )
            _LOGGER.info(
                "dropped %d CSV rows: missing image files under %s",
                len(missing),
                images_dir,
            )

        return tuple(kept_records), tuple(kept_paths)

    @staticmethod
    def _build_sample(record: NIHRecord, image_path: Path) -> Sample:
        absolute = str(image_path.resolve())
        metadata = MappingProxyType(
            {
                "follow_up": str(record.follow_up),
                "patient_age": str(record.age),
                "patient_gender": record.gender,
                "view_position": record.view_position,
                "width": str(record.width),
                "height": str(record.height),
                "pixel_spacing_x": str(record.pixel_spacing_x),
                "pixel_spacing_y": str(record.pixel_spacing_y),
            }
        )
        return Sample(
            sample_id=record.image_index,
            patient_id=record.patient_id,
            image_ref=absolute,
            labels=record.label_vector,
            metadata=metadata,
        )
