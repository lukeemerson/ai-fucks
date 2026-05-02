""":class:`CachedBackbone` -- content-addressable feature cache around any :class:`BackbonePort`.

Wraps an inner ``BackbonePort`` and persists per-image embeddings to disk so
repeat extractions (e.g. across Step 4 ablations that hold the backbone fixed
and only vary the head / calibrator / threshold) skip the expensive forward
pass entirely.

Key layout::

    {cache_dir}/{backbone_id}/{sha[:2]}/{sha}.npy

* ``cache_dir`` is supplied by the caller (e.g. ``runs/<ts>/features`` or
  ``~/.cache/harness/features``).
* ``backbone_id`` is the inner adapter's :attr:`identifier`. The wrapper
  raises :class:`AdapterError` at construction time if the inner backbone
  does not expose a non-empty ``identifier`` string -- without one we cannot
  safely scope cache entries and a backbone swap would silently return stale
  features. (Decision recorded in `PAPER_CHECKLIST.md` Step 3.5 spec.)
* ``sha`` is ``hashlib.sha256(image.tobytes()).hexdigest()`` over a single
  image's per-row tensor (post any caller-side preprocessing). Two-character
  prefix sharding keeps any single directory under ~256 children even for
  the full 112k NIH corpus.

Atomicity. Each cache write goes through a sibling ``*.npy.tmp`` file that is
opened directly as a binary file handle (sidestepping numpy's auto-append of
``.npy`` to a path-by-name argument), then renamed via :meth:`Path.replace`
which is atomic on POSIX. A crash mid-write therefore never leaves a partially
written ``.npy`` for a future run to mistakenly load.

Determinism. Same input + same inner backbone -> byte-identical output across
runs, regardless of whether each row is a hit or a miss. The contract tests
(``tests/harness/contract/test_backbone_contract.py::TestCachedBackboneContract``)
assert this end-to-end.

Corrupt-file handling. ``np.load`` can raise ``ValueError`` (bad magic),
``OSError`` (truncated header), or ``EOFError`` (zero-byte file) when an
on-disk cache entry was tampered with or partially restored from backup.
The wrapper catches these and re-raises :class:`AdapterError` with the
offending path so the failure stays inside the harness exception hierarchy
(per the "no silent failures" rule in `CLAUDE.md`).

Hexagonal placement. This module imports stdlib + numpy + ``harness.domain``
+ ``harness.ports``. It does NOT depend on torch, sklearn, fakes, or
composition.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from harness.domain.errors import AdapterError
from harness.ports.backbone import BackbonePort

__all__ = ["CachedBackbone"]


def _require_identifier(inner: BackbonePort) -> str:
    """Return ``inner.identifier`` if present and non-empty, else raise.

    The cache scopes its on-disk layout by backbone identifier; an unscoped
    cache would silently mix features from different backbones together,
    corrupting downstream ablations.
    """
    identifier = getattr(inner, "identifier", None)
    if not isinstance(identifier, str) or not identifier:
        raise AdapterError(
            "CachedBackbone requires the inner backbone to expose a non-empty "
            f"`identifier` string; got {identifier!r} from {type(inner).__name__}"
        )
    return identifier


def _hash_image(image: NDArray[np.float32]) -> str:
    """SHA-256 hex digest over the image's contiguous byte representation."""
    contiguous = np.ascontiguousarray(image)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


class CachedBackbone:
    """Disk-backed cache wrapping any :class:`BackbonePort`.

    Constructor arguments:

    * ``inner`` -- the wrapped :class:`BackbonePort`. Must expose a non-empty
      ``identifier`` string property (see module docstring for rationale).
    * ``cache_dir`` -- root directory under which cached feature files are
      written. Created on first write. Must already exist OR be creatable
      by the caller's process.

    The instance is stateless beyond the on-disk cache and the cached
    ``identifier`` / ``embedding_dim``. ``extract`` is safe to call from a
    single process; concurrent multi-process use is not a v1 requirement
    (atomic rename means concurrent writers cannot corrupt files but may
    duplicate work).
    """

    def __init__(self, *, inner: BackbonePort, cache_dir: Path) -> None:
        backbone_id = _require_identifier(inner)
        self._inner: BackbonePort = inner
        self._backbone_id: str = backbone_id
        self._cache_root: Path = Path(cache_dir) / backbone_id
        self._embedding_dim: int = int(inner.embedding_dim)
        self._identifier: str = f"cached:{backbone_id}"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def identifier(self) -> str:
        return self._identifier

    def extract(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return ``(N, embedding_dim)`` features, hitting the cache per row.

        Each row is hashed independently, looked up in the cache, and -- on
        miss -- forwarded to the inner backbone in a single batched call.
        Output rows preserve input batch order. The inner backbone's own
        validation (shape / dtype / channel-count checks) is exercised
        either via the miss-batch forward (most cases) or, when every row
        is cached, by a no-op short circuit; an explicit ndim check on the
        input ensures malformed batches always reach the inner adapter for
        validation rather than silently mis-hashing.

        On read, ``numpy.load`` failures (bad magic, truncated header,
        empty file) are translated to :class:`AdapterError` so callers
        always see a harness exception with the offending path.
        """
        # Validation pass. Forward malformed batches to the inner adapter so
        # shape / ndim / dtype errors surface with the canonical AdapterError
        # message; the cache does not duplicate validation rules.
        if images.ndim != 4:
            self._inner.extract(images)
            # Defensive: if the inner adapter accepted a non-4D tensor
            # (which would itself be a contract bug), fail loudly here
            # instead of mis-hashing a row.
            raise AdapterError(
                f"CachedBackbone expected 4-D NHWC tensor, got ndim={images.ndim} "
                f"(shape={images.shape})"
            )

        n = int(images.shape[0])
        if n == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        out = np.empty((n, self._embedding_dim), dtype=np.float32)
        miss_indices: list[int] = []
        miss_paths: list[Path] = []

        for i in range(n):
            sha = _hash_image(images[i])
            path = self._cache_path(sha)
            if path.is_file():
                try:
                    cached = np.load(path)
                except (ValueError, OSError, EOFError) as exc:
                    raise AdapterError(
                        f"corrupt cache file at {path}: {exc!r}"
                    ) from exc
                out[i] = cached
            else:
                miss_indices.append(i)
                miss_paths.append(path)

        if miss_indices:
            miss_batch = np.stack([images[i] for i in miss_indices], axis=0)
            miss_features = self._inner.extract(miss_batch)
            if miss_features.shape != (len(miss_indices), self._embedding_dim):
                raise AdapterError(
                    "inner backbone returned features of unexpected shape "
                    f"{miss_features.shape}; expected "
                    f"{(len(miss_indices), self._embedding_dim)}"
                )
            cast_features = miss_features.astype(np.float32, copy=False)
            for j, i in enumerate(miss_indices):
                feature_row = cast_features[j]
                self._write_atomic(miss_paths[j], feature_row)
                out[i] = feature_row

        return out

    # --------------------------------------------------------------- private

    def _cache_path(self, sha: str) -> Path:
        return self._cache_root / sha[:2] / f"{sha}.npy"

    @staticmethod
    def _write_atomic(path: Path, feature: NDArray[np.float32]) -> None:
        """Write ``feature`` to ``path`` via tmp + rename so partial writes never persist.

        ``np.save`` appends ``.npy`` when handed a path-by-name without that
        suffix, which would land the file at ``*.npy.tmp.npy``. We open the
        tmp file as a binary handle ourselves so the on-disk name is exactly
        what we asked for, then atomically rename onto the final path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npy.tmp")
        with tmp.open("wb") as fh:
            np.save(fh, feature, allow_pickle=False)
        tmp.replace(path)
