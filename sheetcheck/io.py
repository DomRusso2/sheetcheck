"""Data access for Vesuvius Challenge open data.

Two things live here:

* :class:`Surface` -- a ``tifxyz`` traced papyrus surface, i.e. a 2D grid of
  (z, y, x) volume coordinates plus a validity mask.
* :class:`Volume` -- a multiscale OME-Zarr CT volume, sampled lazily by
  chunk so that a whole scroll never has to be downloaded.

Both accept either a local path or an ``s3://`` / bucket-relative path in the
public ``vesuvius-challenge-open-data`` bucket.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
import tifffile

BUCKET = "vesuvius-challenge-open-data"

# tifxyz marks cells with no surface as -1 in all three coordinate planes.
INVALID = -1.0


@lru_cache(maxsize=1)
def _fs():
    import s3fs

    return s3fs.S3FileSystem(anon=True)


def _read_bytes(path: str) -> bytes:
    if path.startswith(("http://", "https://")):
        import requests

        r = requests.get(path, timeout=120)
        r.raise_for_status()
        return r.content
    if "://" in path or path.startswith(BUCKET):
        return _fs().cat(path.replace("s3://", ""))
    with open(path, "rb") as fh:
        return fh.read()


def read_json(path: str) -> dict:
    return json.loads(_read_bytes(path))


@dataclass
class Surface:
    """A traced papyrus surface in ``tifxyz`` form.

    ``points`` has shape ``(nv, nu, 3)`` holding (z, y, x) volume voxel
    coordinates; ``valid`` is the matching boolean mask.  ``scale`` is the
    tifxyz grid scale, so one grid step spans ``1 / scale`` volume voxels.
    """

    points: np.ndarray
    valid: np.ndarray
    meta: dict
    name: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.points.shape[:2]

    @property
    def grid_step_vox(self) -> float:
        """Nominal spacing between adjacent grid cells, in volume voxels."""
        scale = self.meta.get("scale") or [0.05, 0.05]
        return float(1.0 / np.mean(scale))

    @classmethod
    def load(cls, path: str, cache_dir: str | None = None) -> "Surface":
        """Load a tifxyz surface, optionally caching the decoded array on disk.

        Full-scroll meshes are large (the merged PHerc1667 trace is a
        2061 x 30097 grid, ~745 MB decoded) and take minutes to pull from S3,
        which makes iterating on the detector painful.
        """
        path = path.rstrip("/")

        cache_path = None
        if cache_dir is not None:
            import hashlib
            import os

            os.makedirs(cache_dir, exist_ok=True)
            key = hashlib.sha1(path.encode()).hexdigest()[:16]
            cache_path = os.path.join(cache_dir, f"tifxyz_{key}.npz")
            if os.path.exists(cache_path):
                d = np.load(cache_path, allow_pickle=True)
                return cls(points=d["points"], valid=d["valid"],
                           meta=json.loads(str(d["meta"])),
                           name=path.split("/")[-1])

        planes = []
        for c in ("z", "y", "x"):
            raw = _read_bytes(f"{path}/{c}.tif")
            planes.append(tifffile.imread(io.BytesIO(raw)).astype(np.float32))
        pts = np.stack(planes, axis=-1)
        # A cell is valid only if it was written; unwritten cells are -1.
        valid = np.all(pts > INVALID + 1e-6, axis=-1)
        try:
            meta = read_json(f"{path}/meta.json")
        except Exception:  # noqa: BLE001 - meta.json is advisory
            meta = {}

        if cache_path is not None:
            np.savez(cache_path, points=pts, valid=valid,
                     meta=json.dumps(meta))
        return cls(points=pts, valid=valid, meta=meta, name=path.split("/")[-1])

    def normals(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit surface normals per grid cell, plus a validity mask.

        Normals come from central differences of the coordinate grid along u
        and v.  Cells whose 4-neighbourhood is not fully valid are masked out
        rather than being computed from a one-sided difference, because a
        one-sided normal at a surface boundary is exactly where tracing
        artefacts live and would pollute the statistics.
        """
        p = self.points
        du = np.zeros_like(p)
        dv = np.zeros_like(p)
        du[:, 1:-1] = p[:, 2:] - p[:, :-2]
        dv[1:-1, :] = p[2:, :] - p[:-2, :]

        n = np.cross(dv, du)
        norm = np.linalg.norm(n, axis=-1, keepdims=True)

        v = self.valid
        ok = np.zeros_like(v)
        ok[1:-1, 1:-1] = (
            v[1:-1, 1:-1] & v[1:-1, 2:] & v[1:-1, :-2] & v[2:, 1:-1] & v[:-2, 1:-1]
        )
        ok &= norm[..., 0] > 1e-6

        with np.errstate(invalid="ignore", divide="ignore"):
            n = n / np.where(norm > 1e-6, norm, 1.0)
        return n, ok


class Volume:
    """Lazily-sampled multiscale OME-Zarr CT volume."""

    def __init__(self, path: str, level: int = 0):
        self.path = path.rstrip("/")
        self.level = level
        self._arr = None
        self._meta: dict | None = None

    @property
    def array(self):
        if self._arr is None:
            import zarr

            store = self.path
            if "://" not in store:
                store = f"s3://{store}"
            if store.startswith("s3://"):
                import s3fs

                fs = s3fs.S3FileSystem(anon=True)
                mapper = fs.get_mapper(store.replace("s3://", ""))
                root = zarr.open(mapper, mode="r")
            else:
                root = zarr.open(store, mode="r")
            self._arr = root[str(self.level)]
        return self._arr

    @property
    def voxel_size_um(self) -> float:
        """Physical voxel size at this pyramid level, in micrometres."""
        if self._meta is None:
            try:
                self._meta = read_json(f"{self.path}/metadata.json")
            except Exception:  # noqa: BLE001
                self._meta = {}
        px_mm = (
            self._meta.get("scan", {})
            .get("tomo", {})
            .get("acquisition", {})
            .get("detector", {})
            .get("samplePixelSize")
        )
        if px_mm is not None:
            return float(px_mm) * 1000.0 * (2**self.level)

        # Not every published volume carries samplePixelSize (the PHerc1667
        # 1.129 um rescan does not), but the resolution is always encoded in
        # the volume directory name.
        import re

        m = re.search(r"-([0-9]+\.[0-9]+)um", self.path.rstrip("/").split("/")[-1])
        if m:
            return float(m.group(1)) * (2**self.level)
        raise ValueError(f"cannot determine voxel size for {self.path}")

    @property
    def shape(self):
        return self.array.shape

    def to_level(self, coords_l0: np.ndarray) -> np.ndarray:
        """Convert level-0 volume coordinates (as stored in tifxyz) to this level.

        Surface meshes are always expressed in the full-resolution frame, so
        sampling any downsampled pyramid level requires this rescale.
        """
        return np.asarray(coords_l0, dtype=np.float64) / float(2**self.level)

    def read_box(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fetch an axis-aligned block, clipped to the volume.

        Returns the block and the clipped ``lo`` corner so callers can convert
        global coordinates into block-local ones.  Fetching one block and
        sampling it many times locally is far cheaper than issuing a chunk
        request per ray.
        """
        arr = self.array
        shp = np.array(arr.shape, dtype=np.int64)
        lo = np.maximum(np.floor(lo).astype(np.int64), 0)
        hi = np.minimum(np.ceil(hi).astype(np.int64) + 1, shp)
        if np.any(hi <= lo):
            return np.zeros((0, 0, 0), dtype=arr.dtype), lo
        block = np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        return block, lo

    @staticmethod
    def sample_box(block: np.ndarray, lo: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Nearest-neighbour sample of ``coords`` inside a block from :meth:`read_box`."""
        idx = np.rint(coords).astype(np.int64) - lo
        shp = np.array(block.shape, dtype=np.int64)
        inside = np.all((idx >= 0) & (idx < shp), axis=-1)
        out = np.zeros(coords.shape[:-1], dtype=np.float32)
        if inside.any():
            sel = idx[inside]
            out[inside] = block[sel[..., 0], sel[..., 1], sel[..., 2]]
        return out

    def sample(self, coords: np.ndarray) -> np.ndarray:
        """Nearest-neighbour sample at ``coords`` of shape ``(..., 3)`` in (z, y, x).

        Chunk-aware: points are grouped by chunk so each chunk is fetched at
        most once, which is what makes streaming from S3 tolerable.  Points
        outside the volume return 0.
        """
        arr = self.array
        flat = coords.reshape(-1, 3)
        out = np.zeros(len(flat), dtype=np.float32)

        idx = np.rint(flat).astype(np.int64)
        shp = np.array(arr.shape, dtype=np.int64)
        inside = np.all((idx >= 0) & (idx < shp), axis=1)
        if not inside.any():
            return out.reshape(coords.shape[:-1])

        chunks = np.array(arr.chunks, dtype=np.int64)
        cidx = idx[inside] // chunks
        keys, inv = np.unique(cidx, axis=0, return_inverse=True)
        sub = idx[inside]
        vals = np.zeros(len(sub), dtype=np.float32)

        for k, key in enumerate(keys):
            sel = inv == k
            lo = key * chunks
            hi = np.minimum(lo + chunks, shp)
            block = np.asarray(
                arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            )
            loc = sub[sel] - lo
            vals[sel] = block[loc[:, 0], loc[:, 1], loc[:, 2]]

        out[inside] = vals
        return out.reshape(coords.shape[:-1])


def s3_ls(path: str) -> list[str]:
    return sorted(_fs().ls(path.replace("s3://", ""), detail=False))


def segment_meshes(scroll: str, segment: str) -> Iterable[str]:
    base = f"{BUCKET}/{scroll}/segments/{segment}/mesh"
    for p in s3_ls(base):
        if p.endswith(".tifxyz"):
            yield p
