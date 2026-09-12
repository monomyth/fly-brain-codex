"""Versioned shared assets, verified downloads, and atomic metadata writes."""
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

SOURCE_PAGE = "https://male-cns.janelia.org/download/"
BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
ATTRIBUTION = """# MaleCNS v1.0 attribution

Source: https://male-cns.janelia.org/download/
Male CNS project: FlyEM (HHMI Janelia), University of Cambridge (Dept. of Zoology),
MRC Laboratory of Molecular Biology, Google Research, and collaborators.
Data license: Creative Commons Attribution 4.0 International (CC BY 4.0).
https://creativecommons.org/licenses/by/4.0/

Transformations: identified-neuron selection, stable integer ID indexing,
directed edge aggregation, incoming-weight normalization, and annotated
sensory/output population selection. See manifest.json for the precise recipe,
source hashes, exclusions, and generated file checksums.
This derived controller is not a physiologically validated fly-brain simulation.
"""


@dataclass(frozen=True)
class Asset:
    name: str
    upstream: str
    size: int
    sha256: str
    generation: str

    @property
    def url(self):
        return BASE_URL + self.upstream + "?generation=" + self.generation


ASSETS = (
    Asset("annotations.feather", "body-annotations-male-cns-v1.0-minconf-0.5.feather", 14483314,
          "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2", "1780494878811468"),
    Asset("neurotransmitters.feather", "body-neurotransmitters-male-cns-v1.0.feather", 43282834,
          "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621", "1780894899156750"),
    Asset("edges.feather", "connectome-weights-male-cns-v1.0-minconf-0.5.feather", 1051241946,
          "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1", "1780494887545976"),
)


REFERENCE_HOME = Path("~/code/data/malecns").expanduser()
PROJECT_DATA_HOME = Path(__file__).resolve().parents[2] / "data"


def home(value=None):
    """Generated artifacts belong to the Codex project; raw downloads are linked in."""
    requested=Path(value or os.environ.get("FLY_BRAIN_DATA_HOME") or os.environ.get("MALECNS_HOME") or PROJECT_DATA_HOME).expanduser()
    # Preserve old commands while preventing writes back into the download share.
    if requested.resolve()==REFERENCE_HOME.resolve():requested=PROJECT_DATA_HOME
    return requested.resolve()


def local_artifact(value):
    """Resolve legacy generated-data paths after the project-local migration."""
    path=Path(value).expanduser()
    try:relative=path.relative_to(REFERENCE_HOME)
    except ValueError:return path
    if relative.parts and relative.parts[0] in {"checkpoints","datasets","circuits","motor-circuits","prepared","feature-cache","runs","exports","visualizations"}:
        candidate=PROJECT_DATA_HOME/relative
        # Other projects' files in the reference share are not ours to redirect.
        if candidate.exists() or relative.parts[0]!="prepared":return candidate
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def verify_asset(path, asset):
    path = Path(path)
    if path.stat().st_size != asset.size:
        raise ValueError(f"Incomplete or wrong asset: {path}")
    actual = sha256_file(path)
    if actual != asset.sha256:
        raise ValueError(f"SHA-256 mismatch: {path}; existing data were not overwritten")
    return {"name": asset.name, "bytes": asset.size, "sha256": actual,
            "generation": asset.generation, "url": asset.url}


def ensure_assets(root=None, download=False):
    root = home(root)
    results = []
    for asset in ASSETS:
        path = root / "v1.0" / asset.name
        with locked(root / ".locks" / asset.name):
            if not path.exists():
                if not download:
                    raise FileNotFoundError(f"Missing {path}; run 'fly-brain assets --download'")
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(path.suffix + ".partial")
                try:
                    with urllib.request.urlopen(asset.url, timeout=60) as source, partial.open("wb") as target:
                        generation = source.headers.get("x-goog-generation")
                        if generation != asset.generation:
                            raise ValueError(f"Upstream generation changed for {asset.name}")
                        shutil.copyfileobj(source, target, 8 * 1024 * 1024)
                        target.flush()
                        os.fsync(target.fileno())
                    verify_asset(partial, asset)
                    os.replace(partial, path)
                finally:
                    partial.unlink(missing_ok=True)
            results.append(verify_asset(path, asset))
    report = {"schema": "malecns-assets-v1", "root": str(root), "files": results,
              "source": SOURCE_PAGE, "license": "CC-BY-4.0", "verified": True}
    write_json(root / "reference-verification.json", report)
    return report


def checked_files(directory, manifest):
    for name, digest in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe manifest path")
        if sha256_file(Path(directory) / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")


def artifact_manifest(directory, metadata):
    directory = Path(directory)
    metadata = dict(metadata)
    metadata["files"] = {str(p.relative_to(directory)): sha256_file(p)
                         for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "manifest.json"}
    write_json(directory / "manifest.json", metadata)
    return metadata
