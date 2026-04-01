import os

from pytweezer.servers import tweezerpath


EXPERIMENTS_ROOT = os.path.normpath(
    os.path.join(tweezerpath, "pytweezer", "experiments")
)
_EXPERIMENTS_MARKER = "/experiments/"


def canonical_experiment_filepath(filepath):
    """Return path relative to the experiments directory, using forward slashes."""
    if not filepath:
        return filepath

    raw = str(filepath).strip().replace("\\", "/")
    lower = raw.lower()
    marker_idx = lower.find(_EXPERIMENTS_MARKER)
    if marker_idx >= 0:
        rel = raw[marker_idx + len(_EXPERIMENTS_MARKER) :]
        return rel.lstrip("/")

    if os.path.isabs(raw):
        abs_path = os.path.normpath(raw)
        try:
            common = os.path.commonpath([abs_path, EXPERIMENTS_ROOT])
        except ValueError:
            common = ""
        if common == EXPERIMENTS_ROOT:
            rel = os.path.relpath(abs_path, EXPERIMENTS_ROOT)
            return rel.replace("\\", "/")

    return raw.lstrip("/")


def resolve_experiment_filepath(filepath):
    """Resolve either canonical or absolute experiment path to an absolute local path."""
    if not filepath:
        return filepath

    raw = str(filepath).strip()
    normalized = raw.replace("\\", "/")
    if _EXPERIMENTS_MARKER in normalized.lower():
        rel = canonical_experiment_filepath(normalized)
        return os.path.normpath(os.path.join(EXPERIMENTS_ROOT, rel))

    if os.path.isabs(raw):
        return os.path.normpath(raw)

    rel = canonical_experiment_filepath(raw)
    return os.path.normpath(os.path.join(EXPERIMENTS_ROOT, rel))
