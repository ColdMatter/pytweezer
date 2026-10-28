"""MOTMaster run zips: finding the one a run just wrote, and adding camera frames.

MOTMaster saves each run as ``{element}{ddMMMyy}{batch}_{nnn}.zip`` in a folder per
day. Frames from the camera whose key is MOTMaster's element go into that zip;
any other camera's go into a sibling zip named with its key in place of the
element, so ``analysis3.readFromZip`` (which loads every ``.tif`` in a zip) only
ever sees one camera's frames.
"""

import io
import json
import re
import zipfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
#: Entries ``analysis3.readFromZip`` recognises by suffix; a sibling needs copies.
_MOTMASTER_TEXT_SUFFIXES = ("parameters.txt", "hardwareReport.txt")


def day_folder(data_root: Path, day: date) -> Path:
    """MOTMaster's folder for ``day``, e.g. ``data_root/2026/10Oct/27``."""
    return (
        Path(data_root)
        / f"{day.year}"
        / f"{day.month:02d}{_MONTHS[day.month - 1]}"
        / f"{day.day:02d}"
    )


def _is_run_zip(path: Path, element: str) -> bool:
    return (
        re.fullmatch(
            rf"{re.escape(element)}\d{{2}}[A-Za-z]{{3}}\d{{4}}_\d{{3}}\.zip", path.name
        )
        is not None
    )


def snapshot(folders: Iterable[Path], element: str) -> set[Path]:
    """The run zips for ``element`` already in ``folders``."""
    return {
        path
        for folder in folders
        if Path(folder).is_dir()
        for path in Path(folder).iterdir()
        if _is_run_zip(path, element)
    }


def find_new_zip(before: set[Path], folders: Iterable[Path], element: str) -> Path:
    """The single run zip in ``folders`` that was not in ``before``.

    Raises:
        RuntimeError: if there is no new zip, or more than one.
    """
    folders = list(folders)
    new = sorted(snapshot(folders, element) - before)
    if len(new) != 1:
        where = ", ".join(str(folder) for folder in folders)
        found = "no new run zip" if not new else f"{len(new)} new run zips"
        raise RuntimeError(
            f"Expected MOTMaster to have saved one {element} run zip; found {found} "
            f"in {where}. Is saving on in MOTMaster?"
        )
    return new[0]


def _tiff_bytes(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(frame).astype(np.uint16)).save(buffer, format="TIFF")
    return buffer.getvalue()


def _write_frames(
    archive: zipfile.ZipFile, key: str, frames: np.ndarray, settings: dict
) -> None:
    for index, frame in enumerate(frames):
        archive.writestr(f"{key}_{index:04d}.tif", _tiff_bytes(frame))
    archive.writestr(
        f"{key}_camera_settings.json", json.dumps(settings, indent=1, default=str)
    )


def inject_frames(zip_path: Path, key: str, frames: np.ndarray, settings: dict) -> None:
    """Append ``frames`` as ``{key}_0000.tif``... plus the camera's settings."""
    with zipfile.ZipFile(zip_path, "a") as archive:
        _write_frames(archive, key, frames, settings)


def write_sibling(
    main_zip: Path, element: str, key: str, frames: np.ndarray, settings: dict
) -> Path:
    """Write ``key``'s frames to a new zip beside ``main_zip``, with its parameters.

    Raises:
        FileExistsError: if the sibling already exists.
    """
    main_zip = Path(main_zip)
    sibling = main_zip.with_name(key + main_zip.name[len(element) :])
    with zipfile.ZipFile(main_zip) as source, zipfile.ZipFile(sibling, "x") as archive:
        for name in source.namelist():
            if name.endswith(_MOTMASTER_TEXT_SUFFIXES):
                archive.writestr(name, source.read(name))
        _write_frames(archive, key, frames, settings)
    return sibling
