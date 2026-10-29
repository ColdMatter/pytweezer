import re
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pytweezer.notebook import zips


def read_zip_like_analysis3(path: Path) -> tuple[np.ndarray, dict]:
    """A copy of ``analysis3.readFromZip``'s logic (EDMSuite), minus its file lookup."""

    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]

    archive = zipfile.ZipFile(path, "r")
    images, params = [], {}
    for name in sorted(archive.namelist(), key=natural_keys):
        if name[-3:] == "tif":
            with archive.open(name) as file:
                images.append(np.array(Image.open(file), dtype=float))
        if name[-14:] == "parameters.txt":
            with archive.open(name) as file:
                for line in file.readlines():
                    key, value = line.split(b"\t")
                    params[key.decode("utf-8")] = float(value)
    return np.array(images), params


def fake_motmaster_zip(folder: Path, name: str = "CaF27Oct2600_012") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}_parameters.txt", "dummy\t0\n")
        archive.writestr(f"{name}_hardwareReport.txt", "laser\t1\n")
        archive.writestr(f"{name}.cs", "// script")
    return path


def frames(n: int, shape=(4, 5)) -> np.ndarray:
    return np.stack([np.full(shape, i, dtype=np.uint16) for i in range(n)])


def test_day_folder_matches_motmaster(tmp_path):
    assert (
        zips.day_folder(tmp_path, date(2026, 10, 27))
        == tmp_path / "2026" / "10Oct" / "27"
    )


def test_injected_caf_zip_reads_back_only_caf_frames(tmp_path):
    main = fake_motmaster_zip(tmp_path)
    zips.inject_frames(main, "CaF", frames(3), {"gain_db": 12.0})
    images, params = read_zip_like_analysis3(main)
    assert images.shape == (3, 4, 5)
    assert images.dtype == float
    assert [image[0, 0] for image in images] == [0.0, 1.0, 2.0]
    assert params == {"dummy": 0.0}


def test_frame_order_survives_more_than_ten_frames(tmp_path):
    main = fake_motmaster_zip(tmp_path)
    zips.inject_frames(main, "CaF", frames(12), {})
    images, _ = read_zip_like_analysis3(main)
    assert [image[0, 0] for image in images] == list(range(12))


def test_sibling_zip_holds_its_frames_and_motmaster_parameters(tmp_path):
    main = fake_motmaster_zip(tmp_path)
    sibling = zips.write_sibling(main, "CaF", "Rb", frames(2), {})
    assert sibling == tmp_path / "Rb27Oct2600_012.zip"
    images, params = read_zip_like_analysis3(sibling)
    assert images.shape[0] == 2
    assert params == {"dummy": 0.0}
    assert not any(name.startswith("Rb") for name in zipfile.ZipFile(main).namelist())


def test_sibling_zip_is_never_overwritten(tmp_path):
    main = fake_motmaster_zip(tmp_path)
    zips.write_sibling(main, "CaF", "Rb", frames(2), {})
    with pytest.raises(FileExistsError):
        zips.write_sibling(main, "CaF", "Rb", frames(2), {})


def test_new_zip_is_found(tmp_path):
    folder = tmp_path / "27"
    fake_motmaster_zip(folder, "CaF27Oct2600_011")
    before = zips.snapshot([folder], "CaF")
    new = fake_motmaster_zip(folder, "CaF27Oct2600_012")
    assert zips.find_new_zip(before, [folder], "CaF") == new


def test_no_new_zip_is_an_error(tmp_path):
    folder = tmp_path / "27"
    before = zips.snapshot([folder], "CaF")
    with pytest.raises(RuntimeError, match="27"):
        zips.find_new_zip(before, [folder], "CaF")


def test_two_new_zips_is_an_error(tmp_path):
    folder = tmp_path / "27"
    before = zips.snapshot([folder], "CaF")
    fake_motmaster_zip(folder, "CaF27Oct2600_011")
    fake_motmaster_zip(folder, "CaF27Oct2600_012")
    with pytest.raises(RuntimeError, match="27"):
        zips.find_new_zip(before, [folder], "CaF")


def test_a_sibling_zip_is_not_taken_for_motmasters(tmp_path):
    folder = tmp_path / "27"
    before = zips.snapshot([folder], "CaF")
    main = fake_motmaster_zip(folder, "CaF27Oct2600_012")
    zips.write_sibling(main, "CaF", "Rb", frames(1), {})
    assert zips.find_new_zip(before, [folder], "CaF") == main


def test_a_zip_written_after_midnight_is_found(tmp_path):
    day_27, day_28 = tmp_path / "27", tmp_path / "28"
    before = zips.snapshot([day_27, day_28], "CaF")
    new = fake_motmaster_zip(day_28, "CaF28Oct2600_000")
    assert zips.find_new_zip(before, [day_27, day_28], "CaF") == new


def test_settings_file_is_not_mistaken_for_parameters(tmp_path):
    main = fake_motmaster_zip(tmp_path)
    zips.inject_frames(main, "CaF", frames(1), {"gain_db": 12.0})
    added = [
        name for name in zipfile.ZipFile(main).namelist() if name.startswith("CaF_")
    ]
    assert added and not any(name.endswith("parameters.txt") for name in added)
