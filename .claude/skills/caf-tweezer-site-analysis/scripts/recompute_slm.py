"""Recompute runs' SLM holograms and simulated light as the rig did; run on PH-BEAST (needs its GPU).

usage (on PH-BEAST, from the pytweezer clone):
    .venv\\Scripts\\python.exe recompute_slm.py OUT_DIR RID [RID ...]

Reads each run's arguments and its generator_settings and envelope constants,
and writes OUT_DIR/rid<RID>_slm_sim.npz with the hologram (mask, x_um, y_um,
weights, grids, simulated_trap_plane, ...) and blaze_um. Prints the recorded
and recomputed uniformity, min/max ratio and MSE, which should agree.
"""

import sys

import numpy as np

from pytweezer.experiment import Dataset
from pytweezer.helper_scripts import array_hologram


def recompute(out_dir: str, rid: int) -> None:
    ds = Dataset(rid, results=["uniformity", "min_max_ratio", "mse"])
    args = dict(ds.arguments)
    settings = dict(ds.constants["generator_settings"])
    blaze_um = (args["blaze_x"] * 1e6, args["blaze_y"] * 1e6)
    generator = array_hologram.make_generator(
        {
            **settings,
            "slm_res": tuple(settings["slm_res"]),
            "zernike_coeff_dict": {
                int(k): float(v) for k, v in settings["zernike_coeff_dict"].items()
            },
            "fresnel_f_mm": args["fresnel_focal_length"] * 1e3,
            "blaze_dx_dy_um": blaze_um,
        }
    )
    hologram = array_hologram.array_hologram(
        generator,
        rows=int(args["rows"]),
        columns=int(args["columns"]),
        spacing_um=args["spacing"] * 1e6,
        angle_deg=float(np.degrees(args["angle"])),
        envelope=ds.constants["envelope"] if args["compensate_envelope"] else None,
        blaze_um=blaze_um,
        fresnel_focal_length_mm=args["fresnel_focal_length"] * 1e3,
        zernike=bool(args["zernike"]),
        pupil_radius_um=args["pupil_radius"] * 1e6,
        pupil_dump_um=(args["pupil_dump_x"] * 1e6, args["pupil_dump_y"] * 1e6),
        iterations=int(args["iterations"]),
        damping=float(args["damping"]),
        phase_randomness=float(args["phase_randomness"]),
    )
    np.savez_compressed(
        f"{out_dir}/rid{rid}_slm_sim.npz",
        blaze_um=np.array(blaze_um),
        **{k: np.asarray(v) for k, v in hologram.items()},
    )
    recorded = ds.get_data("dict")
    for name in ("uniformity", "min_max_ratio", "mse"):
        print(
            f"rid {rid} {name}: recorded {np.nanmean(recorded[name]):.6g}, "
            f"recomputed {hologram[name]:.6g}"
        )


if __name__ == "__main__":
    for rid in sys.argv[2:]:
        recompute(sys.argv[1], int(rid))
