"""Which Fresnel focal length puts the tweezer spot most in focus, from a Fresnel scan.

usage: uv run python fresnel_focus.py OUT_DIR RID [CENTRE_X CENTRE_Y]

Fits a Gaussian spot to each step's mean image − background within 12 px of
the array centre: the predicted centre for files with the site recordings,
otherwise CENTRE_X CENTRE_Y (pixels of the frame). Errors come from resampling
the step's shots. Writes OUT_DIR/rid<RID>_focus.png: each step's image around
the spot above peak brightness and width against focal length.
"""

import sys

import matplotlib
import numpy as np
from scipy import ndimage, optimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytweezer.analyses.default.caf_tweezer_sites import missing, site_positions
from pytweezer.experiment.archive import fetch_measurement

FIT_RADIUS = 12
CROP = 13
RESAMPLES = 40


def spot_fit(mean, x0, y0):
    """``(peak, x, y, sigma, background)`` of a Gaussian fitted near ``(x0, y0)``."""
    yy, xx = np.indices(mean.shape)
    near = np.hypot(xx - x0, yy - y0) <= FIT_RADIUS

    def residual(p):
        peak, x, y, sigma, background = p
        model = background + peak * np.exp(
            -((xx[near] - x) ** 2 + (yy[near] - y) ** 2) / (2 * sigma**2)
        )
        return model - mean[near]

    return optimize.least_squares(
        residual,
        [500, x0, y0, 2.5, np.median(mean)],
        bounds=([0, x0 - 6, y0 - 6, 0.7, -1e4], [1e5, x0 + 6, y0 + 6, 8, 1e4]),
    ).x


def main(out_dir, rid, centre=None):
    m = fetch_measurement(
        rid, results=["hamcam_image", "hamcam_background", "slm_array"]
    )
    if centre is None:
        if missing(m):
            sys.exit(f"rid {rid}: {missing(m)} Give CENTRE_X CENTRE_Y.")
        array = m.constants["slm_arrays"]["arrays"][int(m.results["slm_array"][0])]
        y0, x0 = site_positions(m, array).mean(axis=0)
    else:
        x0, y0 = centre
    frames = np.asarray(m.results["hamcam_image"], float) - np.asarray(
        m.results["hamcam_background"], float
    )
    focal_mm = np.round(np.asarray(m.points["fresnel_focal_length"]) * 1e3).astype(int)
    steps = sorted(set(focal_mm))
    rng = np.random.default_rng(0)
    fits = []
    for f in steps:
        shots = np.flatnonzero(focal_mm == f)
        best = spot_fit(frames[shots].mean(0), x0, y0)
        resampled = np.array(
            [
                spot_fit(frames[rng.choice(shots, len(shots))].mean(0), x0, y0)
                for _ in range(RESAMPLES)
            ]
        )
        fits.append(
            (
                f,
                best[0],
                resampled[:, 0].std(),
                best[3],
                resampled[:, 3].std(),
                best[1],
                best[2],
            )
        )
        print(
            f"{f} mm: peak {best[0]:.0f} ± {resampled[:, 0].std():.0f} counts, "
            f"sigma {best[3]:.2f} ± {resampled[:, 3].std():.2f} px, "
            f"centre ({best[1]:.1f}, {best[2]:.1f}), {len(shots)} shots"
        )
    fits = np.array(fits)
    brightest = int(fits[np.argmax(fits[:, 1]), 0])

    fig = plt.figure(figsize=(14, 5.6))
    grid = fig.add_gridspec(
        2, len(steps), height_ratios=[1, 2.2], hspace=0.12, wspace=0.08
    )
    cx, cy = int(np.rint(x0)), int(np.rint(y0))
    for j, f in enumerate(steps):
        mean = ndimage.gaussian_filter(frames[focal_mm == f].mean(0), 1.0)
        crop = mean[cy - CROP : cy + CROP + 1, cx - CROP : cx + CROP + 1]
        ax = fig.add_subplot(grid[0, j])
        ax.imshow(crop - np.median(crop), cmap="viridis", vmin=-150, vmax=900)
        ax.set_title(
            f"{f} mm", fontsize=9, fontweight="bold" if f == brightest else "normal"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(grid[1, :])
    twin = ax.twinx()
    ax.errorbar(
        fits[:, 0],
        fits[:, 1],
        fits[:, 2],
        fmt="o-",
        color="tab:green",
        capsize=3,
        label="peak brightness",
    )
    twin.errorbar(
        fits[:, 0] + 1.5,
        fits[:, 3],
        fits[:, 4],
        fmt="s--",
        color="tab:purple",
        capsize=3,
        label="width σ",
    )
    ax.axvline(brightest, color="k", lw=0.6, ls=":")
    ax.set_xlabel("Fresnel focal length (mm)")
    ax.set_ylabel("peak (counts)", color="tab:green")
    twin.set_ylabel("width σ (px)", color="tab:purple")
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="upper right")
    ax.text(
        0.01,
        0.95,
        "Gaussian fit to each step's mean image; errors from resampling the shots",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    fig.suptitle(
        f"rid {rid}: tweezer spot per Fresnel focal length (mean image − background, smoothed 1 px)",
        fontsize=11,
        y=0.98,
    )
    out = f"{out_dir}/rid{rid}_focus.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(f"brightest at {brightest} mm; {out}")


if __name__ == "__main__":
    centre = tuple(map(float, sys.argv[3:5])) if len(sys.argv) >= 5 else None
    main(sys.argv[1], int(sys.argv[2]), centre)
