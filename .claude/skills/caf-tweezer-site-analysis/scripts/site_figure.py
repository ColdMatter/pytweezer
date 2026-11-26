"""Figure and numbers of the caf_tweezer_sites analysis of one run, straight from its file.

usage: uv run python site_figure.py OUT_DIR RID

Needs a file with the site recordings (written by commit 81784db onwards).
"""

import sys

import matplotlib
import numpy as np
from matplotlib.patches import Circle
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytweezer.analyses.default.caf_tweezer_sites import (
    SiteSettings,
    caf_tweezer_sites,
    missing,
    needed_results,
)
from pytweezer.analysis.site_photometry import score_sites
from pytweezer.experiment.archive import fetch_measurement

APERTURE = 4


def standard_error(values, axis=0):
    return values.std(axis=axis, ddof=1) / np.sqrt(values.shape[axis])


def main(out_dir: str, rid: int) -> None:
    header = fetch_measurement(rid, results=[])
    reason = missing(header)
    if reason:
        sys.exit(f"rid {rid}: {reason}")
    m = fetch_measurement(
        rid, results=[*needed_results(header), "uniformity", "min_max_ratio"]
    )
    args = m.arguments
    wide = caf_tweezer_sites(m, SiteSettings())
    tight = caf_tweezer_sites(m, SiteSettings(ring_inner=5, ring_outer=8))

    frames = np.asarray(m.results["hamcam_image"], float)
    if "hamcam_background" in m.results:
        frames = frames - np.asarray(m.results["hamcam_background"], float)
    centre = wide.sites.mean(axis=0)
    at_centre = score_sites(frames, centre[None]).z[:, 0]

    rows, columns, spacing_um = (
        int(args["rows"]),
        int(args["columns"]),
        args["spacing"] * 1e6,
    )
    print(
        f"rid {rid}: {wide.shots} shots, {rows}x{columns} at {spacing_um:g} um, "
        f"Fresnel {args['fresnel_focal_length'] * 1e3:g} mm, "
        f"frame median {np.median(wide.offset):+.0f} counts"
    )
    origin_x, origin_y = np.asarray(m.constants["hamcam_roi_origin"], float)
    print(
        f"array centre: frame px (x {centre[1]:.1f}, y {centre[0]:.1f}), sensor px "
        f"(x {centre[1] + origin_x:.1f}, y {centre[0] + origin_y:.1f}): "
        f"z {at_centre.mean():.2f} ± {standard_error(at_centre):.2f}"
    )
    for label, r in (("8-12", wide), ("5-8", tight)):
        reached = int((r.combined_chance >= r.combined.mean()).sum())
        print(
            f"ring {label} px: per-site z {np.round(r.z.mean(0), 2).tolist()} "
            f"± {np.round(standard_error(r.z), 2).tolist()}; "
            f"combined {r.combined.mean():.2f} ± {standard_error(r.combined):.2f}, "
            f"{reached}/{len(r.combined_chance)} random empty sets reach it "
            f"(they scatter by {r.combined_chance.std():.2f})"
        )

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    x0, x1, y0, y1 = wide.trap_plane_extent
    ax.imshow(wide.trap_plane, cmap="magma", extent=(x0, x1, y1, y0))
    ax.plot(*wide.traps_um.T, "c+", ms=10)
    for k, (x, y) in enumerate(wide.traps_um):
        ax.text(x + 0.3, y - 0.3, str(k), color="c", fontsize=9)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(
        f"rid {rid} simulated trap plane ({rows}×{columns}, {spacing_um:g} µm)\n"
        f"uniformity {np.nanmean(m.results['uniformity']):.3f}, "
        f"min/max {np.nanmean(m.results['min_max_ratio']):.3f}",
        fontsize=10,
    )

    ax = fig.add_subplot(grid[0, 1])
    shown = ndimage.gaussian_filter(wide.mean_image, 1.5)
    shown -= np.median(shown)
    image = ax.imshow(
        shown, cmap="viridis", vmin=-150, vmax=max(350, np.percentile(shown, 99.9))
    )
    for k, (y, x) in enumerate(wide.sites):
        ax.add_patch(Circle((x, y), APERTURE, fill=False, ec="r", lw=1))
        ax.text(x + 4.5, y - 4.5, str(k), color="r", fontsize=9)
    fig.colorbar(image, ax=ax, label="counts above median")
    frames_label = "image − background" if wide.background_subtracted else "image"
    ax.set_title(
        f"mean {frames_label} ({wide.shots} shots, smoothed 1.5 px)\n"
        "red: predicted sites",
        fontsize=10,
    )

    ax = fig.add_subplot(grid[0, 2])
    half = int(np.ceil(np.abs(wide.sites - centre).max())) + 8
    cy, cx = np.rint(centre).astype(int)
    region = wide.mean_image[cy - half : cy + half + 1, cx - half : cx + half + 1]
    image = ax.imshow(
        region - np.median(wide.mean_image),
        cmap="viridis",
        extent=(-half - 0.5, half + 0.5, half + 0.5, -half - 0.5),
    )
    for y, x in wide.sites - centre:
        ax.add_patch(Circle((x, y), APERTURE, fill=False, ec="r", lw=0.8))
    fig.colorbar(image, ax=ax, label="counts above median")
    ax.set_title("array region, unsmoothed\n(px from array centre)", fontsize=10)

    ax = fig.add_subplot(grid[1, :2])
    bins = np.linspace(-5, 10, 61)
    ax.hist(
        wide.empty_z,
        bins,
        density=True,
        alpha=0.35,
        color="grey",
        label="empty positions",
    )
    for k in range(wide.z.shape[1]):
        ax.hist(
            wide.z[:, k], bins, density=True, histtype="step", lw=1.6, label=f"site {k}"
        )
    ax.hist(
        at_centre,
        bins,
        density=True,
        histtype="step",
        lw=2,
        color="k",
        ls="--",
        label="array centre",
    )
    ax.axvline(3, ls=":", c="k", lw=0.8)
    ax.set_xlabel("aperture signal / noise (z)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title("per-shot score (4 px aperture, background ring 8–12 px)", fontsize=10)

    ax = fig.add_subplot(grid[1, 2])
    index = np.arange(wide.z.shape[1])
    width = 0.38
    for offset, r, label, colour in (
        (-width / 2, wide, "ring 8–12 px", "tab:blue"),
        (width / 2, tight, "ring 5–8 px", "tab:orange"),
    ):
        ax.bar(
            index + offset,
            r.z.mean(0),
            width,
            yerr=standard_error(r.z),
            capsize=3,
            color=colour,
            label=label,
        )
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xticks(index)
    ax.set_xticklabels([f"site {k}" for k in index])
    ax.set_ylabel("mean z per shot")
    ax.set_title(
        f"all sites combined: {wide.combined.mean():.2f} ± {standard_error(wide.combined):.2f} "
        f"(8–12), {tight.combined.mean():.2f} ± {standard_error(tight.combined):.2f} (5–8)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    out = f"{out_dir}/rid{rid}_sites.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
