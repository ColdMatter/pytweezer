"""Synthetic tweezer-array data for notebook exploration.

This module keeps the clean notebook hardware-free by generating synthetic
frames that can still be pushed through the existing analysis helpers in this
workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pytweezer.experiments.fake_tweezer_array import FakeTweezerArray
from pytweezer.analysis import analysis as pyt
from pytweezer.analysis.analysis import TweezerExperimentAnalysis


@dataclass(frozen=True)
class SyntheticDemoConfig:
    """Configuration for a synthetic tweezer demo dataset."""

    image_shape: tuple[int, int] = (256, 256)
    initial_grid_shape: tuple[int, int] = (16, 16)
    final_grid_shape: tuple[int, int] = (10, 10)
    grid_spacing_px: float = 14.0
    border_px: float = 24.0
    initial_filling_fraction: float = 0.62
    final_filling_fraction: float = 0.78
    background_filling_fraction: float = 0.03
    atom_peak: float = 250.0
    atom_fwhm_px: float = 5.0
    background_level: float = 20.0
    noise_sigma: float = 8.0
    poisson_noise: bool = False
    seed: int = 7
    sample_frames: int = 40
    animation_frames: int = 24
    feature_size: int = 10
    detection_step: int = 100
    window_size: int = 3
    binning: int = 60


def _render_gaussian_frame(
    centers: np.ndarray,
    image_shape: tuple[int, int],
    *,
    background_level: float,
    atom_peak: float,
    atom_fwhm_px: float,
    noise_sigma: float,
    rng: np.random.Generator,
    poisson_noise: bool,
) -> np.ndarray:
    height, width = image_shape
    sigma = max(atom_fwhm_px / 2.35482, 1e-6)
    image = np.full((height, width), background_level, dtype=np.float32)

    y_coords = np.arange(height, dtype=np.float32)
    x_coords = np.arange(width, dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    for cy, cx in centers:
        image += atom_peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))

    if noise_sigma > 0:
        image += rng.normal(loc=0.0, scale=noise_sigma, size=image.shape).astype(np.float32)

    image = np.clip(image, 0.0, None)

    if poisson_noise:
        image = rng.poisson(image).astype(np.float32)

    return image


def _jitter_occupancy(
    centers: np.ndarray,
    rng: np.random.Generator,
    filling_fraction: float,
) -> np.ndarray:
    occupied = rng.random(len(centers)) < filling_fraction
    return centers[occupied]


def build_demo_dataset(config: SyntheticDemoConfig = SyntheticDemoConfig()) -> dict[str, Any]:
    """Build a synthetic dataset and run the real analysis helpers on it."""

    rng = np.random.default_rng(config.seed)
    analysis = TweezerExperimentAnalysis(day="03", month="06Jun", year="2026")

    FakeTweezerArray._validate_grid_fits(
        config.image_shape[0],
        config.image_shape[1],
        config.initial_grid_shape[0],
        config.initial_grid_shape[1],
        config.grid_spacing_px,
        config.border_px,
    )
    FakeTweezerArray._validate_grid_fits(
        config.image_shape[0],
        config.image_shape[1],
        config.final_grid_shape[0],
        config.final_grid_shape[1],
        config.grid_spacing_px,
        config.border_px,
    )

    initial_centers = FakeTweezerArray._grid_centers(
        config.initial_grid_shape[0],
        config.initial_grid_shape[1],
        config.grid_spacing_px,
        config.border_px,
    )
    final_centers = FakeTweezerArray._grid_centers(
        config.final_grid_shape[0],
        config.final_grid_shape[1],
        config.grid_spacing_px,
        config.border_px,
    )

    signal_images = []
    background_images = []
    for _ in range(config.sample_frames):
        signal_centers = _jitter_occupancy(initial_centers, rng, config.initial_filling_fraction)
        background_centers = _jitter_occupancy(final_centers, rng, config.background_filling_fraction)

        signal_images.append(
            _render_gaussian_frame(
                signal_centers,
                config.image_shape,
                background_level=config.background_level,
                atom_peak=config.atom_peak,
                atom_fwhm_px=config.atom_fwhm_px,
                noise_sigma=config.noise_sigma,
                rng=rng,
                poisson_noise=config.poisson_noise,
            )
        )
        background_images.append(
            _render_gaussian_frame(
                background_centers,
                config.image_shape,
                background_level=config.background_level,
                atom_peak=config.atom_peak,
                atom_fwhm_px=config.atom_fwhm_px,
                noise_sigma=config.noise_sigma,
                rng=rng,
                poisson_noise=config.poisson_noise,
            )
        )

    filtered_signal = [pyt.morphological_tophat_high_pass(img, feature_size=config.feature_size) for img in signal_images]
    mean_signal = np.mean(filtered_signal, axis=0)
    grid_positions, detection_threshold = pyt.detect_trap_sites(
        mean_signal,
        list(config.initial_grid_shape),
        detection_step=config.detection_step,
    )
    pyt.visualize_results(
        mean_signal,
        grid_positions,
        margin=50,
        window_size=config.window_size,
        threshold=detection_threshold,
        vmaxfactor=0.5,
        bin_sharpness=20,
        bin_thresh_factor=0.7,
    )

    photon_rates, loading_probabilities, threshold, fidelity = analysis.get_array_loading_statistics(
        filtered_signal,
        grid_positions,
        list(config.initial_grid_shape),
        threshold_detection=True,
        window_size=config.window_size,
        binning=config.binning,
        show_histogram=False,
        verbose=False,
    )

    background_subtracted = analysis.tweezer_show_bg_subtracted(
        signal_images,
        background_images,
        reg=[0, -1, 0, -1],
        cmap="gray",
        show=False,
        vmaxfactor=0.8,
        show_grid=True,
    )

    initial_reference = _jitter_occupancy(initial_centers, rng, config.initial_filling_fraction)
    final_reference = _jitter_occupancy(final_centers, rng, config.final_filling_fraction)
    n_anim_atoms = min(len(initial_reference), len(final_reference))
    if n_anim_atoms == 0:
        n_anim_atoms = min(len(initial_centers), len(final_centers))
        initial_reference = initial_centers[:n_anim_atoms]
        final_reference = final_centers[:n_anim_atoms]
    else:
        initial_reference = initial_reference[:n_anim_atoms]
        final_reference = final_reference[:n_anim_atoms]

    frames = []
    for t in np.linspace(0.0, 1.0, config.animation_frames):
        interpolated_centers = (1.0 - t) * initial_reference + t * final_reference
        frames.append(
            _render_gaussian_frame(
                interpolated_centers,
                config.image_shape,
                background_level=config.background_level,
                atom_peak=config.atom_peak,
                atom_fwhm_px=config.atom_fwhm_px,
                noise_sigma=config.noise_sigma,
                rng=rng,
                poisson_noise=config.poisson_noise,
            )
        )

    return {
        "config": config,
        "analysis": analysis,
        "signal_images": np.array(signal_images),
        "background_images": np.array(background_images),
        "filtered_signal": np.array(filtered_signal),
        "mean_signal": mean_signal,
        "grid_positions": grid_positions,
        "detection_threshold": detection_threshold,
        "photon_rates": photon_rates,
        "loading_probabilities": loading_probabilities,
        "threshold": threshold,
        "fidelity": fidelity,
        "background_subtracted": background_subtracted,
        "frames": frames,
        "initial_centers": initial_centers,
        "final_centers": final_centers,
    }


def save_demo_preview(dataset: dict[str, Any], output_dir: str | Path) -> Path:
    """Save a simple preview of the synthetic dataset."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    preview_path = output_path / "synthetic_tweezer_preview.png"
    frames = dataset["frames"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(dataset["signal_images"][0], cmap="magma")
    ax[0].set_title("Synthetic signal")
    ax[1].imshow(dataset["background_subtracted"], cmap="gray")
    ax[1].set_title("Background-subtracted mean")
    ax[2].imshow(frames[-1], cmap="magma")
    ax[2].set_title("Final animation frame")
    for axis in ax:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)
    return preview_path


def main() -> None:
    dataset = build_demo_dataset()
    preview = save_demo_preview(dataset, Path("Data") / "synthetic_demo")
    np.savez_compressed(
        Path("Data") / "synthetic_demo" / "synthetic_tweezer_demo.npz",
        signal_images=dataset["signal_images"],
        background_images=dataset["background_images"],
        mean_signal=dataset["mean_signal"],
        background_subtracted=dataset["background_subtracted"],
        threshold=dataset["threshold"],
        fidelity=dataset["fidelity"],
    )
    print(f"Saved preview to {preview}")
    print(f"Detection threshold: {dataset['threshold']:.3f}")
    print(f"Detection fidelity: {dataset['fidelity']:.3f}")


if __name__ == "__main__":
    main()
