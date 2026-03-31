import time

import numpy as np

from pytweezer.experiment.experiment import Experiment, ImageResultChannel
from pytweezer.experiment.experiment import BoolValue
from pytweezer.experiment.experiment import NumberValue
from pytweezer.experiment.dummy_drivers import DummyCamera


class FakeTweezerArray(Experiment):
    """Generate synthetic tweezer-array images with adjustable filling and noise."""

    motmaster_script = "dummy_script"

    def build(self):
        super().build()
        self.setattr_device("dummy_camera", mode="test", n_frames=1, servers=True)
        self.dummy_camera: DummyCamera
        
        self.setattr_result("image_channel", ImageResultChannel)

        self.setattr_argument("n_frames", NumberValue, ndecimals=0, step=1, value=10, minval=1, maxval=1000)
        self.setattr_argument("image_height", NumberValue, ndecimals=0, step=1, value=256, minval=32, maxval=4096)
        self.setattr_argument("image_width", NumberValue, ndecimals=0, step=1, value=256, minval=32, maxval=4096)
        self.setattr_argument("grid_rows", NumberValue, ndecimals=0, step=1, value=10, minval=1, maxval=256)
        self.setattr_argument("grid_cols", NumberValue, ndecimals=0, step=1, value=10, minval=1, maxval=256)
        self.setattr_argument("filling_fraction", NumberValue, ndecimals=3, step=0.01, value=0.5, minval=0.0, maxval=1.0)
        self.setattr_argument("grid_spacing_px", NumberValue, ndecimals=2, step=0.5, value=16.0, minval=1.0, maxval=256.0)
        self.setattr_argument("border_px", NumberValue, ndecimals=0, step=1, value=24, minval=0, maxval=1024)
        self.setattr_argument("atom_peak", NumberValue, ndecimals=2, step=10.0, value=250.0, minval=0.0, maxval=1e6)
        self.setattr_argument("atom_fwhm_px", NumberValue, ndecimals=2, step=0.2, value=5.0, minval=0.5, maxval=128.0)
        self.setattr_argument("background_level", NumberValue, ndecimals=2, step=1.0, value=20.0, minval=0.0, maxval=1e6)
        self.setattr_argument("noise_sigma", NumberValue, ndecimals=2, step=0.5, value=8.0, minval=0.0, maxval=1e6)
        self.setattr_argument("poisson_noise", BoolValue, value=False)
        self.setattr_argument("seed", NumberValue, ndecimals=0, step=1, value=-1, minval=-1, maxval=2**31 - 1)
        self.setattr_argument("sleep_time", NumberValue, ndecimals=3, step=0.01, value=0.05, minval=0.0, maxval=10.0)

    def pre_run(self):
        super().pre_run()
        self.dummy_camera.start_acquisition()

    @staticmethod
    def _validate_grid_fits(height, width, rows, cols, spacing, border):
        needed_h = 2 * border + (rows - 1) * spacing
        needed_w = 2 * border + (cols - 1) * spacing
        if needed_h > height - 1 or needed_w > width - 1:
            raise ValueError(
                "Grid does not fit image. Increase image size or reduce rows/cols/spacing/border. "
                f"Needed (h, w)=({needed_h:.1f}, {needed_w:.1f}), image=(h={height}, w={width})."
            )

    @staticmethod
    def _grid_centers(rows, cols, spacing, border):
        ys = border + np.arange(rows) * spacing
        xs = border + np.arange(cols) * spacing
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        return np.column_stack((yy.ravel(), xx.ravel()))

    def _make_frame(self, rng, height, width, centers, filling_fraction, peak, sigma, background_level, noise_sigma, add_poisson):
        image = np.full((height, width), background_level, dtype=np.float32)

        n_sites = centers.shape[0]
        occupied = rng.random(n_sites) < filling_fraction
        occupied_centers = centers[occupied]

        if occupied_centers.size > 0:
            y_coords = np.arange(height, dtype=np.float32)
            x_coords = np.arange(width, dtype=np.float32)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

            # Add equal-amplitude Gaussian atom spots at each occupied tweezer site.
            for cy, cx in occupied_centers:
                image += peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))

        if noise_sigma > 0:
            image += rng.normal(loc=0.0, scale=noise_sigma, size=image.shape).astype(np.float32)

        image = np.clip(image, 0.0, None)

        if add_poisson:
            image = rng.poisson(image).astype(np.float32)

        return image

    def run(self):
        height = int(self.image_height.get())
        width = int(self.image_width.get())
        rows = int(self.grid_rows.get())
        cols = int(self.grid_cols.get())
        n_frames = int(self.n_frames.get())

        filling_fraction = float(self.filling_fraction.get())
        spacing = float(self.grid_spacing_px.get())
        border = float(self.border_px.get())
        peak = float(self.atom_peak.get())
        fwhm = float(self.atom_fwhm_px.get())
        sigma = max(fwhm / 2.35482, 1e-6)
        background_level = float(self.background_level.get())
        noise_sigma = float(self.noise_sigma.get())
        add_poisson = bool(self.poisson_noise.get())

        self._validate_grid_fits(height, width, rows, cols, spacing, border)

        centers = self._grid_centers(rows, cols, spacing, border)

        seed_value = int(self.seed.get())
        rng = np.random.default_rng(None if seed_value <= 0 else seed_value)

        images = [
            self._make_frame(
                rng=rng,
                height=height,
                width=width,
                centers=centers,
                filling_fraction=filling_fraction,
                peak=peak,
                sigma=sigma,
                background_level=background_level,
                noise_sigma=noise_sigma,
                add_poisson=add_poisson,
            )
            for _ in range(n_frames)
        ]
        
        self.image_channel.set(np.array(images))

        self.dummy_camera.dummy_images = images
        time.sleep(self.sleep_time.get())

    def post_run(self):
        exp_info = {"task": self._task, "run": self._run, "rep": self._rep}
        print("Broadcasting images with exp_info:", exp_info)
        self.dummy_camera.acquire_n_frames(exp_info=exp_info, autosave=True)
        time.sleep(0.1)
