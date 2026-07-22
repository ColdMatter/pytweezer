import sys
from contextlib import contextmanager

import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext

if sys.platform == "win32":
    # Windows (MSVC)
    # /GL = Link-Time Optimization (Compile)
    # /LTCG = Link-Time Optimization (Link)
    tophat_compile_args = ["/O2", "/openmp:llvm", "/GL"]
    tophat_link_args = ["/LTCG"]
    
    sum_compile_args = ["/O2", "/GL"]
    sum_link_args = ["/LTCG"]
else:
    # Linux/macOS (GCC/Clang)
    # -flto = Link-Time Optimization
    tophat_compile_args = ["-O3", "-fopenmp", "-flto"]
    tophat_link_args = ["-fopenmp", "-flto"]
    
    sum_compile_args = ["-O3", "-flto"]
    sum_link_args = ["-flto"]

# extensions
ext_modules = [
    # Morphological Top Hat
    Pybind11Extension(
        "pytweezer.cpp.morph_tophat_cpp",
        ["cpp_implementations/Morph_top_hat/Morphological_top_hat_cpp.cpp"],
        cxx_std=17, 
        extra_compile_args=tophat_compile_args,
        extra_link_args=tophat_link_args,
    ),
    
    #  Sum Pixel Values
    Pybind11Extension(
        "pytweezer.cpp.sum_pixel_values_cpp",
        ["cpp_implementations/Pixel_Summing/sum_pixel_values.cpp"],
        cxx_std=14, 
        include_dirs=[np.get_include()],
        extra_compile_args=sum_compile_args,
        extra_link_args=sum_link_args,
    )
]

class OptionalBuildExt(build_ext):
    """``build_ext`` that treats every extension as optional.

    The compiled modules are an accelerator, not a requirement — a machine that
    only runs the client GUI or a device server never imports them. Building
    them needs a C++ toolchain and the Python development headers
    (``python3.x-dev`` on Debian/Ubuntu, MSVC Build Tools on Windows), which
    not every lab PC has. Without this, a missing header aborts the whole
    ``poetry install`` with nothing but "Failed to install <path>".

    A skipped extension stays unimportable, so ``from pytweezer.cpp import
    morph_tophat_cpp`` raises ``ImportError``; callers that need the speed-up
    must handle that. Extensions are independent: one failing to compile does
    not stop the others from being built and installed.
    """

    def initialize_options(self):
        super().initialize_options()
        self._failed = []

    def run(self):
        # Catches failures before any single extension is reached, e.g. no
        # compiler on PATH at all.
        try:
            super().run()
        except Exception as exc:
            _warn_skipped("all C++ extensions", exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            self._failed.append(ext.name)
            _warn_skipped(ext.name, exc)

    def copy_extensions_to_source(self):
        # Copying is all-or-nothing per invocation: it raises on the first
        # missing .so, which would discard the extensions that *did* build.
        with self._only_built():
            super().copy_extensions_to_source()

    def get_outputs(self):
        with self._only_built():
            return super().get_outputs()

    @contextmanager
    def _only_built(self):
        """Hide extensions that failed to compile from the base class."""
        original = self.extensions
        self.extensions = [e for e in original if e.name not in self._failed]
        try:
            yield
        finally:
            self.extensions = original


def _warn_skipped(what, exc):
    print(
        f"\n*** WARNING: skipping {what}: {exc.__class__.__name__}: {exc}\n"
        "*** pytweezer will install without the accelerated code paths.\n"
        "*** Install a C++ compiler and the Python development headers "
        "(e.g. `apt install python3.11-dev`), then re-run `poetry install`.\n",
        file=sys.stderr,
        flush=True,
    )


#  for Poetry
def build(setup_kwargs):
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": OptionalBuildExt},
        "zip_safe": False,
    })