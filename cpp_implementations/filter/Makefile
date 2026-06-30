# Build the compiled white-tophat filter as a pybind11 Python module.
#
#   make            # build fast_tophat_filter*.so
#   make clean
#
# Uses the project venv's Python by default; override with `make PY=...`.

PY      ?= ../../venv/bin/python
CXX     ?= c++
CXXFLAGS = -O3 -std=c++17 -Wall -funroll-loops -DNDEBUG

# Discover Python / pybind11 include paths and the extension suffix.
PYINC    := $(shell $(PY) -c "import sysconfig; print('-I'+sysconfig.get_path('include'))")
PYBINC   := $(shell $(PY) -c "import pybind11; print('-I'+pybind11.get_include())")
EXT      := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  MODFLAGS = -shared -undefined dynamic_lookup
else
  MODFLAGS = -shared -fPIC
endif

MODULE = fast_tophat_filter$(EXT)

.PHONY: all clean
all: $(MODULE)

$(MODULE): bindings.cpp tophat_filter.hpp
	$(CXX) $(CXXFLAGS) -fPIC $(MODFLAGS) $(PYINC) $(PYBINC) bindings.cpp -o $(MODULE)

clean:
	rm -f fast_tophat_filter*.so
