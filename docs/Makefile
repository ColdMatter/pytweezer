# pytweezer documentation build.
#
# Usage (from this directory, inside the Poetry env):
#     poetry run make html      # build, output in _build/html/index.html
#     poetry run make clean     # remove _build/ and the generated api/ stubs
#
# The API reference under api/ is regenerated from the package docstrings on
# every build by sphinx-apidoc and is intentionally not checked in.

SPHINXBUILD  ?= sphinx-build
SPHINXAPIDOC ?= sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = _build
PKG          = ../pytweezer

# Excluded from the API reference:
#  - GUI/, cpp/, firmware, and the heavy standalone modules phasemask.py/arduino.py;
#  - servers/model_sync.py, which reads a CONFIG key that is not always present;
#  - the two Thorlabs camera wrappers, which do module-level work that the mocked
#    pylablib cannot satisfy (drop them from this list once that is guarded);
#  - the vendor 'Blink Plus' SLM SDK tree.
# Patterns are matched against the file paths sphinx-apidoc walks.
APIDOC_EXCLUDE = $(PKG)/GUI $(PKG)/cpp $(PKG)/arduino_pulser_firmware \
                 $(PKG)/phasemask.py $(PKG)/arduino.py \
                 $(PKG)/servers/model_sync.py $(PKG)/drivers/Blink* \
                 $(PKG)/drivers/thorcam.py $(PKG)/drivers/tweezermonitorcam.py

.PHONY: html apidoc clean

apidoc:
	$(SPHINXAPIDOC) -f -o api $(PKG) $(APIDOC_EXCLUDE) \
		--separate --module-first --no-toc --implicit-namespaces

html: apidoc
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo "\nBuilt: $(BUILDDIR)/html/index.html"

clean:
	rm -rf "$(BUILDDIR)" api
