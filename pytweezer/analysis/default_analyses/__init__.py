"""The analyses the Results tab offers for a measurement, one module each.

Each module holds an analysis's computation, its Results-tab view and its
``ANALYSES`` entries, ``{name: AnalysisEntry}``; an experiment offers an
analysis by naming it in ``Experiment.analyses``. To add one, write its module
here and merge its entries into :data:`ANALYSES`.
"""

from pytweezer.analysis.default_analyses import cloud, tweezer_histogram

ANALYSES = tweezer_histogram.ANALYSES | cloud.ANALYSES
