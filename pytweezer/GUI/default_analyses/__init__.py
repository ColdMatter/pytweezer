"""The analyses the Results tab offers for a measurement, one module each.

Each module holds an analysis's Results-tab view and its ``ANALYSES`` entries,
``{name: AnalysisEntry}``; the calculation it draws is in
:mod:`pytweezer.analyses.default`. An experiment offers an analysis by naming
it in ``Experiment.analyses``. To add one, write its calculation there, its
view here and merge its entries into :data:`ANALYSES`.
"""

from pytweezer.GUI.default_analyses import cloud, tweezer_histogram

ANALYSES = tweezer_histogram.ANALYSES | cloud.ANALYSES
