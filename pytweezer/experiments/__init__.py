"""User experiments, one :class:`~pytweezer.experiment.Experiment` subclass per module.

The Experiment Manager lists every module under this package in its
experiments catalogue, for the Experiments tab; optimisers live in
:mod:`pytweezer.optimisers`. Modules are imported to read their argument
declarations, so nothing at module level may touch hardware.
"""
