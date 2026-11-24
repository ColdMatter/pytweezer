"""User optimisers: each module holds one :class:`~pytweezer.experiment.optimisers.Optimiser` and any variants of it.

The Experiment Manager lists every module under this package in its optimisers
catalogue, for the Optimisers tab. Modules are imported to read their argument
declarations, so nothing at module level may touch hardware.
"""
