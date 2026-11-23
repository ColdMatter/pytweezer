"""Base drawings: widgets that show the data they are handed.

A view knows nothing of streams, Properties, devices, the archive or threads;
the panel or applet around it feeds it. A specialisation of a base drawing
lives with the one panel that uses it, and moves here once a second panel
needs it.
"""
