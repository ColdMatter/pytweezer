"""Image-viewer applet: subscribes to image streams and displays them live.

Everything it does comes from :class:`pytweezer.GUI.image_applet.ImageApplet`,
whose docstring lists its Properties, how frames are placed and how to copy an
ROI as a camera ROI.
"""

from pytweezer.GUI.applet import run_applet
from pytweezer.GUI.image_applet import ImageApplet


class ImageDisplay(ImageApplet):
    """Self-updating image display applet.

    Subscribes to the image streams named in the ``imagestreams`` property and
    to optional mask streams (``maskstreams``), drawing them live with a
    colour map, histogram LUT, and configurable ROIs.
    """


def main(name):
    run_applet(ImageDisplay, default_name=name)


if __name__ == "__main__":
    run_applet(ImageDisplay, default_name="Image Monitor")
