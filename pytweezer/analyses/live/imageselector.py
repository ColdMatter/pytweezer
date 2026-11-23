"""Select one image per shot from an image stream, by its ``index`` in the shot.

Input:
    One image stream

Output:
    Image stream, the selected image

Properties:
    *   imagestreams: ([str]) input image streams
    *   SelectedImage: (int) the number of the image to select, counting from 0.
"""

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.servers import PropertyAttribute


class ImageSelector(ImageAnalysis):
    _index = PropertyAttribute("SelectedImage", 1)

    def process(self, head, data):
        if head.get("index") == self._index:
            return head, data
        return None


if __name__ == "__main__":
    run_analysis(ImageSelector)
