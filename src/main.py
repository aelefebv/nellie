from src.io.im_info import ImInfo
from src.pipeline.frangi_filter import FrangiFilter
from src.pipeline.segmentation import Segment


def run(input_path: str):
    im_info = ImInfo(input_path)
    # todo: idea, go backwards from last pipeline step, see if path is populated with a valid file.
    #  If invalid, keep going backwards until a valid one is found, then start pipeline from there.
    frangi = FrangiFilter(im_info)
    frangi.run_filter()
    segmentation = Segment(im_info)
    segmentation.semantic()

