from src.io.im_info import ImInfo
from src.pipeline import frangi_filter


def run(input_path: str):
    im_info = ImInfo(input_path)
    frangi_filter.run(im_info)
