from src.io.im_info import ImInfo


def run(im_info: ImInfo):
    im_memmap = im_info.get_im_memmap(im_info.im_path)

