from src import logger


def get_reshaped_image(im, num_t = None, im_info = None):
    logger.debug('Reshaping image.')
    im_to_return = im
    if 'T' not in im_info.axes or (len(im_info.axes) > 3 and len(im_to_return.shape) == 3):
        im_to_return = im_to_return[None, ...]
        logger.debug(f'Adding time dimension to image, shape is now {im_to_return.shape}.')
    elif num_t is not None:
        num_t = min(num_t, im_to_return.shape[0])
        im_to_return = im_to_return[:num_t, ...]
        logger.debug(f'{num_t} timepoints found, shape is now {im_to_return.shape}.')
    return im_to_return
