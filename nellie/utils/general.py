from nellie import logger, xp


def get_reshaped_image(im, num_t=None, im_info=None):
    logger.debug('Reshaping image.')
    im_to_return = im
    if im_info.no_z:
        ndim = 2
    else:
        ndim = 3
    if 'T' not in im_info.axes or (len(im_info.axes) > ndim and len(im_to_return.shape) == ndim):
        im_to_return = im_to_return[None, ...]
        logger.debug(f'Adding time dimension to image, shape is now {im_to_return.shape}.')
    elif num_t is not None:
        num_t = min(num_t, im_to_return.shape[0])
        im_to_return = im_to_return[:num_t, ...]
        logger.debug(f'{num_t} timepoints found, shape is now {im_to_return.shape}.')
    return im_to_return


def bbox(im):
    if len(im.shape) == 2:
        rows = xp.any(im, axis=1)
        cols = xp.any(im, axis=0)
        if (not rows.any()) or (not cols.any()):
            return 0, 0, 0, 0
        rmin, rmax = xp.where(rows)[0][[0, -1]]
        cmin, cmax = xp.where(cols)[0][[0, -1]]
        return int(rmin), int(rmax), int(cmin), int(cmax)

    elif len(im.shape) == 3:
        r = xp.any(im, axis=(1, 2))
        c = xp.any(im, axis=(0, 2))
        z = xp.any(im, axis=(0, 1))
        if (not r.any()) or (not c.any()) or (not z.any()):
            return 0, 0, 0, 0, 0, 0
        rmin, rmax = xp.where(r)[0][[0, -1]]
        cmin, cmax = xp.where(c)[0][[0, -1]]
        zmin, zmax = xp.where(z)[0][[0, -1]]
        return int(rmin), int(rmax), int(cmin), int(cmax), int(zmin), int(zmax)

    else:
        print("Image not 2D or 3D... Cannot get bounding box.")
        return None
