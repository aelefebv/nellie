import numpy as xp  # could use cupy, but I think overhead typically takes more time...

def get_reshaped_image(im, num_t = None, im_info = None):
    im_to_return = im
    if 'T' not in im_info.axes or (len(im_info.axes) > 3 and len(im_to_return.shape) == 3):
        im_to_return = im_to_return[None, ...]
    elif num_t is not None:
        num_t = min(num_t, im_to_return.shape[0])
        im_to_return = im_to_return[:num_t, ...]
    return im_to_return

def bbox(im):
    """
    Computes the bounding box coordinates for a 2D or 3D image.

    Args:
    im (numpy.ndarray): The input image, as a NumPy array.

    Returns:
    A tuple containing the bounding box coordinates, as integers. For a 2D image, the tuple contains
    four elements: rmin, rmax, cmin, and cmax, where rmin and rmax are the minimum and maximum row
    indices that contain non-zero pixels, and cmin and cmax are the minimum and maximum column indices.
    For a 3D image, the tuple contains six elements: rmin, rmax, cmin, cmax, zmin, and zmax, where
    zmin and zmax are the minimum and maximum depth indices.
    If the image is not 2D or 3D, the function returns a bounding box with zero coordinates.
    """
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
