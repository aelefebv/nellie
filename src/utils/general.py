import numpy as xp


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
            return 0, 0, 0, 0
        rmin, rmax = xp.where(r)[0][[0, -1]]
        cmin, cmax = xp.where(c)[0][[0, -1]]
        zmin, zmax = xp.where(z)[0][[0, -1]]
        return int(rmin), int(rmax), int(cmin), int(cmax), int(zmin), int(zmax)
    else:
        print("Image not 2D or 3D... Cannot get bounding box.")
