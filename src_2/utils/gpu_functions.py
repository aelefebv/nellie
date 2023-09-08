from src import xp


def otsu_threshold(matrix, nbins=256):
    # gpu version of skimage.filters.threshold_otsu
    counts, bin_edges = xp.histogram(matrix.reshape(-1), bins=nbins, range=(matrix.min(), matrix.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    counts = counts / xp.sum(counts)

    weight1 = xp.cumsum(counts)
    weight2 = xp.cumsum(counts[::-1])[::-1]
    mean1 = xp.cumsum(counts * bin_centers) / weight1
    mean2 = (xp.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = xp.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold

def triangle_threshold(matrix, nbins=256):
    # gpu version of skimage.filters.threshold_triangle
    hist, bin_edges = xp.histogram(matrix.reshape(-1), bins=nbins, range=(matrix.min(), matrix.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    hist = hist / xp.sum(hist)

    arg_peak_height = xp.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = xp.flatnonzero(hist)[[0, -1]]

    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = hist[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1
    del(arg_high_level)

    width = arg_peak_height - arg_low_level
    x1 = xp.arange(width)
    y1 = hist[x1 + arg_low_level]

    norm = xp.sqrt(peak_height**2 + width**2)
    peak_height = peak_height / norm
    width = width / norm

    length = peak_height * x1 - width * y1
    arg_level = xp.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return bin_centers[arg_level]
