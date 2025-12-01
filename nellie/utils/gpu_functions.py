from nellie import xp, device_type


def otsu_threshold(matrix, nbins=256):
    """
    GPU/CPU-agnostic implementation of Otsu's threshold.
    Operates on an n-d array using the current xp backend.
    """
    # Flatten and build histogram
    flat = matrix.reshape(-1)
    counts, bin_edges = xp.histogram(
        flat,
        bins=nbins,
        range=(flat.min(), flat.max())
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    counts = counts / xp.sum(counts)

    weight1 = xp.cumsum(counts)
    mean1 = xp.cumsum(counts * bin_centers) / weight1

    weight2 = xp.cumsum(counts[::-1])[::-1]
    mean2 = (xp.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = xp.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold, variance12[idx]


def triangle_threshold(matrix, nbins=256):
    """
    GPU/CPU-agnostic implementation of triangle threshold.
    Operates on an n-d array using the current xp backend.
    """
    flat = matrix.reshape(-1)
    hist, bin_edges = xp.histogram(
        flat,
        bins=nbins,
        range=(xp.min(flat), xp.max(flat))
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist = hist / xp.sum(hist)

    arg_peak_height = xp.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = xp.flatnonzero(hist)[[0, -1]]

    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = xp.flip(hist, axis=0)

        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1
    del arg_high_level

    width = arg_peak_height - arg_low_level
    x1 = xp.arange(width)
    y1 = hist[x1 + arg_low_level]

    norm = xp.sqrt(peak_height ** 2 + width ** 2)
    peak_height = peak_height / norm
    width = width / norm

    length = peak_height * x1 - width * y1
    arg_level = xp.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return bin_centers[arg_level]