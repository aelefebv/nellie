from src import xp, ndi, logger, is_gpu, measure
from src.io.im_info import ImInfo
import tifffile


def clean_labels(frame):
    """
    Cleans up and labels branches in a 3D image stack. Assumes branch points are labeled as 3,
    edges are labeled as 2, and tips are labeled as 1.

    Parameters:
    -----------
    frame : np.ndarray
        The input 3D image stack. Assumes branch points are labeled as 3, edges as 2, and tips as 1.

    Returns:
    --------
    np.ndarray
        A labeled image where each branch is labeled with a unique integer.
    """
    # Find edge points in image
    edge_points = frame == 2
    # Find edge points in image
    edge_labels, _ = ndi.label(edge_points, structure=xp.ones((3, 3, 3)))
    # Label individual branch points
    bp_labels, _ = ndi.label(frame == 3, structure=xp.ones((3, 3, 3)))
    bp_regions = measure.regionprops(bp_labels)

    # Loop over each branch point region
    for branch_num, bp_region in enumerate(bp_regions):
        logger.debug(f'Cleaning branch point {branch_num}/{len(bp_regions) - 1}')

        # Get the coords of the neighboring pixels
        coords = []
        for coord in bp_region.coords:
            z, y, x = coord
            coords.extend([(z + i, y + j, x + k)
                           for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]
                           if (i != 0 or j != 0 or k != 0)])

        # Get tip neighbors
        tip_neigh = xp.asarray([idx for idx in coords if frame[idx] == 1])

        # Order matters:

        # if tip is a neighbor, tip disappears
        for neigh in tip_neigh:
            frame[tuple(neigh)] = 0
        # get branch point's edge neighbors
        edge_neigh = xp.asarray([idx for idx in coords if frame[idx] == 2])
        # if only 1 neighbor, branch point becomes a tip
        if len(edge_neigh) == 1:
            frame[bp_labels == bp_region.label] = 1
        # if 2 neighbors only, label bp becomes label 1, label 2 becomes label 1
        if len(edge_neigh) == 2:
            frame[bp_labels == bp_region.label] = 2
            edge_labels[edge_labels == edge_labels[tuple(edge_neigh[-1])]] = edge_labels[tuple(edge_neigh[0])]
            edge_labels[bp_labels == bp_region.label] = edge_labels[tuple(edge_neigh[0])]

    return edge_labels


class BranchSegments:
    def __init__(self, im_info: ImInfo):
        """
        Initialize a BranchSegments object with an ImInfo object.

        Args:
            im_info: An ImInfo object containing metadata for the input image.
        """
        self.im_info = im_info
        self.segment_memmap = None
        self.shape = ()

    def segment_branches(self, num_t, dtype='uint32'):
        """
        Segment individual branches in the image.

        Args:
            num_t: The number of frames to process. If None, all frames will be processed.
            dtype: The data type to use for the branch segments image. Defaults to 'uint32'.
        """
        # Load the neighbor image file as memory-mapped files
        neighbor_im = tifffile.memmap(self.im_info.path_im_neighbors, mode='r+')

        # Load only a subset of frames if num_t is not None
        if num_t is not None:
            num_t = min(num_t, neighbor_im.shape[0])
            neighbor_im = neighbor_im[:num_t, ...]
        self.shape = neighbor_im.shape

        # Allocate memory for the branch segment volume and load it as a memory-mapped file
        self.im_info.allocate_memory(
            self.im_info.path_im_label_seg, shape=self.shape, dtype=dtype, description='Branch segments image'
        )
        self.segment_memmap = tifffile.memmap(self.im_info.path_im_label_seg, mode='r+')

        # Label any individual branch and save it to the memmap.
        for frame_num, frame in enumerate(neighbor_im):
            logger.info(f'Running branch point analysis, volume {frame_num}/{len(neighbor_im)-1}')
            frame_neighbor = xp.asarray(frame)

            edge_labels = clean_labels(frame_neighbor)

            if is_gpu:
                self.segment_memmap[frame_num] = edge_labels.get()
            else:
                self.segment_memmap[frame_num] = edge_labels


# def angular_reconnection(bp_region, frame_neighbor, edge_labels, edge_regions):
#     # could also be connection intensity or seg radius based?
#
#     # initialize variables to keep track of the smallest angle and corresponding branch labels
#     min_angle = xp.inf
#     connect_label_1 = None
#     connect_label_2 = None
#     relabel = False
#
#     # Get the coords of the neighboring pixels
#     coords = []
#     for coord in bp_region.coords:
#         z, y, x = coord
#         coords.extend([(z + i, y + j, x + k)
#                        for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]
#                        if (i != 0 or j != 0 or k != 0)])
#
#     # find the neighboring branch points connected to the current branch point
#     neigh_idx = xp.asarray([idx for idx in coords if frame_neighbor[idx] == 2])
#
#     # loop over each pair of neighboring branch points
#     for i_neigh in range(len(neigh_idx) - 1):
#         i_z, i_y, i_x = neigh_idx[i_neigh]
#         i_neigh_coords = [(i_z + i, i_y + j, i_x + k)
#                           for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]
#                           if (i != 0 or j != 0 or k != 0)]
#
#         i_neigh_idx = xp.asarray([idx for idx in i_neigh_coords if frame_neighbor[idx] == 2])
#         if len(i_neigh_idx) == 0:
#             continue
#         for j_neigh in range(i_neigh + 1, len(neigh_idx)):
#             # compute the angle between the two neighboring branch points
#             # # method 1, angle from branch->bp centroid-> branch
#             # segment_1 = neigh_idx[i] - bp_region.centroid
#             # segment_2 = neigh_idx[j] - bp_region.centroid
#             # # method 2, angle similarity in branch
#             j_z, j_y, j_x = neigh_idx[j_neigh]
#             j_neigh_coords = [(j_z + i, j_y + j, j_x + k)
#                               for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]
#                               if (i != 0 or j != 0 or k != 0)]
#
#             j_neigh_idx = xp.asarray([idx for idx in j_neigh_coords if frame_neighbor[idx] == 2])
#             if len(j_neigh_idx) == 0:
#                 continue
#             segment_1 = i_neigh_idx[0] - neigh_idx[i_neigh]
#             segment_2 = neigh_idx[j_neigh] - j_neigh_idx[0]
#
#             # angle calculation using dot product formula for the angle between 2 vectors
#             angle = xp.abs(xp.arccos(
#                 xp.dot(segment_1, segment_2) / (xp.linalg.norm(segment_1) * xp.linalg.norm(segment_2))
#             )) * 180 / xp.pi
#             # angle = xp.arccos(
#             #     xp.dot(segment_1, segment_2) / (xp.linalg.norm(segment_1) * xp.linalg.norm(segment_2))
#             # ) * 180 / xp.pi
#             # print(angle1, angle)
#
#             # if the angle between the two branches is smaller than the current smallest angle,
#             # update the smallest angle and the branch indices
#             angle = edge_regions[edge_labels[tuple(neigh_idx[i_neigh])]].intensity_mean - \
#                     edge_regions[edge_labels[tuple(neigh_idx[j_neigh])]].intensity_mean
#             if angle < min_angle:
#                 relabel = True
#                 min_angle = angle
#                 connect_label_1 = edge_labels[tuple(neigh_idx[i_neigh])]
#                 connect_label_2 = edge_labels[tuple(neigh_idx[j_neigh])]
#
#     # relabel label 2 to label 1
#     if relabel and connect_label_1 != 0 and connect_label_2 != 0:
#         edge_labels[edge_labels == connect_label_2] = connect_label_1
#         # edge_labels[z, y, x] = connect_label_1
#     # else:
#     #     edge_labels[z, y, x] = edge_labels[tuple(neigh_idx[0])]


if __name__ == "__main__":
    import os

    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    if not os.path.isfile(filepath):
        filepath = "/Users/austin/Documents/Transferred/deskewed-single.ome.tif"
    try:
        test = ImInfo(filepath, ch=0)
    except FileNotFoundError:
        logger.error("File not found.")
        exit(1)
    branch = BranchSegments(test)
    branch.segment_branches(2)
    print('hi')
