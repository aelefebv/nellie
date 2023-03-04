from src import xp, ndi, logger, is_gpu
from src.io.im_info import ImInfo
import tifffile


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
        neighbor_im = tifffile.memmap(self.im_info.path_im_neighbors, mode='r')

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
            logger.info(f'Running branch point analysis, volume {frame_num}/{len(neighbor_im)}')
            frame_neighbor = xp.asarray(frame)

            # segment individual branches
            edge_points = frame_neighbor == 2
            branch_labels, _ = ndi.label(edge_points, structure=xp.ones((3, 3, 3)))

            # loop over each branch point
            branch_idx = xp.argwhere(frame_neighbor == 3)
            for branch_num, bp_idx in enumerate(branch_idx):
                logger.debug(f'Reconnecting branch {branch_num}/{len(branch_idx)}')

                # initialize variables to keep track of the smallest angle and corresponding branch labels
                min_angle = xp.inf
                connect_label_1 = None
                connect_label_2 = None

                z, y, x = bp_idx

                # Get the coords of the neighboring pixels
                coords = [(z+i, y+j, x+k)
                          for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]
                          if (i != 0 or j != 0 or k != 0)]

                # find the neighboring branch points connected to the current branch point
                neigh_idx = xp.asarray([idx for idx in coords if frame_neighbor[idx] != 0])

                # loop over each pair of neighboring branch points
                for i in range(len(neigh_idx) - 1):
                    for j in range(i + 1, len(neigh_idx)):
                        # compute the angle between the two neighboring branch points
                        segment_1 = neigh_idx[i] - bp_idx
                        segment_2 = neigh_idx[j] - bp_idx

                        # angle calculation using dot product formula for the angle between 2 vectors
                        angle = xp.abs(xp.arccos(
                            xp.dot(segment_1, segment_2) / (xp.linalg.norm(segment_1) * xp.linalg.norm(segment_2))))
                        # * 180 / xp.pi  # don't need this multiplier since doing comparisons

                        # if the angle between the two branches is smaller than the current smallest angle,
                        # update the smallest angle and the branch indices
                        if angle < min_angle:
                            min_angle = angle
                            connect_label_1 = branch_labels[tuple(neigh_idx[i])]
                            connect_label_2 = branch_labels[tuple(neigh_idx[j])]

                # relabel label 2 to label 1
                if connect_label_1 != 0 and connect_label_2 != 0:
                    branch_labels[branch_labels == connect_label_2] = connect_label_1

            if is_gpu:
                self.segment_memmap[frame_num] = branch_labels.get()
            else:
                self.segment_memmap[frame_num] = branch_labels

if __name__ == "__main__":
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    test = ImInfo(filepath, ch=0)
    branch = BranchSegments(test)
    branch.segment_branches(2)
    print('hi')
