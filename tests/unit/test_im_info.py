import numpy as np
import os
import tempfile
import unittest
import tifffile
from src.io.im_info import ImInfo


class TestImInfo(unittest.TestCase):

    def test_im_info(self):
        # Create a temporary file and write example TIFF data with metadata to it
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            data = np.zeros((3, 10, 10), dtype=np.uint16)
            tifffile.imwrite(tmp_file.name, data, imagej=True,
                             metadata={'axes': 'TYX', 'physicalsizex': 0.1, 'physicalsizey': 0.1,
                                       'spacing': 0.5, 'finterval': 0.1})

            # Create an ImInfo object for the temporary TIFF file
            im_info = ImInfo(tmp_file.name)

            # Check that the object attributes were set correctly
            self.assertEqual(im_info.im_path, tmp_file.name)
            self.assertIsNone(im_info.ch)
            self.assertDictEqual(im_info.dim_sizes, {'xy': 0.1, 'z': 0.5, 't': 0.1})

        # Delete the temporary file
        os.remove(tmp_file.name)


if __name__ == '__main__':
    unittest.main()
