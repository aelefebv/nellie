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
            data = np.zeros((3, 3, 3, 3, 3), dtype=np.uint8)
            tifffile.imwrite(tmp_file.name, data, imagej=True,
                             metadata={'axes': 'TZCYX', 'physicalsizex': 0.1, 'physicalsizey': 0.2,
                                       'spacing': 0.5, 'finterval': 0.1})

            # Create an ImInfo object for the temporary TIFF file
            im_info = ImInfo(tmp_file.name)

            # Check that the object attributes were set correctly
            self.assertEqual(im_info.im_path, tmp_file.name)
            self.assertIsNone(im_info.ch)
            self.assertDictEqual(im_info.dim_sizes, {'x': 0.1, 'y': 0.2, 'z': 0.5, 't': 0.1})

            # Check that metadata was loaded correctly
            self.assertEqual(im_info.axes, 'TZCYX')
            self.assertEqual(im_info.shape, (3, 3, 3, 3, 3))
            self.assertEqual(im_info.extension, 'tif')
            self.assertEqual(im_info.filename, os.path.splitext(os.path.basename(tmp_file.name))[0])
            self.assertEqual(im_info.dirname, os.path.basename(os.path.dirname(tmp_file.name)))

            # Call create_output_dirs with a temporary output directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_dir = os.path.join(tmp_dir, "output")
                im_info.create_output_dirs(tmp_dir)

                # Check that the output directories were created correctly
                self.assertTrue(os.path.exists(output_dir))
                self.assertTrue(os.path.exists(os.path.join(output_dir, "images")))
                self.assertTrue(os.path.exists(os.path.join(output_dir, "pickles")))
                self.assertTrue(os.path.exists(os.path.join(output_dir, "csv")))

        # Delete the temporary file
        os.remove(tmp_file.name)


if __name__ == '__main__':
    unittest.main()
