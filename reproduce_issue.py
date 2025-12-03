
import unittest
from unittest.mock import MagicMock, patch
from nellie.segmentation.mocap_marking import Markers

class TestMocapMarkingSkip(unittest.TestCase):
    def test_skip_no_t(self):
        # Mock ImInfo
        mock_im_info = MagicMock()
        mock_im_info.no_t = True
        mock_im_info.shape = (1, 100, 100) # Z, Y, X
        mock_im_info.axes = 'ZYX'
        mock_im_info.dim_res = {'Z': 1, 'Y': 1, 'X': 1}
        mock_im_info.no_z = False

        # Instantiate Markers
        markers = Markers(mock_im_info)

        # Mock _run_mocap_marking to verify it's NOT called
        with patch.object(markers, '_run_mocap_marking') as mock_run_mocap:
            with patch.object(markers, '_allocate_memory') as mock_allocate: # Also mock allocate to avoid side effects
                 with patch.object(markers, '_set_default_sigmas') as mock_sigmas:
                    markers.run()
                    
                    # Assert that _run_mocap_marking was NOT called
                    mock_run_mocap.assert_not_called()
                    print("Success: _run_mocap_marking was not called.")

if __name__ == '__main__':
    unittest.main()
