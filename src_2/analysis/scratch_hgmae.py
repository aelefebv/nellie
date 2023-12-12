# import skeleton im, label branch im, label organelle im
# impo

import numpy as np
np_path = '/Users/austin/Downloads/test_20.npy'
# load npy file
np_data = np.load(np_path)

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")