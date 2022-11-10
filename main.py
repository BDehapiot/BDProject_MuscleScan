#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path

from functions import ranged_uint8

#%% Get raw name

raw_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'

#%% Get path and open data

raw_path = Path(Path.cwd(), 'data', raw_name)
raw = io.imread(raw_path)
test = raw[9,...]
# test = ranged_uint8(raw[9,...], 0.1, 99.9)

#%%

from skimage.filters import gabor

# for i, phi in enumerate
# test_filt = gabor(test, 0.1, 0)
test_filt = gabor(test, 0.2, theta=np.pi/2, bandwidth=1, sigma_x=None, sigma_y=None, n_stds=3)


#%%
viewer = napari.Viewer()
viewer.add_image(test)
viewer.add_image(test_filt[1])
# viewer.grid.enabled = True