#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path

#%% Get raw name

raw_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'

#%% Get path and open data

raw_path = Path(Path.cwd(), 'data', raw_name)
raw = io.imread(raw_path)

#%%



#%%

viewer = napari.Viewer()
viewer.add_image(raw)