#%% Imports

import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed
from skimage.filters import gabor_kernel

#%% Get raw name

# stack_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
stack_name = 'C1-2022.07.05_Luminy_24hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_26hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_2_2.tif'

#%% Get path and open data

data_path = Path(Path.cwd(), 'data')
stack_path = Path(data_path, stack_name)
stack = io.imread(stack_path)
stack = stack[9,...]

#%% Parameters

n_filters = 16
frequency = 20 # (pixels)


#%%

# Get gabor kernels
kernels = []

for theta in np.arange(0, np.pi, np.pi/n_filters):

    kernel = gabor_kernel(
        frequency, 
        theta=theta, 
        ).real
    
    kernels.append(kernel)
            
viewer = napari.Viewer()
viewer.add_image(kernels[0])            


