#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
from aicsimageio.readers import CziReader
from joblib import Parallel, delayed
from functions import get_mask, process_gabor

from BDTools.dtype import ranged_conversion
from BDTools.nan import nanfilt, nanreplace

#%% Hyperstack name

# hstack_name = '2022.10.31_22hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
hstack_name = '2022.10.31_24hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_26hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_28hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_30hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_32hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_48hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'

#%% Parameters

ridge_channel = 1
thresh_coeff = 0.75

#%% Open czi file

hstack_path = Path(Path(Path.cwd(), 'data'), hstack_name)
hstack_czi = CziReader(hstack_path)
hstack = hstack_czi.data.squeeze()

# Get voxel size (µm)
vox_size = (
    hstack_czi.physical_pixel_sizes.Y,
    hstack_czi.physical_pixel_sizes.X,
    hstack_czi.physical_pixel_sizes.Z
    )

#%% 

zstack = hstack[ridge_channel,...]
img = zstack[zstack.shape[0]//2,...]

#%%

from skimage.morphology import disk
from skimage.transform import rescale
from skimage.morphology import binary_dilation
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.filters import threshold_li, scharr_h, scharr_v, gaussian, rank

rsize_factor = 0.5
min_magnitude = 0.05
smooth_size = 32

start = time.time()
print('get_orientations')

# Initialize
smooth_size = int(smooth_size*rsize_factor)

# Resize & process img
rsize = rescale(img, rsize_factor, preserve_range=True)
rsize = ranged_conversion(
    rsize, intensity_range=(5,95), spread=3, dtype='float')

# Get mask
mask = rsize > threshold_li(rsize, tolerance=1)*thresh_coeff
mask = remove_small_holes(mask, area_threshold=rsize.shape[0])
mask = remove_small_objects(mask, min_size=rsize.shape[0])

# Extract local gradient
gradient_h = scharr_h(rsize); gradient_v = scharr_v(rsize)
magnitudes = np.sqrt((gradient_h ** 2) + (gradient_v ** 2))
orientations = np.arctan2(gradient_h, gradient_v) * (180/np.pi) % 180

# Remove low magnitude orientations
orientations[magnitudes < min_magnitude] = np.nan
orientations[mask == False] = np.nan
orientations = nanreplace(
    orientations, kernel_size=smooth_size, method='median', mask=mask)

# Smooth orientations map   
orientations = rank.median(
    orientations.astype('uint8'), 
    footprint=disk(smooth_size),
    mask=mask,
    ).astype('float')
orientations[mask == False] = np.nan

# # Rescale orientation and remove out of mask signal
# orientations = rescale(orientations, 1/rsize_factor, preserve_range=True)

end = time.time()
print(f'  {(end-start):5.3f} s')  

#%%

viewer = napari.Viewer()
# viewer.add_image(img)
viewer.add_image(rsize)
# viewer.add_image(mask)
viewer.add_image(magnitudes)
viewer.add_image(orientations)
# viewer.add_image(orientations_sd)

#%%

# # Get mask
# thresh = threshold_li(rsize, tolerance=1)
# mask = rsize > thresh*thresh_coeff
# mask = remove_small_holes(mask, area_threshold=rsize.shape[0])
# mask = remove_small_objects(mask, min_size=rsize.shape[0])
