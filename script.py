import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


#1. PARAMETERS

h5_path = r"//home/martinfranc/Stažené/Magnify/2D/MAGNIFY_2_Image_G001_ch01_fixed_Probabilities.h5"  #CHANGE - path to the Ilastik .h5 file with probabilities
output_csv = "Output.csv"

pixel_size_nm = 58.0  #CHANGE - Size of one pixel in nanometers
expansion_factor = 4.0  #CHANGE - Expansion factor
probability_threshold = 0.7  #CHANGE - Threshold probability
min_size_pixels = 20  #CHANGE - Removes noise smaller than X pixels

effective_pixel_size = pixel_size_nm / expansion_factor
pixel_area_bio_nm2 = effective_pixel_size**2


#2. Loading of the dataset (.h5)

with h5py.File(h5_path, "r") as f:
    prob_map = f["exported_data"][:, :, 0]

print(f"Probability map loaded. Resolution: {prob_map.shape}")


#3. SEGMENTATION
binary_mask = prob_map > probability_threshold
cleaned_mask = remove_small_objects(binary_mask, min_size=min_size_pixels)
final_mask = cleaned_mask
labeled_mask = label(final_mask)

#WATERSHED SPLITTING

distance = ndi.distance_transform_edt(final_mask)
smoothed_distance = ndi.gaussian_filter(distance, sigma=3)

coords = peak_local_max(
    smoothed_distance, 
    min_distance=20,  #Pokud pořád seká moc
    labels=final_mask, 
    footprint=np.ones((3, 3))
)

mask_peaks = np.zeros(distance.shape, dtype=bool)
mask_peaks[tuple(coords.T)] = True
markers, _ = ndi.label(mask_peaks)

labeled_mask = watershed(-smoothed_distance, markers, mask=final_mask)


#4. EXTRACTION
props = regionprops(labeled_mask)

data = []
for region in props:
    data.append(
        {
            "object_id": region.label,
            "area_pixels": region.area,
            "area_bio_nm2": region.area * pixel_area_bio_nm2,
            "centroid_y": region.centroid[0],
            "centroid_x": region.centroid[1],
        }
    )

df = pd.DataFrame(data)
df = df[df['area_pixels'] >= 100]
df.to_csv(output_csv, index=False)
print(f"Measurement completed. Found {len(df)} objects. Saved to '{output_csv}'.")


#5. CONTROL VISUALIZATION

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(prob_map, cmap="hot")
ax[0].set_title("Ilastik Probabilities ")
ax[0].axis("off")

ax[1].imshow(final_mask, cmap="gray")
ax[1].set_title(f"Cleaned mask (Threshold > {probability_threshold})")
ax[1].axis("off")

ax[2].imshow(labeled_mask, cmap="nipy_spectral")
ax[2].set_title(f"Segmented objects (Total: {len(df)})")
ax[2].axis("off")

plt.tight_layout()
plt.savefig('/home/martinfranc/Stažené/Magnify/2D/vysledek.png', dpi=300, bbox_inches='tight')
print("Obrázek uložen jako vysledek.png v /home/martinfranc/Stažené/Magnify/2D/")
