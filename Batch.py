from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.io import imread
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
import tifffile



def process_pair(
    tif_path,
    h5_path,
    expansion_factor=4.0,
    prob_thresh=0.7,
    min_size_px=20,
    min_area_filter=100,
):
    tif_path = Path(tif_path)
    h5_path = Path(h5_path)

    #METADATA
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        x_res = 1.0
        if "XResolution" in page.tags:
            res_tuple = page.tags["XResolution"].value
            x_res = res_tuple[0] / res_tuple[1]
        unit = page.tags.get("ResolutionUnit", None)
        unit_val = unit.value if unit else None

    #RECALCULATE PIXEL SIZE
    if x_res <= 1.0:
        pixel_size_nm = 20.0
    else:
        if unit_val == 2:  #Inches
            pixel_size_nm = 25_400_000 / x_res
        elif unit_val == 3:  #Centimeters
            pixel_size_nm = 10_000_000 / x_res
        else:
            pixel_size_nm = 1_000_000_000 / x_res
            if pixel_size_nm > 500:
                pixel_size_nm = 10_000_000 / x_res

    #LOAD IMAGE
    img = imread(tif_path)

    #WATERSHED SEGMENTACE
    with h5py.File(h5_path, "r") as f:
        prob_map = f["exported_data"][:, :, 0]

    binary_mask = prob_map > prob_thresh
    cleaned_mask = remove_small_objects(binary_mask, min_size=min_size_px)

    distance = ndi.distance_transform_edt(cleaned_mask)
    smoothed_distance = ndi.gaussian_filter(distance, sigma=3)

    coords = peak_local_max(
        smoothed_distance,
        min_distance=20,
        labels=cleaned_mask,
        footprint=np.ones((3, 3)),
    )

    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_peaks)

    labeled_mask = watershed(-smoothed_distance, markers, mask=cleaned_mask)

    #EFFCETIVE PIXEL SIZE 
    effective_pixel_size = pixel_size_nm / expansion_factor
    pixel_area_bio_nm2 = effective_pixel_size**2

    #Mask from Ilastik to image
    props = regionprops(labeled_mask, intensity_image=img)

    objects_data = []
    for region in props:
        if region.area >= min_area_filter:
            mean_int = region.mean_intensity
            int_density = region.area * mean_int

            objects_data.append({
                "filename": tif_path.name,
                "object_id": region.label,
                "area_pixels": region.area,
                "area_bio_nm2": region.area * pixel_area_bio_nm2,
                "mean_intensity": round(mean_int, 2),
                "integrated_density": round(int_density, 2),
                "centroid_y": region.centroid[0],
                "centroid_x": region.centroid[1],
            })

    return pd.DataFrame(objects_data)

folder_path = Path(r"C:\Users\franc\Desktop\ExM_analyza")
all_dataframes = []

#Look for all .tif files
for tif_file in folder_path.glob("*.tif"):
    matching_h5 = list(folder_path.glob(f"{tif_file.stem}*.h5"))

if matching_h5:
    h5_file = matching_h5[0]
    print(f"Zpracovávám dvojici: {tif_file.name} + {h5_file.name}")
    df_pair = process_pair(tif_file, h5_file)
    all_dataframes.append(df_pair)
else:
    print(f"Warning: No corresponding H5 file found for {tif_file.name}")

    if h5_file.exists():
        print(f"Processing pair: {tif_file.name} + {h5_file.name}")
        df_pair = process_pair(tif_file, h5_file)
        all_dataframes.append(df_pair)
    else:
        print(f"Warning: No corresponding H5 file found for {tif_file.name}")

#Combine all dataframes and save to CSV
if all_dataframes:
    final_df = pd.concat(all_dataframes, ignore_index=True)
    output_csv = folder_path / "Final_Output_Batch.csv"
    final_df.to_csv(output_csv, index=False)

    print("\nBatch completed.")
    print(f"Detected objects: {len(final_df)}")
    print(f"Saved to: {output_csv}")
    print(final_df.head())