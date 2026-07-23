import pandas as pd
import tifffile

# 1. PARAMETRY A NAČTENÍ OBRÁZKU
path_to_file = (
    "/home/martinfranc/Stažené/Magnify/2D/MAGNIFY_2_Image_G001_ch01_fixed.tif"
)
expansion_factor = 4

# Načtení/kontrola metadat
with tifffile.TiffFile(path_to_file) as tif:
    page = tif.pages[0]
    if "XResolution" in page.tags:
        res_tuple = page.tags["XResolution"].value
        x_res = res_tuple[0] / res_tuple[1]

# Fallback logika pro velikost pixelu
if x_res == 1.0 or x_res <= 0:
    print("⚠️ Detekována neplatná metadata, používám výchozí hodnotu 20 nm/px.")
    pixel_size_nm = 20
else:
    print(f"✅ Nalezena platná hodnota rozlišení: {x_res}")
    pixel_size_nm = x_res  # Zde by byl případný přepočet z cm/palců

# 2. TESTOVACÍ DATA (Simulace výstupu ze segmentace)
# Představ si, že ilastik našel 3 jaderná tělíska o různých velikostech v pixelech:
detected_objects = [
    {"object_id": 1, "area_pixels": 150},
    {"object_id": 2, "area_pixels": 420},
    {"object_id": 3, "area_pixels": 85},
]

# 3. AUTOMATICKÝ PŘEPOČET PRO VŠECHNY OBJEKTY
for obj in detected_objects:
    px_area = obj["area_pixels"]
    # Vzorec pro biologickou plochu
    bio_area_nm2 = (px_area * (pixel_size_nm**2)) / (expansion_factor**2)
    obj["area_bio_nm2"] = bio_area_nm2

# 4. ZOBRAZENÍ VÝSLEDKŮ V TABULCE
df = pd.DataFrame(detected_objects)
print("\n--- VÝSLEDNÁ TABULKA MĚŘENÍ ---")
print(df)