import numpy as np
import tifffile as tf

path_in = '/home/martinfranc/Stažené/Magnify/2D/MAGNIFY 2_Image_G001_ch01.tif'
path_out = '/home/martinfranc/Stažené/Magnify/2D/MAGNIFY_2_Image_G001_ch01_fixed.tif'

print("Rekonstruuji obrazová data ze syrových bajtů...")

try:
    with open(path_in, 'rb') as f:
        raw_data = f.read()
    
    # 1816 x 1816 px u 16-bit obrázku = 1816 * 1816 * 2 bajty
    num_pixels = 1816 * 1816
    expected_bytes = num_pixels * 2
    
    # Načteme přesně data z konce souboru (přeskočíme poškozenou hlavičku)
    raw_pixels = raw_data[-expected_bytes:]
    
    array_1d = np.frombuffer(raw_pixels, dtype=np.uint16)
    array_2d = array_1d.reshape((1816, 1816))
    
    tf.imwrite(path_out, array_2d)
    print(f"Rekonstrukce úspěšná! Uloženo do: {path_out}")

except Exception as e:
    print(f"Chyba při rekonstrukci: {e}")