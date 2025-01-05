import os
import rasterio
import matplotlib
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.mask import mask
from shapely.geometry import shape

matplotlib.use('inline')

def classify_raster(shapefile_path, raster_path, output_path, color_map):
    # Read Shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Open the Raster (.img) File
    with rasterio.open(raster_path) as src:
        # Mask the raster data with the shapefile polygon(s)
        geometry = [shape(shapefile.geometry.iloc[0])]
        out_image, out_transform = mask(src, geometry, crop=True)

        # Retrieve the metadata of the raster
        out_meta = src.meta

        # Update the metadata for the output raster
        out_meta.update({
            "driver": "GTiff",
            "count": 1,
            "dtype": out_image.dtype,
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    # Convert raster pixels to polygons (single-band raster)
    raster_classes = out_image[0]  # Assuming a single-band raster
    raster_classes[raster_classes == src.nodata] = 0  # Replace NoData values with 0

    # Apply the color map
    colored_raster = np.zeros((raster_classes.shape[0], raster_classes.shape[1], 3), dtype=np.uint8)

    for class_val, color in color_map.items():
        colored_raster[raster_classes == class_val] = color

    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Update metadata for the colored image (3 channels for RGB)
    out_meta.update({
        "count": 3,  # Three channels (RGB)
        "dtype": "uint8",
    })

    # Write the colored image to a new GeoTIFF file
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(colored_raster[:, :, 0], 1)  # Red channel
        dst.write(colored_raster[:, :, 1], 2)  # Green channel
        dst.write(colored_raster[:, :, 2], 3)  # Blue channel

    print(f"GeoTIFF with colored classes saved to {output_path}")

    # Optionally, visualize the result
    plt.imshow(colored_raster)
    plt.title("Classified Raster with Color Map")
    plt.show()

    return output_path  # Return the path to the saved raster
