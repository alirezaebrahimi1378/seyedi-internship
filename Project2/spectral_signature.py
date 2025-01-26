import os
import random
import rasterio
import numpy as np
from rasterio import warp
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

class SpectralSignaturePlotter:
    def __init__(self, cls_path, sentinel2_raster_path, output_folder, sample_points, band_names, color_map):
        self.cls_path = cls_path
        self.sentinel2_raster_path = sentinel2_raster_path
        self.output_folder = output_folder
        self.sample_points = sample_points
        self.rgb_to_label = {v: k for k, v in color_map.items()}
        self.labels = self.rgb_to_class_labels()

        # Default band names for Sentinel-2, if not provided
        if band_names is None:
            self.band_names = [
                "B1", "B2", "B3", "B4",
                "B5", "B6", "B7",
                "B8", "B8A",
                "B11", "B12"
            ]
        else:
            self.band_names = band_names

        os.makedirs(self.output_folder, exist_ok=True)

    def rgb_to_class_labels(self):
        with rasterio.open(self.sentinel2_raster_path) as sat_src:
            self.sat_meta = sat_src.meta.copy()
            sat_shape = (sat_src.height, sat_src.width)
            sat_transform = sat_src.transform

        with rasterio.open(self.cls_path) as cls_src:
            cls_data = cls_src.read()

        resampled_cls = np.empty((3, sat_shape[0], sat_shape[1]), dtype=cls_data.dtype)

        for i in range(3):
            warp.reproject(
                source=cls_data[i],
                destination=resampled_cls[i],
                src_transform=cls_src.transform,
                src_crs=cls_src.crs,
                dst_transform=sat_transform,
                dst_crs=cls_src.crs,
                resampling=Resampling.nearest
            )

        rgb = np.stack([resampled_cls[0], resampled_cls[1], resampled_cls[2]], axis=-1)
        labels = np.zeros(rgb.shape[:2], dtype=np.int32)

        for y in range(rgb.shape[0]):
            for x in range(rgb.shape[1]):
                pixel_tuple = tuple(rgb[y, x])
                labels[y, x] = self.rgb_to_label.get(pixel_tuple, 0)

        return labels

    def plot_by_class(self):
        # Debug: Check labels type and shape
        print(f"Labels type: {type(self.labels)}")
        print(f"Labels shape: {self.labels.shape if hasattr(self.labels, 'shape') else 'No shape'}")

        # Read the Multi-band Sentinel-2 Raster
        with rasterio.open(self.sentinel2_raster_path) as s2_src:
            s2_data = s2_src.read()  # shape => (bands, height, width)

        # Check band consistency
        if len(self.band_names) != s2_data.shape[0]:
            raise ValueError(
                f"Mismatch between the number of band names ({len(self.band_names)}) "
                f"and the Sentinel-2 image bands ({s2_data.shape[0]})."
            )

        # Identify Unique Classes (excluding 0 if NoData)
        unique_classes = np.unique(self.labels)
        if 0 in unique_classes:
            unique_classes = unique_classes[unique_classes != 0]

        # Generate One Plot per Class
        for cls_id in unique_classes:
            # Find all pixels for this class
            row_indices, col_indices = np.where(self.labels == cls_id)
            num_pixels = len(row_indices)
            if num_pixels == 0:
                continue

            # Randomly sample up to 'sample_points'
            n_samples = min(self.sample_points, num_pixels)
            sampled_indices = random.sample(range(num_pixels), n_samples)

            # Extract spectral values for each sampled pixel
            spectral_signatures = []
            for idx in sampled_indices:
                r = row_indices[idx]
                c = col_indices[idx]
                pixel_values = s2_data[:, r, c]  # shape => (num_bands,)
                spectral_signatures.append(pixel_values)

            # Plot each pixel's signature
            plt.figure(figsize=(8, 5))

            for i, spectrum in enumerate(spectral_signatures):
                plt.plot(
                    self.band_names,
                    spectrum,
                    marker='o',
                    label=f"Sample {i + 1}"
                )

            plt.xlabel("Sentinel-2 Bands")
            plt.ylabel("Reflectance / DN")
            plt.title(f"Spectral Signatures - Class {cls_id}")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save Plot as .TIF (not GeoTIFF)
            output_path = os.path.join(
                self.output_folder,
                f"class_{cls_id}_spectral_signatures.tif"
            )
            plt.savefig(output_path, format="tiff", dpi=300)
            plt.close()

            print(f"Saved plot for Class {cls_id} => {output_path}")
