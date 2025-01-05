import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio import warp
from rasterio.enums import Resampling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class ImageClassifier:
    def __init__(self, satellite_images_path, classified_path, color_map, n_neighbors=5):
        """
        Initialize the ImageClassifier with paths and parameters.

        Parameters:
            satellite_images_path (str): Path to the satellite image file.
            classified_path (str): Path to the classified image file.
            color_map (dict): Mapping from class labels to RGB colors.
            n_neighbors (int): Number of neighbors for k-NN classifier.
        """
        self.satellite_path = self.get_best_image(satellite_images_path)
        self.classified_path = classified_path
        self.color_map = color_map
        self.n_neighbors = n_neighbors
        self.rgb_to_label = {v: k for k, v in color_map.items()}
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.sat_meta = None

    def get_best_image(self, folder_path):
        image_list = os.listdir(folder_path)
        best_image = None
        max_valid_pixels = -1

        for name in image_list:
            image_path = os.path.join(folder_path, name)
            try:
                with rasterio.open(image_path) as img:
                    image = img.read()  # Read all bands
                    valid_pixel_count = np.count_nonzero(image != 0)
                    if valid_pixel_count > max_valid_pixels:
                        best_image = name
                        max_valid_pixels = valid_pixel_count
            except Exception as e:
                print(f"Error reading {name}: {e}")
                continue

        if best_image:
            print(f"\nThe image with the highest number of pixels is: {best_image} ({max_valid_pixels} valid pixels)")
            return os.path.join(folder_path, best_image)
        else:
            raise FileNotFoundError("No valid images found to determine the maximum pixel count.")

    def resample_classified_image(self):
        with rasterio.open(self.satellite_path) as sat_src:
            self.sat_meta = sat_src.meta.copy()
            sat_shape = (sat_src.height, sat_src.width)
            sat_transform = sat_src.transform

        with rasterio.open(self.classified_path) as cls_src:
            cls_data = cls_src.read()  # Assuming classified image is RGB (3 bands)

        # Initialize the resampled classified image
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

        return resampled_cls

    def rgb_to_class_labels(self, resampled_cls):
        rgb = np.stack([resampled_cls[0], resampled_cls[1], resampled_cls[2]], axis=-1)
        rows, cols, _ = rgb.shape
        labels = np.zeros((rows, cols), dtype=np.int32)  # Initialize with NoData

        # Flatten arrays for processing
        rgb_flat = rgb.reshape(-1, 3)
        labels_flat = labels.flatten()

        # Convert each RGB tuple to a class label
        for idx, pixel in enumerate(rgb_flat):
            pixel_tuple = tuple(pixel)
            labels_flat[idx] = self.rgb_to_label.get(pixel_tuple, 0)  # Default to 0 (NoData) if not found

        labels = labels_flat.reshape(rows, cols)
        return labels

    def load_satellite_bands(self):
        with rasterio.open(self.satellite_path) as src:
            bands = src.read()  # Shape: (bands, rows, cols)
            self.sat_meta = src.meta.copy()
        return bands, self.sat_meta

    def mask_no_data(self, bands, class_labels, nodata_value=None):
        if nodata_value is not None:
            mask = np.any(bands == nodata_value, axis=0)
        else:
            mask = np.isnan(bands).any(axis=0)

        mask |= (class_labels == 0)  # Also mask NoData class

        return mask

    def prepare_dataset(self, bands, labels, mask):
        # Transpose bands to shape (rows, cols, bands)
        bands = bands.transpose(1, 2, 0)
        rows, cols, num_bands = bands.shape

        # Flatten the arrays
        X = bands.reshape(-1, num_bands)
        y = labels.flatten()

        # Apply mask
        X = X[~mask.flatten()]
        y = y[~mask.flatten()]

        return X, y

    def train_classifier(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def evaluate_classifier(self, X, y, dataset_type="Test"):
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')  # Weighted F1 score

        print(f'{dataset_type} Accuracy: {accuracy * 100:.2f}%')
        print(f'{dataset_type} F1 Score: {f1:.2f}')

        conf_matrix = confusion_matrix(y, y_pred)
        print(f"\n{dataset_type} Confusion Matrix:")
        print(conf_matrix)

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot(cmap='Blues')
        plt.title(f"{dataset_type} Set Confusion Matrix")
        plt.show()

    def classify_image(self, bands, mask):
        # Transpose bands to shape (rows, cols, bands)
        bands = bands.transpose(1, 2, 0)
        rows, cols, num_bands = bands.shape

        # Flatten the bands
        X_all = bands.reshape(-1, num_bands)

        # Apply mask
        X_all_valid = X_all[~mask.flatten()]

        # Predict
        y_pred = self.classifier.predict(X_all_valid)

        # Create a full prediction array
        y_full = np.zeros(rows * cols, dtype=np.int32)
        y_full[~mask.flatten()] = y_pred
        y_full = y_full.reshape(rows, cols)

        return y_full

    def labels_to_rgb(self, predicted_labels):
        rows, cols = predicted_labels.shape
        rgb_image = np.zeros((rows, cols, 3), dtype=np.uint8)

        for class_label, color in self.color_map.items():
            rgb_image[predicted_labels == class_label] = color

        return rgb_image

    def save_classified_image(self, output_path, labels):
        out_meta = self.sat_meta.copy()
        out_meta.update({
            "count": 1,
            "dtype": rasterio.int32,
            "driver": "GTiff"  # Ensure the driver is correct
        })

        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(labels, 1)

    def save_rgb_image(self, output_path, rgb_image):
        out_meta = self.sat_meta.copy()
        out_meta.update({
            "count": 3,
            "dtype": rasterio.uint8,
            "driver": "GTiff"
        })

        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(rgb_image[:, :, 0], 1)  # Red
            dest.write(rgb_image[:, :, 1], 2)  # Green
            dest.write(rgb_image[:, :, 2], 3)  # Blue

    def plot_classified_image(self, rgb_image):
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.title('k-NN Classified Image (RGB Visualization)')
        plt.axis('off')
        plt.show()
