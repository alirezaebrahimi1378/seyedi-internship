import os
import rasterio
import matplotlib
import numpy as np
from rasterio import warp
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

import warnings

matplotlib.use('Agg')  # Keep this if you are in a headless environment

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class ImageClassifier:
    def __init__(self, satellite_images_path, classified_path, color_map, n_neighbors=18):
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

        # # Merge classes
        # labels[labels == 2] = 1
        # labels[labels == 3] = 1
        # labels[labels == 5] = 1
        # labels[labels == 13] = 1

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
        # # Merge classes
        # labels[labels == 2] = 1
        # labels[labels == 3] = 1
        # labels[labels == 5] = 1
        # labels[labels == 13] = 1

        # Transpose bands to shape (rows, cols, bands)
        bands = bands.transpose(1, 2, 0)
        rows, cols, num_bands = bands.shape

        # Flatten the arrays
        X = bands.reshape(-1, num_bands)
        y = labels.flatten()

        # Apply mask
        X = X[~mask.flatten()]
        y = y[~mask.flatten()]

        # Print the unique classes in the data
        unique_classes = np.unique(y)
        print(f"\nUnique classes in the dataset: {unique_classes}")

        return X, y

    def train_classifier(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def evaluate_classifier(self, X, y, dataset_type="Test"):
        y_pred = self.classifier.predict(X)
        unique_classes = np.unique(y)
        overall_accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')  # Weighted F1 score

        print(f'\n{dataset_type} Overall Accuracy: {overall_accuracy * 100:.2f}%')
        print(f'{dataset_type} Weighted F1 Score: {f1:.2f}')

        # Compute confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        print(f"\n{dataset_type} Confusion Matrix:")
        print(conf_matrix)

        # Plot confusion matrix and save it to a file
        # Assuming unique_classes is a list or array of class names (or labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=unique_classes)
        fig, ax = plt.subplots(figsize=(16, 16))
        disp.plot(cmap='Blues', ax=ax)

        # Set the title and adjust tick labels with a smaller font size
        ax.set_title(f"{dataset_type} Set Confusion Matrix", fontsize=24)
        ax.set_xticks(np.arange(len(unique_classes)))
        ax.set_xticklabels(unique_classes, fontsize=18, rotation=45)
        ax.set_yticks(np.arange(len(unique_classes)))
        ax.set_yticklabels(unique_classes, fontsize=18)

        # Loop over text elements in the axes (the numbers in the cells) and set a smaller font size
        for text in ax.texts:
            text.set_fontsize(16)

        plt.tight_layout()
        plt.show()

        # Save the figure to file
        fig.savefig(f"{dataset_type}_confusion_matrix.png")
        print(f"Saved {dataset_type} confusion matrix to {dataset_type}_confusion_matrix.png")
        plt.close(fig)

        # Compute and print per-class accuracy
        per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        class_labels = np.unique(y)
        print(f"\nPer-class accuracy for {dataset_type} set:")
        for cls, acc in zip(class_labels, per_class_accuracy):
            print(f"  Class {cls}: {acc * 100:.2f}%")

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

        self.save_rgb_image("classified_rgb_image.tif", rgb_image)
        print("Saved classified RGB image as classified_rgb_image.tif")
        plt.close()


# ##################################################################
# if __name__ == "__main__":
#     # Get the directory of this script
#     dir = os.path.dirname(__file__)
#
#     # Define Parameters
#     images_path = os.path.join(dir, 'Sentinel2Images')
#     label_path = os.path.join(dir, 'ClipedImage', 'classified_data_colored_little.tif')
#     color_map = {
#         0: (0, 0, 0),  # NoData (black)
#         1: (255, 0, 0),  # Class 1 (red)
#         2: (0, 255, 0),  # Class 2 (green)
#         3: (0, 0, 255),  # Class 3 (blue)
#         4: (255, 255, 0),  # Class 4 (yellow)
#         5: (0, 255, 255),  # Class 5 (cyan)
#         6: (255, 0, 255),  # Class 6 (magenta)
#         7: (128, 128, 128),  # Class 7 (gray)
#         8: (255, 128, 0),  # Class 8 (orange)
#         9: (128, 0, 128),  # Class 9 (purple)
#         11: (0, 128, 0),  # Class 11 (dark green)
#         12: (128, 128, 0),  # Class 12 (olive)
#         13: (0, 0, 128),  # Class 13 (dark blue)
#         14: (128, 0, 0),  # Class 14 (dark red)
#         15: (64, 64, 64),  # Class 15 (dark gray)
#         17: (255, 255, 255),  # Class 17 (white)
#         18: (0, 255, 128),  # Class 18 (lime)
#         19: (0, 128, 128),  # Class 19 (teal)
#     }
#
#     # Here we use n_neighbors equal to the number of classes in the color_map.
#     # If your use case requires ignoring NoData (label 0) then you might subtract one.
#     classifier = ImageClassifier(
#         satellite_images_path=images_path,
#         classified_path=label_path,
#         color_map=color_map,
#         n_neighbors=len(color_map)
#     )
#
#     # Resample the classified image
#     print("Resampling the classified image...")
#     resampled_cls = classifier.resample_classified_image()
#
#     # Convert resampled classified image to class labels
#     print("Converting RGB to class labels...")
#     class_labels = classifier.rgb_to_class_labels(resampled_cls)
#
#     # Load satellite bands
#     print("Loading satellite bands...")
#     satellite_bands, sat_meta = classifier.load_satellite_bands()
#
#     # Handle NoData (replace None with actual NoData value if available)
#     nodata_value = None  # e.g., nodata_value = -9999
#     print("Creating mask for NoData...")
#     mask = classifier.mask_no_data(satellite_bands, class_labels, nodata_value)
#
#     # Prepare dataset
#     print("Preparing dataset...")
#     X, y = classifier.prepare_dataset(satellite_bands, class_labels, mask)
#
#     # Split the dataset
#     print("Splitting dataset into training and testing sets...")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42, stratify=y
#     )
#
#     # Train the classifier
#     print("Training the k-NN classifier...")
#     classifier.train_classifier(X_train, y_train)
#
#     # Evaluate the classifier on training data
#     print("\nEvaluating on Training Data:")
#     classifier.evaluate_classifier(X_train, y_train, dataset_type="Training")
#
#     # Evaluate the classifier on test data
#     print("\nEvaluating on Test Data:")
#     classifier.evaluate_classifier(X_test, y_test, dataset_type="Test")
#
#     # Classify the entire image
#     print("Classifying the entire image...")
#     predicted_labels = classifier.classify_image(satellite_bands, mask)
#
#     # Save the classified image as a single-band grayscale image with class labels
#     output_classified_path = 'classified_output_labels.tif'
#     print(f"Saving classified label image to {output_classified_path}...")
#     classifier.save_classified_image(output_classified_path, predicted_labels)
#     print(f"Classified label image saved to {output_classified_path}")
#
#     # Map predicted labels to RGB for visualization
#     print("Converting class labels to RGB for visualization...")
#     predicted_rgb = classifier.labels_to_rgb(predicted_labels)
#
#     # Save the RGB visualization
#     output_rgb_path = 'classified_output_rgb.tif'
#     print(f"Saving classified RGB image to {output_rgb_path}...")
#     classifier.save_rgb_image(output_rgb_path, predicted_rgb)
#     print(f"Classified RGB image saved to {output_rgb_path}")
#
#     # Visualize the classified RGB image (it will be saved to a file)
#     print("Displaying the classified RGB image...")
#     classifier.plot_classified_image(predicted_rgb)
