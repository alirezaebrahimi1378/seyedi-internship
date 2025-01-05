from classify_raster import classify_raster
from image_downloader import SentinelImageDownloader
from classification import ImageClassifier

import os
from sklearn.model_selection import train_test_split


def main():
    # Get the directory of the script (where main.py is located)
    dir = os.path.dirname(__file__)

    # Define Parameteres
    start_date = "2021-08-01"
    end_date = "2021-09-01"
    shapefile_path = os.path.join(dir, "ShapeFile", "Region_little.shp")
    raster_path = os.path.join(dir, "GroundTruthRaster", "l8_aa13_northwest_mosaic2.img")
    output_image_path = os.path.join(dir, "ClipedImage", "classified_data_colored_little.tif")
    output_folder_path = os.path.join(dir, "Sentinel2Images")
    color_map = {
        0: (0, 0, 0),  # NoData (black)
        1: (255, 0, 0),  # Class 1 (red)
        2: (0, 255, 0),  # Class 2 (green)
        3: (0, 0, 255),  # Class 3 (blue)
        4: (255, 255, 0),  # Class 4 (yellow)
        5: (0, 255, 255),  # Class 5 (cyan)
        6: (255, 0, 255),  # Class 6 (magenta)
        7: (128, 128, 128),  # Class 7 (gray)
        8: (255, 128, 0),  # Class 8 (orange)
        9: (128, 0, 128),  # Class 9 (purple)
        11: (0, 128, 0),  # Class 11 (dark green)
        12: (128, 128, 0),  # Class 12 (olive)
        13: (0, 0, 128),  # Class 13 (dark blue)
        14: (128, 0, 0),  # Class 14 (dark red)
        15: (64, 64, 64),  # Class 15 (dark gray)
        17: (255, 255, 255),  # Class 17 (white)
        18: (0, 255, 128),  # Class 18 (lime)
        19: (0, 128, 128),  # Class 19 (teal)
    }

    # Create Classified Data From Ground Truth Data for the Desired Region
    result_path = classify_raster(shapefile_path, raster_path, output_image_path, color_map)

    print(f"Processed raster saved at {result_path}")

    # Download Sentinel2 Images for the Desired Region
    downloader = SentinelImageDownloader(shapefile_path, start_date, end_date, output_folder_path)

    # Run the image download process
    downloader.run()


    # Initialize the ImageClassifier
    classifier = ImageClassifier(
        satellite_images_path=output_folder_path,
        classified_path=result_path,
        color_map=color_map,
        n_neighbors=len(color_map)
    )

    # Resample the classified image
    print("Resampling the classified image...")
    resampled_cls = classifier.resample_classified_image()

    # Convert resampled classified image to class labels
    print("Converting RGB to class labels...")
    class_labels = classifier.rgb_to_class_labels(resampled_cls)

    # Load satellite bands
    print("Loading satellite bands...")
    satellite_bands, sat_meta = classifier.load_satellite_bands()

    # Handle NoData (replace None with actual NoData value if available)
    nodata_value = None  # e.g., nodata_value = -9999
    print("Creating mask for NoData...")
    mask = classifier.mask_no_data(satellite_bands, class_labels, nodata_value)

    # Prepare dataset
    print("Preparing dataset...")
    X, y = classifier.prepare_dataset(satellite_bands, class_labels, mask)

    # Split the dataset
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train the classifier
    print("Training the k-NN classifier...")
    classifier.train_classifier(X_train, y_train)

    # Evaluate the classifier on training data
    print("\nEvaluating on Training Data:")
    classifier.evaluate_classifier(X_train, y_train, dataset_type="Training")

    # Evaluate the classifier on test data
    print("\nEvaluating on Test Data:")
    classifier.evaluate_classifier(X_test, y_test, dataset_type="Test")

    # Classify the entire image
    print("Classifying the entire image...")
    predicted_labels = classifier.classify_image(satellite_bands, mask)

    # Save the classified image as a single-band grayscale image with class labels
    output_classified_path = 'classified_output_labels.tif'
    print(f"Saving classified label image to {output_classified_path}...")
    classifier.save_classified_image(output_classified_path, predicted_labels)
    print(f"Classified label image saved to {output_classified_path}")

    # Map predicted labels to RGB for visualization
    print("Converting class labels to RGB for visualization...")
    predicted_rgb = classifier.labels_to_rgb(predicted_labels)

    # Save the RGB visualization
    output_rgb_path = 'classified_output_rgb.tif'
    print(f"Saving classified RGB image to {output_rgb_path}...")
    classifier.save_rgb_image(output_rgb_path, predicted_rgb)
    print(f"Classified RGB image saved to {output_rgb_path}")

    # Visualize the classified RGB image
    print("Displaying the classified RGB image...")
    classifier.plot_classified_image(predicted_rgb)

if __name__ == "__main__":
    main()
