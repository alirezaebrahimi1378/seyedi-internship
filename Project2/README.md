# Satellite Image Processing and Classification Pipeline

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Image Downloading](#1-image-downloading)
  - [2. Raster Classification](#2-raster-classification)
  - [3. Image Classification](#3-image-classification)
  - [4. Running the Pipeline](#4-running-the-pipeline)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

## Overview

This project provides a comprehensive pipeline for downloading, processing, and classifying satellite imagery using Google Earth Engine (GEE) and machine learning techniques. It leverages Sentinel-2 datasets to perform geospatial analysis and classification, enabling users to obtain classified raster outputs based on specified regions and timeframes.

## Features

- **Automated Image Downloading**: Fetch Sentinel-2 images for a specified region and date range.
- **Cloud Masking**: Apply cloud masking to Sentinel-2 images to enhance data quality.
- **Raster Classification**: Convert ground truth raster data into classified GeoTIFFs with customizable color maps.
- **Machine Learning Classification**: Utilize a k-Nearest Neighbors (k-NN) classifier to classify satellite images based on training data.
- **Visualization**: Generate and display classified images in both label and RGB formats.
- **Modular Design**: Organized into separate scripts for downloading, processing, and classification for ease of maintenance and scalability.

## Architecture

The pipeline consists of the following main components:

1. **Image Downloader (`image_downloader.py`)**: Handles authentication with Google Earth Engine, downloads Sentinel-1 and Sentinel-2 images based on user-defined parameters, and saves them locally.

2. **Raster Classifier (`classify_raster.py`)**: Processes ground truth raster data, applies masking based on shapefiles, and generates classified GeoTIFFs with applied color maps.

3. **Image Classifier (`classification.py`)**: Implements a k-NN classifier to train on the classified data and perform classification on satellite images. It also includes evaluation metrics and visualization tools.

4. **Main Pipeline (`main.py`)**: Orchestrates the entire workflow by invoking the downloader, raster classifier, and image classifier in sequence.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **Google Earth Engine Account**: [Sign up here](https://earthengine.google.com/signup/)
- **Earth Engine CLI**: Install using `pip install earthengine-api`


## Usage
### 1. **Image Downloading**
The ```image_downloader.py``` script is responsible for downloading Sentinel-2 images based on the specified shapefile, date range, and output directory.

### 2. **Raster Classification**
The ```classify_raster.py``` script processes ground truth raster data by applying masks based on shapefiles and generating classified GeoTIFFs with a predefined color map.

### 3. **Image Classification**
The ```classification.py``` script trains a k-NN classifier using the classified raster data and applies it to satellite images to produce classified outputs. It also evaluates the classifier's performance and visualizes the results.

### 4. **Running the Pipeline**
The ```main.py``` script orchestrates the entire workflow. Ensure that all required parameters are correctly set within the script before execution.

**Example Execution**
```bash
python main.py
```
Upon running, the script will:
1. Classify the ground truth raster data.
2. Download Sentinel-2 images for the specified region and date range.
3. Initialize the image classifier, resample the classified image, and convert RGB to class labels.
4. Load satellite bands, prepare the dataset, and split it into training and testing sets.
5. Train the k-NN classifier and evaluate its performance.
6. Classify the entire satellite image and save both label and RGB classified outputs.
7. Display the classified RGB image.

## Dependencies
The project relies on the following Python libraries:

- **earthengine-api:** Interface with Google Earth Engine.
- **geemap:** Interactive mapping with Google Earth Engine.
- **geopandas:** Geospatial data manipulation.
- **rasterio:** Raster data access and processing.
- **matplotlib:** Plotting and visualization.
- **numpy:** Numerical operations.
- **scikit-learn:** Machine learning algorithms.
- **shapely:** Geometric operations.

Ensure all dependencies are installed using the provided installation instructions.

## Project Structure
```css
satellite-image-classification/
│
├── ShapeFile/
│   └── Region_little.shp
│
├── GroundTruthRaster/
│   └── l8_aa13_northwest_mosaic2.img
│
├── ClipedImage/
│   └── classified_data_colored_little.tif
│
├── Sentinel2Images/
│   └── [Downloaded Sentinel-2 Images]
│
├── classify_raster.py
├── classification.py
├── image_downloader.py
├── main.py
├── README.md
```

- **ShapeFile/:** Contains shapefiles defining regions of interest.
- **GroundTruthRaster/:** Stores ground truth raster data for classification.
- **ClipedImage/:** Outputs from the raster classification process.
- **Sentinel2Images/:** Directory where downloaded Sentinel-2 images are stored.
- **Scripts:** Python scripts for downloading, processing, and classification.
- **README.md:** Project documentation.


## Acknowledgements
- [Google Earth Engine](https://earthengine.google.com/) for providing access to extensive geospatial datasets.
- [Geemap](https://github.com/gee-community/geemap) for simplifying Earth Engine interactions.
- [Rasterio](https://rasterio.readthedocs.io/en/stable/) for raster data processing.
- [Scikit-learn](https://scikit-learn.org/stable/) for machine learning algorithms.