# Sentinel Image Downloader

This Python script allows you to download Sentinel-1, Sentinel-2, and ESA WorldCover images from Google Earth Engine (GEE) using a user-provided shapefile (in this case, Alamdeh.shp) as the area of interest (AOI). The downloaded images can be saved either directly to the local machine or to Google Drive as GeoTIFF files.

## Requirements
Before running this script, make sure the following dependencies are installed:

- **Google Earth Engine (GEE) API:** Required to access Earth Engine data.
- **geemap:** A library that facilitates working with Earth Engine and downloading images.
- **geopandas:** A library for working with shapefiles.
- **ee (Earth Engine):** Python bindings for Google Earth Engine.

You can install the necessary libraries using the following commands:
```bash
pip install earthengine-api geemap geopandas
```
Additionally, you need to authenticate with Google Earth Engine:
```bash
earthengine authenticate
```
This will guide you through the process of authenticating with your Google account.

## Overview
This script is designed to:
1. Load the shapefile (for AOI) using GeoPandas.
2. Retrieve Sentinel-1 (SAR) and Sentinel-2 (Optical) images for the specified time range and area of interest.
3. Download these images in GeoTIFF format to your local storage or Google Drive.

## Supported Data:
- **Sentinel-1:** GRD (Ground Range Detected) images for SAR (VV and VH polarizations).
- **Sentinel-2:** Optical imagery (Bands: B4 - Red, B3 - Green, B2 - Blue).
- **ESA WorldCover:** Two versions of ESA WorldCover data (v100 and v200).

## How to Use
### 1. Set Up Your Parameters
In the script, modify the following parameters to suit your needs:

- **shapefile_path:** Path to the shapefile (e.g., ```Alamdeh.shp```) that defines the area of interest (AOI).
- **start_date and end_date:** The time period for which you want to download images. Format: ```YYYY-MM-DD```.
- **output_folder:** Path where the images will be saved (e.g., ```/GEE``` or Google Drive folder).

**Example:**
```python
shapefile_path = "/ArcGIS/Alamdeh.shp"
start_date = "2020-01-01"
end_date = "2020-01-02"
output_folder = "/Sentinel_Images" 
```

### 2. Run the Script
Once the parameters are set, run the script using the command:
```bash
python SentinelImageDownloader.py
```
This will initiate the image download process for the specified Sentinel-1, Sentinel-2, and ESA WorldCover data.

### 3. Image Export

- **Download to Local Storage:** By default, the images will be downloaded to your specified local folder (e.g., ```/Sentinel_Images```).
- **Download to Google Drive:** The script also supports exporting images to Google Drive by using Earth Engine's ```Export.image.toDrive``` function.

**Example Output:**
Images will be saved with filenames in the following format:

```swift
{prefix}_{image_id}.tif
```
Where:
- ```prefix```: Describes the type of image (```S1```, ```S2```, ```ESA100```, ```ESA200```).
- ```image_id```: Unique ID of the image.

Example filenames:
```S1_20200101.tif```
```S2_20200101.tif```
```ESA100_20200101.tif```
```ESA200_20200101.tif```

# Code Walkthrough

### 1. Shapefile Loading (```load_shapefile```)
- Loads the provided shapefile using **GeoPandas**.
- Converts the CRS (Coordinate Reference System) to ```EPSG:4326``` for compatibility with Earth Engine.
- Combines all geometries and returns the AOI as an ```ee.Geometry.Polygon```.

### 2. Get Sentinel-1 Images (```get_sentinel1_images```)
 -Filters the Sentinel-1 image collection by the AOI, date range, and polarization (```VV``` and ```VH```).
- Selects only the required bands (```VV``` and ```VH```) for the SAR imagery.

### 3. Get Sentinel-2 Images (```get_sentinel2_images```)
- Filters the Sentinel-2 image collection by the AOI, date range, and cloud coverage (```<30%```).
- Selects the required optical bands (``B4``, ``B3``, and ``B2`` for ``Red``, ``Green``, and ``Blue``).

### 4. Get ESA WorldCover Images (```get_ESA_WorldCover_v100_image``` and ```get_ESA_WorldCover_v200_image```)
- Retrieves the first image from the ESA WorldCover v100 and v200 image collections.

### 5. Download Images (```download_images```)
- Downloads each image in the provided image collection.
- Supports both exporting to Google Drive and local storage.

### 6. Run Method (```run```)
- Calls the functions to get images from Sentinel-1, Sentinel-2, and ESA WorldCover collections.
- Downloads the images for each collection.


## Troubleshooting

**Error:** ```ModuleNotFoundError```: Make sure you have installed all required Python packages (```geemap```, ```earthengine-api```, ```geopandas```).\
**Error: Authentication:** If you face authentication issues, run ```earthengine authenticate``` to authenticate your Earth Engine account.\
**Large Files:** Sentinel-1 and Sentinel-2 images can be large. Make sure you have enough disk space for the downloaded files.

# Conclusion
This script provides a simple way to download both Sentinel-1 and Sentinel-2 images from Earth Engine, allowing for easy processing of satellite data. Whether you're working on a small or large area of interest, this code can be easily adapted to suit your needs.

