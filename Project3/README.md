# **Iran Sentinel-2 and Land Cover Classification Downloader**

This project demonstrates how to download **Sentinel-2** imagery and **land cover labels** for a specific region in Iran using **Google Earth Engine** and a user-provided shapefile. The project includes scripts to:

- [**Download Sentinel-2 Median Composite**](#sentinel-2-download) images for a given date range.
- [**Extract and vectorize Land Cover Classes**](#land-cover-label-download) (13 distinct classes) as separate GeoJSON files.

---

## **Table of Contents**
- [**Project Structure**](#project-structure)
- [**Prerequisites**](#prerequisites)
- [**Land Cover Dataset**](#land-cover-dataset)
- [**How to Run**](#how-to-run)
- [**Outputs**](#outputs)
  - [**Sentinel-2 Download**](#sentinel-2-download)
  - [**Land Cover Label Download**](#land-cover-label-download)
- [**References**](#references)
- [**License**](#license)

---

## **Project Structure**
```css
.
├── main.py
├── image_downloader.py
├── label_downloader.py
├── ShapeFile
│   └── TehranNorthEast.shp
└── README.md

```

- **[`main.py`](#how-to-run)** → Main script to run the Sentinel-2 and land cover label downloads.
- **[`image_downloader.py`](#sentinel-2-download)** → Downloads each Sentinel-2 band as a separate TIFF file.
- **[`label_downloader.py`](#land-cover-label-download)** → Extracts land cover classes and saves them as separate GeoJSON files.
- **`ShapeFile/TehranNorthEast.shp`** → Example shapefile for the region of interest in Iran.

---

## **Prerequisites**

### 1. **Google Earth Engine Account**
- Sign up for Earth Engine at [Google Earth Engine](https://earthengine.google.com/).
- You must be approved for access to use the Earth Engine Python API.

### 2. **Python Environment**
- Recommended Python **3.8+**.
- Install dependencies:
  
  ```bash
  pip install earthengine-api geemap geedim geopandas shapely fiona pyproj


### 3. **Shapefile**
 - A shapefile of your region of interest (ROI) in **EPSG:4326** or convertible to EPSG:4326.
 - In this repo, an example shapefile named ```TehranNorthEast.shp``` is located in the ```ShapeFile``` folder.

### 4. **Earth Engine Authentication**
Run **earthengine authenticate** or in Python **ee.Authenticate()** to authenticate Earth Engine.


## **Land Cover Dataset**
We use the **Iran Land Cover Map v1** 13-class (2017) dataset:
- Earth Engine Snippet:
```python
ee.Image("KNTU/LiDARLab/IranLandCover/V1")
```

### **Classes in the dataset:**

| Value | Name             |
|-------|------------------|
| 1     | Urban           |
| 2     | Water           |
| 3     | Wetland         |
| 4     | Kalut           |
| 5     | Marshland       |
| 6     | Salty_Land      |
| 7     | Clay            |
| 8     | Forest          |
| 9     | Outcrop         |
| 10    | Uncovered_Plain |
| 11    | Sand            |
| 12    | Farm_Land       |
| 13    | Range_Land      |


Dataset Citation:
Ghorbanian, A., Kakooei, M., Amani, M., Mahdavi, S., Mohammadzadeh, A., & Hasanlou, M. (2020). Improved land cover map of Iran using Sentinel imagery within Google Earth Engine and a novel automatic workflow for land cover classification using migrated training samples.
**ISPRS Journal of Photogrammetry and Remote Sensing**, 167, 276-288.
[doi:10.1016/j.isprsjprs.2020.07.013](https://www.sciencedirect.com/science/article/abs/pii/S0924271620302008)

## **How to Run**
1. **Clone** or **download** this repository.
```bash
git clone https://github.com/your-repo.git
cd your-repo
```
2. Check your Python environment is activated (e.g., ```conda activate your_env``` or ```source venv/bin/activate```).
```bash
conda activate your_env  # or source venv/bin/activate
```
3. **Configure** the date range and shapefile in ```main.py```:
```python
start_date = "2016-01-01"
end_date = "2017-01-01"
shapefile_path = os.path.join(dir, "ShapeFile", "TehranNorthEast.shp")
```

4. **Run Sentinel-2 Download**
```python
downloader = SentinelImageDownloader(shapefile_path, start_date, end_date, output_sentinel_path)
print("Starting Sentinel-2 image download...")
downloader.run()
```
This downloads each Sentinel-2 band (B1...B12) as a separate ```.tif``` file into the ```Sentinel2``` folder.

5. **Run Land Cover Label Download**
In ```main.py```, the ```label_downloader.run()``` call will:
 - Extract 13 classes
 - Save each as ```Iran_LandCover_[CLASS_NAME].geojson``` in the``` Label``` folder.

6. **Run the script:**
```bash
python main.py
```

## **Outputs**

### **Sentinel-2 Download**
- Each band is exported as a **GeoTIFF**:
```css
./Sentinel2/B1_medianComposite_2016-01-01_2017-01-01.tif
./Sentinel2/B2_medianComposite_2016-01-01_2017-01-01.tif
...
```

- **Output folder**: `./Sentinel2/`

### **Land Cover Label Download**
- Each class is vectorized to a **GeoJSON** file, e.g.:
```css
./Label/Iran_LandCover_Urban.geojson
./Label/Iran_LandCover_Water.geojson
...
```
- **Output folder**: `./Label/`

---

## **References**

1. **K. N. Toosi University of Technology LiDAR Lab**  
   - Dataset: [KNTU/LiDARLab/IranLandCover/V1](https://developers.google.com/earth-engine/datasets/catalog/KNTU_LiDARLab_IranLandCover_V1)
 
2. **Dataset Citation**  
 Ghorbanian, A., Kakooei, M., Amani, M., Mahdavi, S., Mohammadzadeh, A., & Hasanlou, M. (2020).  
 *Improved land cover map of Iran using Sentinel imagery within Google Earth Engine and a novel automatic workflow for land cover classification using migrated training samples.*  
 **ISPRS Journal of Photogrammetry and Remote Sensing**, *167*, 276-288.  
 [doi:10.1016/j.isprsjprs.2020.07.013](https://doi.org/10.1016/j.isprsjprs.2020.07.013)

---

## **License**
- Refer to the **[Earth Engine Terms of Use](https://earthengine.google.com/terms/)** for usage limits and guidelines.
