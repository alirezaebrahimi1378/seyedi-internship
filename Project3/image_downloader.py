import os
import ee
import geemap
import geopandas as gpd
from geemap import ee_export_image

# Authenticate with Earth Engine
ee.Authenticate()

# Initialize the Earth Engine API
ee.Initialize(project='minaseyedi-internship')


class SentinelImageDownloader:
    def __init__(self, shapefile_path, start_date, end_date, output_folder):
        self.shapefile_path = shapefile_path
        self.start_date = start_date
        self.end_date = end_date
        self.output_folder = output_folder
        self.aoi = self.load_shapefile()
        self.cloud_masking = self.maskS2clouds

    def load_shapefile(self):
        # Load the shapefile using GeoPandas
        gdf = gpd.read_file(self.shapefile_path)

        # Reproject the GeoDataFrame to WGS84 (EPSG:4326) for compatibility with Earth Engine
        gdf = gdf.to_crs(epsg=4326)

        # Combine all geometries using union_all instead of unary_union
        geometry = gdf.geometry.union_all()

        # Get the coordinates of the exterior boundary
        if geometry.geom_type == 'GeometryCollection':
            coords = [list(geom.exterior.coords) if geom.geom_type == 'Polygon' else [] for geom in geometry]
            coords = [coord for sublist in coords for coord in sublist]
        else:
            coords = list(geometry.exterior.coords)

        return ee.Geometry.Polygon(coords)

    def maskS2clouds(self, image):
        # Select the QA60 band (cloud and cirrus bitmask)
        qa = image.select('QA60')

        # Define bitmasks for clouds and cirrus
        cloudBitMask = 1 << 10  # Cloud bit
        cirrusBitMask = 1 << 11  # Cirrus bit

        # Apply bitwise AND to isolate cloud-free pixels
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
            qa.bitwiseAnd(cirrusBitMask).eq(0)
        )
        # Update the image mask and return the masked image
        return image.updateMask(mask) \
            .select("B.*") \
            .copyProperties(image, ["system:time_start"])


    def get_sentinel1_images(self):
        sentinel1_ImageCollections = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(self.aoi) \
            .filterDate(self.start_date, self.end_date) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .select(['VV', 'VH'])  # Select 'VV' and 'VH' bands

        count = sentinel1_ImageCollections.size().getInfo()

        if count > 0:
            sentinel1_composite = sentinel1_ImageCollections.median()  # Compute the median composite
            print("Generated Sentinel-1 median composite.")
            return sentinel1_composite
        else:
            print("No Sentinel-1 images found for the given time range.")
            return None

    def get_sentinel2_images(self):
        sentinel2_ImageCollections = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
            .filterBounds(self.aoi) \
            .filterDate(self.start_date, self.end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
            .map(self.maskS2clouds) \
            .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']))

        count = sentinel2_ImageCollections.size().getInfo()
        print(f"Number of Sentinel-2 images: {count}")

        if count > 0:
            # Take the median across all images in the collection
            median_composite = sentinel2_ImageCollections.median().clip(self.aoi)
            print("Created median composite for Sentinel-2.")
            return median_composite
        else:
            print("No Sentinel-2 images found for the given time range.")
            return None

    def get_ESA_WorldCover_v100_image(self):
        count = ee.ImageCollection('ESA/WorldCover/v100').size().getInfo()
        print(f"Number of Sentinel-100 images: {count}")

        return ee.ImageCollection('ESA/WorldCover/v100').first()

    def get_ESA_WorldCover_v200_image(self):
        count = ee.ImageCollection('ESA/WorldCover/v200').size().getInfo()
        print(f"Number of Sentinel-200 images: {count}")

        return ee.ImageCollection('ESA/WorldCover/v200').first()

    def download_images(self, image_collection, prefix):
        # Ensure the output directory exists
        os.makedirs(self.output_folder, exist_ok=True)

        def download_image_tolocal(image):
            # Make sure the output folder exists
            os.makedirs(self.output_folder, exist_ok=True)

            # Export the image as a GeoTIFF to Local Storage
            try:
                image_id = image.get('system:index').getInfo()
                if not image_id or image_id == "None":
                    image_id = f"medianComposite_{self.start_date}_{self.end_date}"
            except Exception as e:
                print(f"Error retrieving image ID: {e}")
                return

            print(f"Downloading each band of {image_id} as separate files...")

            # List all Sentinel-2 bands you want
            band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

            for b in band_names:
                band_image = image.select(b).clip(self.aoi)
                file_name = os.path.join(self.output_folder, f"{b}_{image_id}.tif")
                print(f"Downloading band {b} to {file_name}")

                # You can use either geemap.download_ee_image or geemap.ee_export_image:
                task = geemap.download_ee_image(
                    image=band_image,
                    filename=file_name,
                    scale=10,
                    region=self.aoi,
                    crs='EPSG:4326'
                )

                if task is not None:
                    task.start()
                    print(f"Started download task for {file_name}")
                else:
                    print(f"Failed to create download task for {file_name}")

        # def download_image_toDrive(image):
        #     # Export the image as a GeoTIFF to Google Drive
        #     image_id = image.get('system:index').getInfo()
        #     filename = f"{self.output_folder}/{prefix}_{image_id}.tif"
        #     task = ee.batch.Export.image.toDrive(
        #         image=image.clip(self.aoi),
        #         description=f"export_{prefix}_{image_id}",
        #         fileNamePrefix=filename,
        #         region=self.aoi,
        #         scale=10,
        #         crs='EPSG:4326'
        #     )
        #     task.start()

        # Check if the input is an ImageCollection or a single Image
        if isinstance(image_collection, ee.ImageCollection):
            image_list = image_collection.toList(image_collection.size())  # Get the list of images
            for i in range(image_list.size().getInfo()):  # Iterate through the list
                image = ee.Image(image_list.get(i))
                download_image_tolocal(image)
                # download_image_toDrive(image)
        elif isinstance(image_collection, ee.Image):
            # If it's a single image, download it directly
            download_image_tolocal(image_collection)
            # download_image_toDrive(image_collection)

    def run(self):
        # sentinel1_image_collection = self.get_sentinel1_images()
        sentinel2_image = self.get_sentinel2_images()
        # esa_100_image_collection = self.get_ESA_WorldCover_v100_image()
        # esa_200_image_collection = self.get_ESA_WorldCover_v200_image()

        # print("Downloading Sentinel-1 images...")
        # if sentinel1_image_collection.size().getInfo() > 0:
        #     self.download_images(sentinel1_image_collection, "S1")
        # else:
        #     print("No Sentinel-1 images found.")

        print("Downloading Sentinel-2 median composite...")
        if sentinel2_image is not None:
            self.download_images(sentinel2_image, "S2")
        else:
            print("No Sentinel-2 images found.")

        # print("Downloading ESA v100 images...")
        # if esa_100_image_collection:
        #     self.download_images(esa_100_image_collection, "ESA100")
        # else:
        #     print("No ESA v100 images found.")

        # print("Downloading ESA v200 images...")
        # if esa_200_image_collection:
        #     self.download_images(esa_200_image_collection, "ESA200")
        # else:
        #     print("No ESA v200 images found.")