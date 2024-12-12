import ee
import geemap
import geopandas as gpd

# Initialize the Earth Engine API
ee.Initialize()


class SentinelImageDownloader:
    def __init__(self, shapefile_path, start_date, end_date, output_folder):
        self.shapefile_path = shapefile_path
        self.start_date = start_date
        self.end_date = end_date
        self.output_folder = output_folder
        self.aoi = self.load_shapefile()

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

    def get_sentinel1_images(self):
        sentinel1_ImageCollections = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(self.aoi) \
            .filterDate(self.start_date, self.end_date) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .select(['VV', 'VH'])   # Select 'VV' and 'VH' bands
        return sentinel1_ImageCollections

    def get_sentinel2_images(self):
        sentinel2_ImageCollections = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
            .filterBounds(self.aoi) \
            .filterDate(self.start_date, self.end_date) \
            .filter(ee.Filter.lt('cloud_coverage', 30)) \
            .select(['B4', 'B3', 'B2'])
        return sentinel2_ImageCollections

    def get_ESA_WorldCover_v100_image(selfself):
        return ee.ImageCollection('ESA/WorldCover/v100').first()

    def get_ESA_WorldCover_v200_image(selfself):
        return ee.ImageCollection('ESA/WorldCover/v200').first()

    def download_images(self, image_collection, prefix):
        def download_image_tolocal(image):
            # Export the image as a GeoTIFF to Local Storage
            image_id = image.get('system_index').getInfo()
            file_name = f'E:\\_M_Aterm\\Sedreh\\GEE\\Sentinel_images\\{prefix}_{image_id}.tif'
            task = geemap.download_ee_image(
                image=image.clip(self.aoi),
                file_name=file_name,
                scale=10,
                region=self.aoi,
                crs='EPSG:4326'
            )

        def download_image_toDrive(image):
            # Export the image as a GeoTIFF to Drive
            image_id = image.get('system:index').getInfo()
            filename = f"{self.output_folder}/{prefix}_{image_id}.tif"
            task = ee.batch.Export.image.toDrive(
                image=image.clip(self.aoi),
                description=f"export_{prefix}_{image_id}",
                fileNamePrefix=filename,
                region=self.aoi,
                scale=10,
                crs='EPSG:4326'
            )
            task.start()

        # Iterate over the collection and download each image
        image_collection.evaluate(lambda coll: [download_image_toDrive(ee.Image(img)) for img in coll['features']])
        image_collection.evaluate(lambda coll: [download_image_tolocal(ee.Image(img)) for img in coll['features']])

    def run(self):
        sentinel1_image_collection = self.get_sentinel1_images()
        sentinel2_image_collection = self.get_sentinel2_images()
        esa_100_image_collection = self.get_ESA_WorldCover_v100_image()
        esa_200_image_collection = self.get_ESA_WorldCover_v200_image()

        print("Downloading Sentinel-1 images...")
        self.download_images(sentinel1_image_collection, "S1")

        print("Downloading Sentinel-2 images...")
        self.download_images(sentinel2_image_collection, "S2")

        print("Downloading ESA v100 images...")
        self.download_images(esa_100_image_collection, "ESA100")

        print("Downloading ESA v200 images...")
        self.download_images(esa_200_image_collection, "ESA100")


if __name__ == "__main__":
    # Define your parameters
    shapefile_path = "E:\\_M_Aterm\\Sedreh\\GEE\\ArcGIS\\Alamdeh.shp"
    start_date = "2020-01-01"
    end_date = "2020-01-02"
    output_folder = "output_directory"

    # Create an instance of the downloader class
    downloader = SentinelImageDownloader(shapefile_path, start_date, end_date, output_folder)

    # Run the image download process
    downloader.run()