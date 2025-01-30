import os
import ee
import geemap
import geopandas as gpd

# Authenticate with Earth Engine
ee.Authenticate()

# Initialize the Earth Engine API
ee.Initialize(project='minaseyedi-internship')


class LandCoverLabelDownloader:
    def __init__(self, shapefile_path, output_folder):
        self.shapefile_path = shapefile_path
        self.output_folder = output_folder
        self.aoi = self.load_shapefile()

        # Ensure the output directory exists
        os.makedirs(self.output_folder, exist_ok=True)

        # Define class mapping {value: name}
        self.class_map = {
            1: "Urban",
            2: "Water",
            3: "Wetland",
            4: "Kalut",
            5: "Marshland",
            6: "Salty_Land",
            7: "Clay",
            8: "Forest",
            9: "Outcrop",
            10: "Uncovered_Plain",
            11: "Sand",
            12: "Farm_Land",
            13: "Range_Land"
        }

    def load_shapefile(self):
        """Load and process the shapefile for use in Earth Engine."""
        gdf = gpd.read_file(self.shapefile_path)
        gdf = gdf.to_crs(epsg=4326)  # Ensure CRS is WGS84

        # Combine all geometries using unary_union
        geometry = gdf.geometry.unary_union

        if geometry.geom_type == 'Polygon':
            coords = list(geometry.exterior.coords)
        elif geometry.geom_type == 'MultiPolygon':
            coords = list(geometry.geoms[0].exterior.coords)
        else:
            raise ValueError("Unsupported geometry type in shapefile")

        return ee.Geometry.Polygon(coords)

    def get_land_cover_labels(self):
        """Retrieve the classification labels from the dataset."""
        dataset = ee.Image('KNTU/LiDARLab/IranLandCover/V1')
        return dataset.clip(self.aoi).select(['classification'])

    def export_class_as_geojson(self, class_value, class_name):
        """
        Vectorize all pixels == class_value and export as a GeoJSON file.
        """
        label_image = self.get_land_cover_labels()

        # 1) Create a mask for pixels == class_value
        class_mask = label_image.eq(class_value).selfMask()

        # 2) Vectorize the masked image
        vectors = class_mask.reduceToVectors(
            geometry=self.aoi,
            crs='EPSG:4326',
            scale=30,
            geometryType='polygon',
            labelProperty='pixel_val',
            bestEffort=True,
            maxPixels=1e13
        )

        # Add class value and class name to each feature
        vectors = vectors.map(
            lambda f: f.set({"class_val": class_value, "class_name": class_name})
        )

        # 3) Export as GeoJSON locally
        out_filename = f"Iran_LandCover_{class_name}.geojson"
        out_path = os.path.join(self.output_folder, out_filename)

        print(f"Exporting class {class_value} ({class_name}) to {out_path}...")

        # Use geemap's ee_export_vector to save locally in GeoJSON format
        geemap.ee_export_vector(
            ee_object=vectors,
            filename=out_path
        )

        print(f"Finished exporting class {class_value} ({class_name}) to {out_path}.")

    def run(self):
        """
        Run the label download process.
        """
        print("Exporting each class from the land cover classification as GeoJSON...")

        for class_value, class_name in self.class_map.items():
            self.export_class_as_geojson(class_value, class_name)

        print("All classes have been exported.")
