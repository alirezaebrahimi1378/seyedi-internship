from image_downloader import SentinelImageDownloader
from label_downloader import LandCoverLabelDownloader
import os


if __name__ == '__main__':
    dir = os.path.dirname(__file__)

    # Define Parameteres
    start_date = "2016-01-01"
    end_date = "2017-01-01"

    shapefile_path = os.path.join(dir, "ShapeFile", "TehranNorthEast.shp")

    output_sentinel_path = os.path.join(dir, "Sentinel2")
    output_label_path = os.path.join(dir, "Label")

    # Download Sentinel2 Images for the Desired Region
    downloader = SentinelImageDownloader(shapefile_path, start_date, end_date, output_sentinel_path)
    
    # Run the image download process
    print("Starting Sentinel-2 image download...")
    downloader.run()

    # Download Land Cover Labels
    label_downloader = LandCoverLabelDownloader(shapefile_path, output_label_path)
    print("Starting Land Cover Label download...")

    # This will export 13 GeoJSON files (classes 1..13)
    label_downloader.run()
    print("Done!")
