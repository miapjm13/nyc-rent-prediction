import pandas as pd
import geopandas as gpd
from shapely import wkt
from math import radians, sin, cos, sqrt, atan2

from config import DATA_RAW, DATA_PROCESSED

NTA_BOUNDARIES_PATH = DATA_RAW.parent / "nta_boundaries.csv"
OUT_PATH = DATA_PROCESSED / "nta_spatial_features.csv"

MIDTOWN_LAT, MIDTOWN_LON = 40.7549, -73.9840


def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def main():
    nta_boundaries = pd.read_csv(NTA_BOUNDARIES_PATH)
    nta_boundaries["geometry"] = nta_boundaries["the_geom"].apply(wkt.loads)
    nta_gdf = gpd.GeoDataFrame(
        nta_boundaries, geometry="geometry", crs="EPSG:4326")

    nta_gdf = nta_gdf.to_crs(epsg=2263)
    nta_gdf["centroid"] = nta_gdf.geometry.centroid

    centroid_wgs84 = nta_gdf["centroid"].to_crs(epsg=4326)
    nta_gdf["centroid_lon"] = centroid_wgs84.x
    nta_gdf["centroid_lat"] = centroid_wgs84.y

    nta_gdf["dist_to_core_miles"] = nta_gdf.apply(
        lambda r: haversine(r["centroid_lat"],
                            r["centroid_lon"], MIDTOWN_LAT, MIDTOWN_LON),
        axis=1
    )

    nta_gdf["nta2020"] = nta_gdf["NTA2020"].str.upper()

    out = nta_gdf[["nta2020", "dist_to_core_miles"]]
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved spatial features: {OUT_PATH}")
    print(out.head())


if __name__ == "__main__":
    main()
