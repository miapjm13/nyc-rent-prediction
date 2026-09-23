"""
One-time script: converts the NTA boundaries CSV (WKT geometry) into a
GeoJSON file the Streamlit app can load directly.

Run this once, commit the output. The app reads the GeoJSON with plain
json.load() rather than geopandas, so geopandas/shapely never need to be
installed on whatever's actually running the deployed app.
"""

import pandas as pd
import geopandas as gpd
from shapely import wkt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES_PATH = ROOT / "data" / "raw" / "nta_boundaries.csv"
OUT_PATH = ROOT / "data" / "processed" / "nta_boundaries.geojson"


def main():
    boundaries = pd.read_csv(BOUNDARIES_PATH)
    boundaries["geometry"] = boundaries["the_geom"].apply(wkt.loads)
    nta_gdf = gpd.GeoDataFrame(boundaries, geometry="geometry", crs="EPSG:4326")

    # Simplify the polygons. full-resolution NTA boundaries are way more
    # detail than a small choropleth actually needs, and it keeps the file small.
    nta_gdf["geometry"] = nta_gdf.geometry.simplify(0.0001, preserve_topology=True)

    nta_gdf = nta_gdf[["NTA2020", "geometry"]]
    nta_gdf.to_file(OUT_PATH, driver="GeoJSON")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()