import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def add_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator

@add_method(pd.DataFrame)
def to_gdf(self, lat_col='latitude', lon_col='longitude', crs=4326):
        # Ensure the latitude and longitude columns are present
    if lat_col not in self.columns or lon_col not in self.columns:
        raise ValueError(f"DataFrame must contain '{lat_col}' and '{lon_col}' columns")
    
    geometry = [Point(xy) for xy in zip(self[lon_col], self[lat_col])]
    gdf = gpd.GeoDataFrame(self, geometry=geometry)
    gdf = gdf.set_crs(crs)

    return gdf