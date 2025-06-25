import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


df = pd.read_csv("path/to/s2_products_global_100.csv")
s = gpd.GeoSeries.from_wkt(df.true_footprint)
gdf = gpd.GeoDataFrame(data=df, geometry=s, crs=4326)
gdf.plot()
plt.savefig("footprints.png") # try this