from torchgeo.datasets import Sentinel2
from torchgeo.samplers import GridGeoSampler
from pyproj import CRS, Transformer

"""This script demonstrates how using EPSG:3857, a gloabal mercator projection, makes Sentinel2 being able to use GridGeoSampler on tiles from different UTM zones.
However, this was not used to extra patches when making the .pt dataset!"""

dataset_native = Sentinel2(paths="/cluster/home/malovhoi/letmecooknow/dino/data", res=10)

# 2. Dataset in EPSG:3857
dataset_webmerc = Sentinel2(paths="/cluster/home/malovhoi/letmecooknow/dino/data", res=10, crs="EPSG:3857")

sampler_native = GridGeoSampler(dataset=dataset_native, size=256, stride=256)
sampler_webmerc = GridGeoSampler(dataset=dataset_webmerc, size=256, stride=256)

native_bbox = list(sampler_native)[0]
webmerc_bbox = list(sampler_webmerc)[0]

print("📦 Native CRS (likely UTM):")
print(native_bbox)

print("🌐 EPSG:3857:")
print(webmerc_bbox)

transformer = Transformer.from_crs(dataset_native.crs, "EPSG:3857", always_xy=True)

# Reproject native_bbox to EPSG:3857
minx, miny = transformer.transform(native_bbox.minx, native_bbox.miny)
maxx, maxy = transformer.transform(native_bbox.maxx, native_bbox.maxy)

print("🧭 Native bbox reprojected to EPSG:3857:")
print(f"minx={minx}, maxx={maxx}, miny={miny}, maxy={maxy}")