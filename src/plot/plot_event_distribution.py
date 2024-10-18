'''
Created on Apr 2, 2022

@author: nejat
'''

import os
#import seaborn as sns
import pandas as pd
#import pysal as ps
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from geopandas import GeoDataFrame
import mapclassify # pysla classification schemes >> discretization of numerical variables
#from shapely.geos import TopologicalError
import src.consts as consts
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import Normalize
from shapely import wkt






def reduce_polygon_memory(gdf):
  # to reduce the size of the output pdf file:
  # source: https://gis.stackexchange.com/questions/321518/rounding-coordinates-to-5-decimals-in-geopandas
  #simpledec = re.compile(r"\d*\.\d+")
  #def mround(match):
  #  return "{:.2f}".format(float(match.group()))
  #final_imd.geometry = final_imd.geometry.apply(lambda x: loads(re.sub(simpledec, mround, x.wkt)))
  new_polygons = []
  for index, row in gdf.iterrows():    
    geom = row['geometry']
    p = wkt.loads(wkt.dumps(geom, rounding_precision=2))
    new_polygons.append(p)
  gdf['geometry'] = new_polygons
  return gdf



def perform_event_distributions(df_events, imd: GeoDataFrame, column_name):
  imd_copy = imd.copy(deep=True)
  imd_copy[column_name] = 0
  imd_copy["gn_id"] = imd_copy["gn_id"].astype(int)
  geonameId2count = dict(zip(imd_copy["gn_id"], imd_copy[column_name]))
  for index, row in df_events.iterrows():
    geonameId = row["ADM1_geonameid"]
    if geonameId in geonameId2count:
      print("girdi")
      geonameId2count[geonameId] += 1
    else:
      print("error", geonameId)
    #try:
    #  imd_copy.loc[imd["gn_id"] == geonameId, 'event'] += 1
    #except TopologicalError:
    #  print("exception")
    #  pass
  #imd_copy.to_file(driver='ESRI Shapefile', filename=out_final_shapefilepath, encoding="utf-8")
  print(geonameId2count)
  imd_copy[column_name] = imd_copy["gn_id"].apply(lambda x: geonameId2count[x])
  return imd_copy




def plot_event_distribution(gdf_events, out_map_figure_filepath, country_shapefilepath, limits, \
                                       column_name, display_country_code):
  # https://matplotlib.org/2.0.2/users/colormaps.html
  imd_country = gpd.read_file(country_shapefilepath, encoding = "utf-8")
  imd_country = imd_country.to_crs(4326)

  gdf_events = reduce_polygon_memory(gdf_events)
  norm = Normalize(vmin=0.05, vmax=max(limits))
  width = 4.5*3 # default, for europe map
  height = 6
  country_code_fontsize = 10

  fig, ax = plt.subplots(figsize=(width, height), tight_layout = True)
  gdf_events_zero = gdf_events[gdf_events[column_name] < 0.00001]
  print(gdf_events_zero.shape)
  gdf_events_non_zero = gdf_events[gdf_events[column_name] > 0.00001]
  print(gdf_events_non_zero.shape)
  ax = gdf_events_non_zero.plot(ax=ax, column=column_name, colormap=plt.cm.Blues, norm=norm, legend=True, linewidth=0, legend_kwds={'fraction':0.03, 'pad':0.04})

  ax = gdf_events_zero.plot(ax=ax, column=column_name, color="white", linewidth=0, legend=False) # we exclude this part from the discretization of the colors
  #ax = final_imd_event_zero.plot(ax=ax, column=column_name, color="whitesmoke", linewidth=0, legend=False) # we exclude this part from the discretization of the colors
  _ = ax.axis('off')
  # highlight the US region borders in red
  imd_country.plot(ax=ax, facecolor='none', linewidth=0.3, edgecolor='grey')

  # add/annotate region names at the centroid point of the regions
  if display_country_code:
    for idx, row in imd_country.iterrows():
      geometry = row['geometry']
      if not isinstance(geometry, Polygon):
        geometry = max(geometry.geoms, key=lambda a: a.area)
      country_codes = None
      if 'CNTR_CODE' in imd_country.columns:
        country_codes = row['CNTR_CODE']
      elif 'ISO_A2' in imd_country.columns:
        country_codes = row['ISO_A2']
      plt.annotate(text=country_codes, xy=geometry.centroid.coords[0], horizontalalignment='center', color="orange", fontsize=country_code_fontsize)
  #plt.show()
  fig.savefig(out_map_figure_filepath, bbox_inches = 'tight')







