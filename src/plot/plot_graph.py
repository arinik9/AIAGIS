'''
Created on Nov 18, 2021

@author: nejat
'''

#from stellargraph import StellarGraph
import networkx as nx
import matplotlib.pyplot as plt

import geopandas as gpd

import os
import numpy as np

from shapely import wkt
import matplotlib.colors as mcolors
from shapely.geometry import Point, Polygon




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



  
  
  
  
def plot_graph_on_map(graph_filepath, map_data, output_filepath):
  graph = nx.read_graphml(graph_filepath)
  
  #map_data = gpd.read_file(map_data_shapefilepath, encoding = "utf-8")
  #map_data = map_data.to_crs(4326)
  
  if graph.number_of_nodes()>0 and graph.number_of_edges()>0:
    graph.remove_nodes_from(list(nx.isolates(graph))) 
    
    x_values = []
    y_values = []
    node_size_values = []
    for node in graph.nodes(data=True):
      x_values.append(node[1]["lng"])
      y_values.append(node[1]["lat"])
      node_size_values.append(node[1]["size"])
    centroids = np.column_stack((x_values, y_values))
    max_value = max(node_size_values)
    node_size_values = [x/max_value for x in node_size_values]
      
    edge_width_values = []
    for edge in graph.edges(data=True):
      edge_width_values.append(edge[2]["weight"])
    max_value = max(edge_width_values)
    edge_width_values = [(x/max_value)*2 for x in edge_width_values]
  
    # To plot with networkx, we need to merge the nodes back to
    # their positions in order to plot in networkx
    positions = dict(zip(graph.nodes, centroids))
    
    # plot with a nice basemap
    ax = map_data.plot(linewidth=0.25, edgecolor="grey", facecolor="lightblue")
    #ax.axis([-12, 45, 33, 66])
    ax.axis("off")
    # https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
    #nx.draw(graph, positions, ax=ax, node_size=node_size_values, width=edge_width_values, arrowsize=2, node_color="r")
    #plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #plt.gca().yaxis.set_major_locator(plt.NullLocator())
    nx.draw(graph, positions, ax=ax, node_size=node_size_values, width=edge_width_values, arrowsize=3, node_color="r", connectionstyle="arc3,rad=0.4")

    #plt.show()  
    plt.savefig(output_filepath, bbox_inches = 'tight')
    plt.close()
  else:
    print("graph is empty or only isoleted nodes !!")
    
    
def plot_graph_on_map2(graph_filepath, map_data, output_filepath):
  graph = nx.read_graphml(graph_filepath)
  #map_data = gpd.read_file(map_data_shapefilepath, encoding = "utf-8")
  #map_data = map_data.to_crs(4326)
  
  if graph.number_of_nodes()>0 and graph.number_of_edges()>0:
    x_values = []
    y_values = []
    node_size_values = []
    node_color_values = []

    for node in graph.nodes(data=True):
      x_values.append(node[1]["lng"])
      y_values.append(node[1]["lat"])
      node_size_values.append(node[1]["size"])
      if "color" not in node[1].keys():
        node_color_values.append("r")
      else:
        node_color_values.append(node[1]["color"])
    centroids = np.column_stack((x_values, y_values))
    max_value = max(node_size_values)
    node_size_values = [(x/max_value)*10 for x in node_size_values]
      
    edge_width_values = []
    for edge in graph.edges(data=True):
      edge_width_values.append(edge[2]["weight"])
    max_value = max(edge_width_values)
    #edge_width_values = [(x/max_value) for x in edge_width_values]
    edge_width_values = [w**(1/2) for w in edge_width_values] # sqrt transformation for better visualization

    edge_colors = []
    for edge in graph.edges(data=True):
      if "color" in edge[2]:
        edge_colors.append(edge[2]["color"])
      else:
        edge_colors.append("black")


    # To plot with networkx, we need to merge the nodes back to
    # their positions in order to plot in networkx
    positions = dict(zip(graph.nodes, centroids))
    
    fig, ax = plt.subplots(1,1, figsize=(4.5*2, 6), tight_layout = True)
    #map_data = reduce_polygon_memory(map_data)

    # plot with a nice basemap
    ax = map_data.plot(linewidth=0.1, edgecolor="grey", facecolor="lightblue", ax = ax, alpha = 0.5)
    # values = ["['pacific americas']", "['mississippi americas']", "['atlantic americas']", "['east atlantic']",
    #           "['black sea mediterranean']", "['east africa - west asia']", "['central asia']",
    #           "['east asian - australasian']", "[]"]
    # TABLEAU_COLORS_hex = [mcolors.rgb2hex(color) for name, color in mcolors.TABLEAU_COLORS.items()]
    #
    # for i in range(len(values)):
    #   value = values[i]
    #   color = TABLEAU_COLORS_hex[i]
    #   df_sub = map_data[map_data["flyway_inf"] == value]
    #   df_sub.plot(ax=ax, facecolor=color, linewidth=0.1, edgecolor='gray')


    country_shapefilepath = "/Users/narinik/Mirror/workspace/MulNetFer/in/map_shapefiles/world/ne_10m_admin_0_countries/ne_10m_admin_0_countries2_fixed_geo.shp"
    imd_country = gpd.read_file(country_shapefilepath, encoding = "utf-8")
    imd_country = imd_country.to_crs(3857)
    # if "name_engli" in imd_country.columns:
    #   imd_country = imd_country.rename(index=str, columns={"name_engli":"COUNTRY"})
    # if "ADMIN" in imd_country.columns:
    #   imd_country = imd_country.rename(index=str, columns={"ADMIN":"COUNTRY"})
    # if "iso" in imd_country.columns:
    #   imd_country = imd_country.rename(index=str, columns={"iso":"CNTR_CODE"})
    # if "ISO_A2" in imd_country.columns:
    #   imd_country = imd_country.rename(index=str, columns={"ISO_A2":"CNTR_CODE"})


    imd_country.plot(ax=ax, facecolor='none', linewidth=0.1, edgecolor='black')

    # add/annotate region names at the centroid point of the US regions
    display_country_code = True
    country_code_fontsize = 10
    if display_country_code:
      for idx, row in imd_country.iterrows():
        geometry = row['geometry']
        if not isinstance(geometry, Polygon):
          geometry = max(geometry.geoms, key=lambda a: a.area)
        country_codes = None
        if 'CNTR_CODE' in imd_country.columns:
          country_codes = row['CNTR_CODE']
        elif 'ISO_A2_EH' in imd_country.columns:
          country_codes = row['ISO_A2_EH']
        plt.annotate(text=country_codes, xy=geometry.centroid.coords[0], horizontalalignment='center', color="orange", fontsize=country_code_fontsize)

    # bird_flyways_shapefilepath = "/Users/narinik/Mirror/workspace/MulNetFer/in/external_data/bird_flyways/bird_flyways.shp"
    # imd_flyways = gpd.read_file(bird_flyways_shapefilepath, encoding="utf-8")
    # imd_flyways = imd_flyways.to_crs(4326)
    # imd_flyways.plot(ax=ax, facecolor='red', linewidth=0.1, edgecolor='red', alpha = 0.5)

    #ax.axis([-33, 150, -3, 70]) # europe, africa, asia
    ax.axis([-5, 32, 40, 64]) # europe
    #ax.axis([50, 150, 10, 60])  # asia
    #ax.axis("off")
    # https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
    nx.draw(graph, positions, ax=ax, node_size=node_size_values, width=edge_width_values, arrowsize=6, node_color=node_color_values, edge_color=edge_colors, connectionstyle="arc3,rad=0.4")
    #plt.show()
    #plt.savefig(output_filepath, dpi=500) >> If it is PNG
    fig.savefig(output_filepath)
    plt.close()
  else:
    print("graph is empty or only isolated nodes !!")

