'''
Created on Aug 23, 2022

@author: nejat
'''

import os 
import pandas as pd
import numpy as np

from src.geocoding.relative_spatial_entity import idenitfy_relative_spatial_cardinal

from iso3166 import countries

import nltk
nltk.download('stopwords')

def build_spatial_entity_lookup_table(data_folder):

  gadm1_filepath = os.path.join(data_folder, "map_shapefiles", "world", "gadm", "gadm36_1.csv")
  df_gadm1 = pd.read_csv(gadm1_filepath, sep=";", keep_default_na=False)
  
  gadm2_filepath = os.path.join(data_folder, "map_shapefiles", "world", "gadm", "gadm36_2.csv")
  df_gadm2 = pd.read_csv(gadm2_filepath, sep=";", keep_default_na=False)
  
  gadm3_filepath = os.path.join(data_folder, "map_shapefiles", "world", "gadm", "gadm36_3.csv")
  df_gadm3 = pd.read_csv(gadm3_filepath, sep=";", keep_default_na=False)
  
  lookup_dict = {}
  country_to_entites_dict = {}
  ambiguity_list = []
            
  # ========================================
  
  for index, row in df_gadm1.iterrows():
    country_code   = row["GID_0"]
    country_name = row["NAME_0"]
    province_name = row["NAME_1"]
    # print(index, country_code, country_name)
    
    if country_code in countries: # we process a country code, which is known >> not known example: XNC Northern Cyprus
      province_other_names = []
      province_other_names_str = row["VARNAME_1"]
      if province_other_names_str != "":
        province_other_names = province_other_names_str.split("|")
        
      province_name_list = [province_name] + province_other_names
      for p_name in province_name_list:
        p_name = p_name.replace("Province", "")
        p_name = p_name.replace("Region", "")
        p_name = idenitfy_relative_spatial_cardinal(p_name)
        if p_name != "-1":
          if country_code not in country_to_entites_dict:
            country_to_entites_dict[country_code] = []
          country_to_entites_dict[country_code].append(p_name)
          if p_name in lookup_dict:
            if country_code != lookup_dict[p_name]:
              ambiguity_list.append(p_name)
              #print("in gadm1, at index: ", index, " ==> ", p_name, " (" + country_name + ")", " already exists !  => ", \
              #       lookup_dict[p_name], ", ", countries.get(lookup_dict[p_name]).name)
          else:
            lookup_dict[p_name] = country_code
            
        
  # ========================================
  
  for index, row in df_gadm2.iterrows():
    country_code   = row["GID_0"]
    country_name = row["NAME_0"]
    # print(index, country_code, country_name)
    
    # there are many ambigous country names in USA
    if country_code != "USA" and country_code in countries: # we process a country code, which is known >> not known example: XNC Northern Cyprus
      city_county_name = row["NAME_2"]
      city_county_other_names = []
      city_county_other_names_str = row["VARNAME_2"]
      if city_county_other_names_str != "":
        city_county_other_names = city_county_other_names_str.split("|")
    
      city_county_name_list = [city_county_name] + city_county_other_names
      for c_name in city_county_name_list:
        c_name = c_name.replace("Province", "")
        c_name = c_name.replace("Region", "")
        c_name = idenitfy_relative_spatial_cardinal(c_name)
        if c_name != "-1":
          if country_code not in country_to_entites_dict:
            country_to_entites_dict[country_code] = []
          country_to_entites_dict[country_code].append(c_name)
          if c_name in lookup_dict:
            if country_code != lookup_dict[c_name]:
              ambiguity_list.append(c_name)
              #print("in gadm2, at index: ", index, " ==> ", c_name, " (" + country_name + ")", " already exists !  => ", \
              #       lookup_dict[c_name], ", ", countries.get(lookup_dict[c_name]).name)
          else:
            lookup_dict[c_name] = country_code
          
          
  # ========================================
  
  for index, row in df_gadm3.iterrows():
    country_code   = row["GID_0"]
    country_name = row["NAME_0"]
    # print(index, country_code, country_name)
    
    # there are many ambigous country names in USA  
    if country_code != "USA" and country_code in countries: # we process a country code, which is known >> not known example: XNC Northern Cyprus
      city_county_name = row["NAME_3"]
      city_county_other_names = []
      city_county_other_names_str = row["VARNAME_3"]
      if city_county_other_names_str != "":
        city_county_other_names = city_county_other_names_str.split("|")
    
      city_county_name_list = [city_county_name] + city_county_other_names
      for c_name in city_county_name_list:
        c_name = c_name.replace("Province", "")
        c_name = c_name.replace("Region", "")
        c_name = idenitfy_relative_spatial_cardinal(c_name)
        if c_name != "-1":
          if country_code not in country_to_entites_dict:
            country_to_entites_dict[country_code] = []
          country_to_entites_dict[country_code].append(c_name)
          if c_name in lookup_dict:
            if country_code != lookup_dict[c_name]:
              ambiguity_list.append(c_name)
              print("in gadm3, at index: ", index, " ==> ", c_name, " (" + country_name + ")", " already exists !  => ", \
                     lookup_dict[c_name], ", ", countries.get(lookup_dict[c_name]).name)
          else:
            lookup_dict[c_name] = country_code
    
  # =================================
  # post-processing
  print("ambiguity: ", len(np.unique(ambiguity_list)))  
  print("lookup: ", len(lookup_dict.keys())) 
  
  for ambiguity_name in ambiguity_list:
    lookup_dict[ambiguity_name] = "AMBIGUITY"
  
  return lookup_dict, country_to_entites_dict



  
if __name__ == '__main__':
  data_folder = "/home/nejat/eclipse/github/epidnews2event/data"
  build_spatial_entity_lookup_table(data_folder)
  
       
    