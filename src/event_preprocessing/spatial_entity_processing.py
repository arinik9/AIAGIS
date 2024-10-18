'''
Created on Nov 16, 2021

@author: nejat
'''

import os
import pandas as pd
import time
import csv

from geopy.geocoders import Nominatim
from src.geocoding.geocode import geocode_raw_with_geonames
    
import src.consts as consts
from src.event_normalization.spatial_entity_normalization import update_geocoding_results_in_DB


def retrieve_loc_list_searchable_on_geonames(loc_name_list, geonames_results_in_batch_filepath):
  print(geonames_results_in_batch_filepath)
  if not os.path.exists(geonames_results_in_batch_filepath):
    print("CREATE")
    # create an empty file 
    columns = [str(i) for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST)] + ["text"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(geonames_results_in_batch_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  #
  df_batch_geonames_results = pd.read_csv(geonames_results_in_batch_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_geonames_results = dict(zip(df_batch_geonames_results["text"], df_batch_geonames_results.index))

  
  client_nominatim = Nominatim(user_agent="geoapi")
  
  processed_loc_list = []
  for i in range(len(loc_name_list)):
    loc_name = loc_name_list[i]
    
    print("----- ", i, loc_name)
    
    final_loc_name = "-1"
    if loc_name == "":
      final_loc_name = ""
    elif loc_name in inverted_indices_for_df_batch_geonames_results:
      final_loc_name = loc_name
    else:
      try:
        #res = client_nominatim.geocode(loc_name, exactly_one=True, addressdetails=True, language="en", timeout=20)
        res = geocode_raw_with_geonames(loc_name)
        if len(res)>0: # success
          final_loc_name = loc_name
          if loc_name not in inverted_indices_for_df_batch_geonames_results:
            df_batch, inverted_indices = update_geocoding_results_in_DB(geonames_results_in_batch_filepath, loc_name)
            df_batch_geonames_results = df_batch
            inverted_indices_for_df_batch_geonames_results = inverted_indices
        else:
          print("error with geonames with name=", loc_name)
          print("trying with nominatim")
          time.sleep(0.1)
          res = client_nominatim.geocode(loc_name, exactly_one=True, addressdetails=True, language="en", timeout=20)
          if res is not None and len(res)>0:
            loc = res[0]
            toponymName = loc.raw['display_name'].split(", ")[0]
            print("result of nominatim with name=", toponymName)
            final_loc_name = toponymName
      except:
        print("error in retrieve_loc_list_searchable_on_geonames() with name=", loc_name)
        pass
    processed_loc_list.append(final_loc_name)

  return processed_loc_list

