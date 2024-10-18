'''
Created on Nov 16, 2021

@author: nejat
'''

import os
import json

import pandas as pd

import src.consts as consts

import csv
import src.event.location as location

from src.util_gis import retrieve_continent_from_country_code

from src.geocoding.geocode import geocode_raw_with_geonames
from src.geocoding.enhance_geonames_data_with_hierarchy import get_loc_hierarchy
    
from iso3166 import countries







def update_geocoding_results_in_DB(geonames_results_in_batch_filepath, text):
  print("in update_geocoding_results_in_DB()")
  df_batch_geonames_results = pd.read_csv(geonames_results_in_batch_filepath, sep=";", keep_default_na=False)
  results = {} # each key corresponds to a column in the future file
  for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST):
    results[str(i)] = []
  results["text"] = []
  locs = geocode_raw_with_geonames(text)
  #print("locs:", locs)
  if locs is None: # len(locs)>0:
    locs = []
    print("NOT FOUND for: ", text)
    
  for i in range(len(locs)):
    results[str(i)].append(json.dumps(locs[i], ensure_ascii=False))
  for i in range(len(locs), consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST):
    results[str(i)].append(json.dumps("-1"))
  results["text"].append(text)
  df_curr = pd.DataFrame(results)
  df_batch_geonames_results = df_batch_geonames_results.append(df_curr, ignore_index=True)
  df_batch_geonames_results.to_csv(geonames_results_in_batch_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
  df_batch_geonames_results = pd.read_csv(geonames_results_in_batch_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_geonames_results = dict(zip(df_batch_geonames_results["text"], df_batch_geonames_results.index))
  return(df_batch_geonames_results, inverted_indices_for_df_batch_geonames_results)



def update_geonames_hierarhcy_results_in_DB(out_folder, geonameId):
  print(out_folder)
  geonames_hierarchy_filepath = os.path.join(out_folder, \
                          consts.GEONAMES_HIERARCHY_INFO_FILENAME + "." + consts.FILE_FORMAT_CSV)
  cols_geonames_hierarcy = [consts.COL_GEONAMES_ID, "hierarchy_name", "hierarchy_geoname_id"] 
  df_geonames_hierarchy = pd.read_csv(geonames_hierarchy_filepath, usecols=cols_geonames_hierarcy, sep=";", keep_default_na=False)
  
  hierarchy_result = get_loc_hierarchy(geonameId)
  df_curr = pd.DataFrame({consts.COL_GEONAMES_ID: [geonameId], \
                          "hierarchy_name": [str(hierarchy_result["name"])], \
                          "hierarchy_geoname_id": [str(hierarchy_result["geoname_id"])]
                        })
  df_geonames_hierarchy = df_geonames_hierarchy.append(df_curr, ignore_index=True)

  df_geonames_hierarchy.to_csv(geonames_hierarchy_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
  df_batch_geonames_results = pd.read_csv(geonames_hierarchy_filepath, sep=";", keep_default_na=False)
  geonames_hierarchy_ass = dict(zip(df_geonames_hierarchy[consts.COL_GEONAMES_ID], df_geonames_hierarchy["hierarchy_geoname_id"]))
  return(df_batch_geonames_results, geonames_hierarchy_ass)
  
  
  
# the aim of this method is just to populate 'geonames_results_in_batch' file for admin1 names
def search_geonames_for_admin1_loc_list(output_folder, df_events):
  geonames_results_in_batch_filepath = os.path.join(output_folder, "geonames_results_in_batch" + "." + consts.FILE_FORMAT_CSV)
  print(geonames_results_in_batch_filepath)
  if not os.path.exists(geonames_results_in_batch_filepath):
    # create an empty file 
    columns = [str(i) for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST)] + ["text"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(geonames_results_in_batch_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  #
  df_batch_geonames_results = pd.read_csv(geonames_results_in_batch_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_geonames_results = dict(zip(df_batch_geonames_results["text"], df_batch_geonames_results.index))


  filepath = os.path.join(output_folder, consts.GEONAMES_HIERARCHY_INFO_FILENAME + "." + consts.FILE_FORMAT_CSV)
  if not os.path.exists(filepath):
    # create an empty file 
    columns = [consts.COL_GEONAMES_ID, "hierarchy_name", "hierarchy_geoname_id"] 
    df = pd.DataFrame(columns=columns)
    df.to_csv(filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  #
  df_geonames_hierarchy = pd.read_csv(filepath, sep=";", keep_default_na=False)
  geonames_hierarchy_ass = dict(zip(df_geonames_hierarchy[consts.COL_GEONAMES_ID], df_geonames_hierarchy["hierarchy_geoname_id"]))

  # ----------------------------
  location_list = []
  for index, row in df_events.iterrows():
    print("----", str(index)+"/"+str(df_events.shape[0]), row["city"] + ", " + row["region"] + ", " + row["country"])
    region_name = row["region"] + ", " + row["country"]
    if row["region"] == "":
      region_name = row["country"]
    
    #print("---------------", index, place_name)
    res = {"geonamesId": "-1", "name": "-1", "country_code": "-1", "raw_data": "-1"}
    # ==================================================
    # retrieve the results from our database
    if region_name not in inverted_indices_for_df_batch_geonames_results:
      print("entered")
      df_batch, inverted_indices = update_geocoding_results_in_DB(geonames_results_in_batch_filepath, region_name)
      df_batch_geonames_results = df_batch
      inverted_indices_for_df_batch_geonames_results = inverted_indices





def retrieve_normalized_loc_list(output_folder, df_events):
  geonames_results_in_batch_filepath = os.path.join(output_folder, "geonames_results_in_batch" + "." + consts.FILE_FORMAT_CSV)
  #print(filepath)
  if not os.path.exists(geonames_results_in_batch_filepath):
    # create an empty file 
    columns = [str(i) for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST)] + ["text"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(geonames_results_in_batch_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  #
  df_batch_geonames_results = pd.read_csv(geonames_results_in_batch_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_geonames_results = dict(zip(df_batch_geonames_results["text"], df_batch_geonames_results.index))


  filepath = os.path.join(output_folder, consts.GEONAMES_HIERARCHY_INFO_FILENAME + "." + consts.FILE_FORMAT_CSV)
  #print(filepath)
  if not os.path.exists(filepath):
    # create an empty file 
    columns = [consts.COL_GEONAMES_ID, "hierarchy_name", "hierarchy_geoname_id"] 
    df = pd.DataFrame(columns=columns)
    df.to_csv(filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  #
  df_geonames_hierarchy = pd.read_csv(filepath, sep=";", keep_default_na=False)
  geonames_hierarchy_ass = dict(zip(df_geonames_hierarchy[consts.COL_GEONAMES_ID], df_geonames_hierarchy["hierarchy_geoname_id"]))

  # ----------------------------
  location_list = []
  for index, row in df_events.iterrows():
    #print("----", str(index)+"/"+str(df_events.shape[0]), row["city"] + ", " + row["region"] + ", " + row["country"])
    region_name = ""
    region_geoname = ""
    if row["region"] != "":
      region_name = row["region"] + ", " + row["country"]
      region_gindx = inverted_indices_for_df_batch_geonames_results[region_name]
      row_region_geonames_results = df_batch_geonames_results.iloc[region_gindx]
      geonames_results = [json.loads(row_region_geonames_results[str(i)]) for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST)] # we access the columns
      # =============================================================================
      region_geoname_json = geonames_results[0] # get the first result from geonames >> init, by default
      if region_geoname_json != "-1":
        region_geoname = region_geoname_json["adminName1"]
        for geoname_json_cand in geonames_results:
          if "fcode" in geoname_json_cand and geoname_json_cand["fcode"] == "ADM1":
            region_geoname = geoname_json_cand["adminName1"]
            region_geoname_json = geoname_json_cand
            break
      # =============================================================================
    # ----------------------------------------------------
    place_name = row["city"] + ", " + row["country"]
    if row["city"] == "" and region_name != "":
      place_name = region_name
    elif row["city"] == "" and region_name == "":
      place_name = row["country"]
    
    #print("---------------", index, place_name)
    res = {"geonamesId": "-1", "name": "-1", "country_code": "-1", "raw_data": "-1"}
    # ==================================================
    # retrieve the results from our database
    if place_name not in inverted_indices_for_df_batch_geonames_results:
      #print("girdi for:", place_name)
      df_batch, inverted_indices = update_geocoding_results_in_DB(geonames_results_in_batch_filepath, place_name)
      df_batch_geonames_results = df_batch
      inverted_indices_for_df_batch_geonames_results = inverted_indices
      
      
    gindx = inverted_indices_for_df_batch_geonames_results[place_name]
    row_geonames_results = df_batch_geonames_results.iloc[gindx]
    # print(row_geonames_results["0"])
    # {"adminCode1": "00", "lng": "34.75", "geonameId": 294640, "toponymName": "State of Israel", "countryId": "294640", "fcl": "A", "population": 8883800, "countryCode": "IL", "name": "Israel", "fclName": "country, state, region,...", "countryName": "Israel", "fcodeName": "independent political entity", "adminName1": "", "lat": "31.5", "fcode": "PCLI"}

    
    loc = location.Location(place_name, -1, -1, -1, -1, -1, -1, {})
    if row_geonames_results["0"] == "-1":
      print("Error in geonames with place name=",place_name)
    if row_geonames_results["0"] != "-1":
      #print(row_geonames_results["0"])
      geonames_results = [json.loads(row_geonames_results[str(i)]) for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST)] # we access the columns
      # --------------------------------------------
      # get the geoname results with the correct admin1 id
      geoname_json = geonames_results[0] # get the first result from geonames # default
      found = False # sterp 1: try to find the exact toponym name
      for geoname_json_cand in geonames_results:
        #print(row["city"], "==>", region_geoname, geoname_json_cand)
        if geoname_json_cand != "-1" and row["city"] != "" and geoname_json_cand["adminName1"] == region_geoname \
        and "fcode" in geoname_json_cand and "ADM" in geoname_json_cand["fcode"]: # if row["city"] == "", then we take the first json result
          #print("found", geoname_json_cand)
          if geoname_json_cand["toponymName"].lower() == row["city"].lower() or geoname_json_cand["name"].lower() == row["city"].lower():
            geoname_json = geoname_json_cand
            found = True
            #print("found2", geoname_json_cand)
            break
      if not found:
        for geoname_json_cand in geonames_results:
          #print(row["city"], "==>", region_geoname, geoname_json_cand)
          if geoname_json_cand != "-1" and row["city"] != "" and geoname_json_cand["adminName1"] == region_geoname \
          and "fcode" in geoname_json_cand and "PPL" in geoname_json_cand["fcode"]: # if row["city"] == "", then we take the first json result
            #print("found", geoname_json_cand)
            if geoname_json_cand["toponymName"].lower() == row["city"].lower() or geoname_json_cand["name"].lower() == row["city"].lower():
              geoname_json = geoname_json_cand
              found = True
              #print("found2", geoname_json_cand)
              break
      if not found: # step 2: get the first one without looking for the exact toponym name
        for geoname_json_cand in geonames_results:
          if geoname_json_cand != "-1" and row["city"] != "" and geoname_json_cand["adminName1"] == region_geoname \
          and "fcode" in geoname_json_cand and ("ADM" in geoname_json_cand["fcode"] or "PPL" in geoname_json_cand["fcode"]): # if row["city"] == "", then we take the first json result
              geoname_json = geoname_json_cand
              #print("found3", geoname_json_cand)
              break
      # --------------------------------------------
      if "countryCode" in geoname_json: # it is not a continent or sea
        geoname_id = geoname_json["geonameId"]
        # ---------------------------------------
        if geoname_id not in geonames_hierarchy_ass:
            df_geonames_hierarchy, geonames_hierarchy_ass = \
              update_geonames_hierarhcy_results_in_DB(output_folder, geoname_id)
        #print("DB ACCESS!!!!")
        hierarchy_data = eval(geonames_hierarchy_ass[geoname_id])
        # ---------------------------------------
        print(geoname_json.keys())
        print(geoname_json)
        name = geoname_json["toponymName"]
        lat = geoname_json["lat"]
        lng = geoname_json["lng"]
        fcode = "-1"
        if "fcode" in geoname_json:
          fcode = geoname_json["fcode"]
        country_code_alpha2 = geoname_json["countryCode"]
        country_code_alpha3 = countries.get(country_code_alpha2).alpha3
        continent = retrieve_continent_from_country_code(country_code_alpha3)
        loc = location.Location(name, geoname_id, geoname_json, lat, lng, country_code_alpha3, continent, hierarchy_data)
    
    location_list.append(loc)
  return(location_list)
  
  
  
  
  
