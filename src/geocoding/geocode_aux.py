'''
Created on Oct 23, 2023

@author: nejat
'''

from geopy.geocoders import Nominatim, GeoNames, ArcGIS
import geocoder # >>> https://geocoder.readthedocs.io/providers/GeoNames.html

import time
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderQueryError, GeocoderQuotaExceeded
import random
from iso3166 import countries
import os
import pandas as pd
import src.consts as consts
import csv
import json


def update_geocoding_results_in_DB_aux(geocoding_results_in_batch_filepath, loc_text, geocoding_result_list):
  print("in update_geocoding_results_in_DB()")
  df_batch_geocoding_results = pd.read_csv(geocoding_results_in_batch_filepath, sep=";", keep_default_na=False)
  results = {} # each key corresponds to a column in the future file
  for i in range(consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST):
    results[str(i)] = []
  results["text"] = []
  #
  locs = geocoding_result_list
  #print("locs:", locs)
  if locs is None: # len(locs)>0:
    locs = []
    print("NOT FOUND for: ", loc_text)
    
  for i in range(len(locs)):
    results[str(i)].append(json.dumps(locs[i], ensure_ascii=False))
  for i in range(len(locs), consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST):
    results[str(i)].append(json.dumps("-1"))
  results["text"].append(loc_text)
  df_curr = pd.DataFrame(results)
  df_batch_geocoding_results = pd.concat([df_batch_geocoding_results, df_curr], ignore_index=True)

  df_batch_geocoding_results.to_csv(geocoding_results_in_batch_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
  df_batch_geocoding_results = pd.read_csv(geocoding_results_in_batch_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_geonames_results = dict(zip(df_batch_geocoding_results["text"], df_batch_geocoding_results.index))
  return(df_batch_geocoding_results, inverted_indices_for_df_batch_geonames_results)


  
# Some spatial entity texts are very detailed information, and this makes the geocoding task difficult
#  That is why, if we do not succeed in geocding the original text, we split into a numerous smaller texts, see an example below
# example: "India: Dey Para,Nadia,WB"
# alternative_spatial_entity_list = ["India: Dey Para", "India: Nadia", "India: WB"]
def geocode_with_arcgis_aux(spatial_entity_text, alternative_spatial_entity_list, country_bias, arcgis_results_in_batch_filepath):
  
  #{'address': 'Paris, Île-de-France', 'location': {'x': 2.361657337, 'y': 48.863697576}, 'score': 100, 
  # 'attributes': {'Loc_name': 'World', 'Status': 'T', 'Score': 100, 'Match_addr': 'Paris, Île-de-France', 'LongLabel': 'Paris, Île-de-France, FRA', 
  #                'ShortLabel': 'Paris', 'Addr_type': 'Locality', 'Type': 'City', 'PlaceName': 'Paris', 'Place_addr': 'Paris, Île-de-France',
  #                 'Phone': '', 'URL': '', 'Rank': 2.5, 'AddBldg': '', 'AddNum': '', 'AddNumFrom': '', 'AddNumTo': '', 'AddRange': '', 
  #                 'Side': '', 'StPreDir': '', 'StPreType': '', 'StName': '', 'StType': '', 'StDir': '', 'BldgType': '', 'BldgName': '', 
  #                 'LevelType': '', 'LevelName': '', 'UnitType': '', 'UnitName': '', 'SubAddr': '', 'StAddr': '', 'Block': '', 'Sector': '', 
  #                 'Nbrhd': '', 'District': '', 'City': 'Paris', 'MetroArea': '', 'Subregion': 'Paris', 'Region': 'Île-de-France', 'RegionAbbr': '', 
  #                 'Territory': '', 'Zone': '', 'Postal': '', 'PostalExt': '', 'Country': 'FRA', 'CntryName': 'France', 'LangCode': 'FRE', 
  #                 'Distance': 0, 'X': 2.361657337, 'Y': 48.863697576, 'DisplayX': 2.361657337, 'DisplayY': 48.863697576, 'Xmin': 2.278657337, 
  #                 'Xmax': 2.444657337, 'Ymin': 48.780697576, 'Ymax': 48.946697576, 'ExInfo': ''}, 
  # 'extent': {'xmin': 2.278657337, 'ymin': 48.780697576, 'xmax': 2.444657337, 'ymax': 48.946697576}}
  
  if not os.path.exists(arcgis_results_in_batch_filepath):
    # create an empty file 
    columns = ["text"] + [str(i) for i in range(consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST)]
    df = pd.DataFrame(columns=columns)
    df.to_csv(arcgis_results_in_batch_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  #
  df_batch_arcgis_results = pd.read_csv(arcgis_results_in_batch_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_arcgis_results = dict(zip(df_batch_arcgis_results["text"], df_batch_arcgis_results.index))
  
  relevant_locations = []
  
  if spatial_entity_text in inverted_indices_for_df_batch_arcgis_results:
    print("!!!!!! DB ACCESS for:", spatial_entity_text)
    gindx = inverted_indices_for_df_batch_arcgis_results[spatial_entity_text]
    row_locs = df_batch_arcgis_results.iloc[gindx]
    relevant_locations = [json.loads(row_locs[str(i)]) for i in range(consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST)] # we access the columns
    
  else:
    client1_arcgis = ArcGIS(username="arinik9", password="Na271992",\
                     referer="AAPK82fb6deaf02541588bf004728607e880z1SJBCIFjsz9QncRCrvnLpTJPRchMUEYUrVXSJo6OgZGCQvkS5EXeSOuWpTeJ_aF", user_agent="application")
    client2_arcgis = ArcGIS(username="dodorag226", password="Na428361Na", \
                     referer="AAPKdc46e77ca38e450b8723d35afb8355a6aI9_YSSuXT1RGiPVZOcXASEOc5HCS_mzwnEsst5-aGRuiH9Bo8J-VXyOTaQX8SRu", user_agent="application") #   >> dodorag226@yubua.com
    clients_arcgis = [client1_arcgis, client2_arcgis]
    random.shuffle(clients_arcgis)
  
    
    try:
      #print(spatial_entity_text, country_bias)
      client_arcgis = clients_arcgis[0] # get the first one after shuffling
  
      arcgis_locations = []
      for text in [spatial_entity_text]+alternative_spatial_entity_list:
        print("text for arcgis:", text, "country_bias:", country_bias)
        curr_arcgis_locations = client_arcgis.geocode(text, exactly_one=False, out_fields="*", timeout=10)
        if curr_arcgis_locations is not None:
          count = 0
          for i in range(len(curr_arcgis_locations)):
            curr_res = curr_arcgis_locations[i]
            arcgis_locations.append(curr_res)
            count += 1
            if count == 3:
              break

      #print("len(arcgis_locations):", len(arcgis_locations))
      #print([loc.raw["attributes"]["Country"] for loc in arcgis_locations if "Country" in loc.raw["attributes"]])
      for loc in arcgis_locations:
        # example: Gauteng in South Africa

        if loc != -1 and "Country" in loc.raw["attributes"]: # TODO: do we really need this: "loc != -1" ??
          country_code_alpha3 = loc.raw["attributes"]["Country"]
          if country_code_alpha3 == country_bias:
            res = prepare_arcgis_loc_data_aux(loc)
            relevant_locations.append(res)
    except:
      print("error in geocode_batch_with_arcgis() with name=", spatial_entity_text)
      pass

    if spatial_entity_text not in inverted_indices_for_df_batch_arcgis_results:  # update DB
      df_batch, inverted_indices = update_geocoding_results_in_DB_aux(arcgis_results_in_batch_filepath, spatial_entity_text, relevant_locations)
      df_batch_arcgis_results = df_batch
      inverted_indices_for_df_batch_arcgis_results = inverted_indices


  if len(relevant_locations) == 0: # for instance, when it is a continent
    relevant_locations = ["-1" for i in range(consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST)]

  return relevant_locations


def prepare_arcgis_loc_data_aux(loc):
  res = {"name": "-1", "country_name": "-1", "country_code": "-1", "region_name": "-1", "city_name": "-1",
         "raw_data": "-1"}

  res["raw_data"] = loc.raw

  if 'PlaceName' in loc.raw["attributes"]:
    toponymName = loc.raw["attributes"]['PlaceName'].split(", ")[0]
    res["name"] = toponymName
  if 'Country' in loc.raw["attributes"]:
    country_code_alpha3 = loc.raw["attributes"]["Country"]
    # res["country_code"] = country_code_alpha3
    res["country_code"] = countries.get(country_code_alpha3).alpha2
  if 'CntryName' in loc.raw["attributes"]:
    country_name = loc.raw["attributes"]['CntryName']
    res["country_name"] = country_name
  if 'Region' in loc.raw["attributes"]:
    region_name = loc.raw["attributes"]['Region']
    res["region_name"] = region_name
  #if 'City' in loc.raw["attributes"]:
  #  city_name = loc.raw["attributes"]['City']
  #  res["city_name"] = city_name
  if 'Type' in loc.raw["attributes"]:
    type = loc.raw["attributes"]['Type']
    if type == "City" or type == "County":
      res["city_name"] = res["name"]

  return res


def geocode_batch_with_arcgis_aux(spatial_entity_list, alternative_spatial_entity_list_of_list, country_bias_list, arcgis_results_in_batch_filepath):
  result_list = []
  tot = len(spatial_entity_list)
  for i in range(len(spatial_entity_list)):
    print("--- i:",i, "/", tot)
    spatial_entity_text = spatial_entity_list[i]
    alternative_spatial_entity_list = alternative_spatial_entity_list_of_list[i]
    country_bias = country_bias_list[i]
    relevant_locations = geocode_with_arcgis_aux(spatial_entity_text, alternative_spatial_entity_list, country_bias, arcgis_results_in_batch_filepath)
    first_relevant_location = relevant_locations[0]
    result_list.append(first_relevant_location)
  return result_list


# -------------------------------------------------------------------------------------------------


# Some spatial entity texts are very detailed information, and this makes the geocoding task difficult
#  That is why, if we do not succeed in geocoding the original text, we split into a numerous smaller texts, see an example below
# example: "India: Dey Para,Nadia,WB"
# alternative_spatial_entity_list = ["India: Dey Para", "India: Nadia", "India: WB"]
def geocode_with_nominatim_aux(spatial_entity_text, alternative_spatial_entity_list, country_bias, nominatim_results_in_batch_filepath):
  country_bias = country_bias.lower()

  # {'place_id': 83293737, 'licence': 'Data © OpenStreetMap contributors, ODbL 1.0. http://osm.org/copyright',
  # 'osm_type': 'relation', 'osm_id': 71525, 'lat': '48.8534951', 'lon': '2.3483915', 'class': 'boundary', 'type': 'administrative',
  # 'place_rank': 12, 'importance': 0.8317101715588673, 'addresstype': 'city', 'name': 'Paris',
  # 'display_name': 'Paris, Ile-de-France, Metropolitan France, France',
  # 'address': {'city': 'Paris', 'ISO3166-2-lvl6': 'FR-75', 'state': 'Ile-de-France', 'ISO3166-2-lvl4': 'FR-IDF', 'region': 'Metropolitan France',
  #             'country': 'France', 'country_code': 'fr'}, 'boundingbox': ['48.8155755', '48.9021560', '2.2241220', '2.4697602']}


  if not os.path.exists(nominatim_results_in_batch_filepath):
    # create an empty file
    columns = ["text"] + [str(i) for i in range(consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST)]
    df = pd.DataFrame(columns=columns)
    df.to_csv(nominatim_results_in_batch_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  #
  df_batch_nominatim_results = pd.read_csv(nominatim_results_in_batch_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_nominatim_results = dict(zip(df_batch_nominatim_results["text"], df_batch_nominatim_results.index))

  relevant_locations = []

  if spatial_entity_text in inverted_indices_for_df_batch_nominatim_results:
    print("!!!!!! DB ACCESS for:", spatial_entity_text)
    gindx = inverted_indices_for_df_batch_nominatim_results[spatial_entity_text]
    row_locs = df_batch_nominatim_results.iloc[gindx]
    relevant_locations = [json.loads(row_locs[str(i)]) for i in range(consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST)] # we access the columns

  else:
    client_nominatim = Nominatim(user_agent="geoapi")

    try:
      print(spatial_entity_text, country_bias.upper())

      # country_bias must be an alpha2 code and lowercase
      nominatim_locations = []
      for text in [spatial_entity_text]+alternative_spatial_entity_list:
        print("text for nominatim:", text, "country_bias:", country_bias)
        curr_nominatim_locations = client_nominatim.geocode(text, country_codes=country_bias, exactly_one=False, addressdetails=True, language="en", timeout=20)
        if curr_nominatim_locations is not None:
          count = 0
          for i in range(len(curr_nominatim_locations)):
            curr_res = curr_nominatim_locations[i]
            if curr_res.raw["addresstype"] in ["city", "country", "state", "settlement", "town"]:
              nominatim_locations.append(curr_res)
              count += 1
              if count == 3:
                break

      for loc in nominatim_locations:
        # example: Gauteng in South Africa
        if loc != -1 and "country_code" in loc.raw["address"]: # TODO: do we really need this: "loc != -1" ??
          res = prepare_nominatim_loc_data(loc)
          relevant_locations.append(res)
    except:
      print("error in geocode_batch_with_nominatim() with name=", spatial_entity_text)
      pass

    if spatial_entity_text not in inverted_indices_for_df_batch_nominatim_results:  # update DB
      df_batch, inverted_indices = update_geocoding_results_in_DB_aux(nominatim_results_in_batch_filepath, spatial_entity_text, relevant_locations)
      df_batch_nominatim_results = df_batch
      inverted_indices_for_df_batch_nominatim_results = inverted_indices

  if len(relevant_locations) == 0: # for instance, when it is a continent
    relevant_locations = ["-1" for i in range(consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST)]

  return relevant_locations


def prepare_nominatim_loc_data_aux(loc):
  res = {"name": "-1", "country_name": "-1", "country_code": "-1", "region_name": "-1", "city_name": "-1",
         "raw_data": "-1"}

  res["raw_data"] = loc.raw
  country_codes_alpha2 = loc.raw["address"]["country_code"]
  res["country_code"] = country_codes_alpha2.upper()


  if 'display_name' in loc.raw:
    print("!! GIRDI")
    toponymName = loc.raw['display_name'].split(", ")[0]
    res["name"] = toponymName
  if 'country' in loc.raw["address"]:
    country_name = loc.raw["address"]['country']
    res["country_name"] = country_name
  #
  if 'state' in loc.raw["address"]:
    region_name = loc.raw["address"]['state']
    res["region_name"] = region_name
  elif 'province' in loc.raw["address"]:
    region_name = loc.raw["address"]['province']
    res["region_name"] = region_name
  elif 'region' in loc.raw["address"]:
    region_name = loc.raw["address"]['region']
    res["region_name"] = region_name
  #
  if 'district' in loc.raw["address"]:
    city_name = loc.raw["address"]['district']
    res["city_name"] = city_name
  elif 'town' in loc.raw["address"]:
    city_name = loc.raw["address"]['town']
    res["city_name"] = city_name
  elif 'city' in loc.raw["address"]:
    city_name = loc.raw["address"]['city']
    res["city_name"] = city_name
  elif 'county' in loc.raw["address"]:
    city_name = loc.raw["address"]['county']
    res["city_name"] = city_name


  return res


def geocode_batch_with_nominatim_aux(spatial_entity_list, alternative_spatial_entity_list_of_list, country_bias_list, nominatim_results_in_batch_filepath):
  result_list = []
  tot = len(spatial_entity_list)
  for i in range(len(spatial_entity_list)):
    print("--- i:",i, "/", tot)
    spatial_entity_text = spatial_entity_list[i]
    alternative_spatial_entity_list = alternative_spatial_entity_list_of_list[i]
    country_bias = country_bias_list[i]
    relevant_locations = geocode_with_nominatim_aux(spatial_entity_text, alternative_spatial_entity_list, country_bias, nominatim_results_in_batch_filepath)
    first_relevant_location = relevant_locations[0]
    result_list.append(first_relevant_location)
  return result_list



# ----------------------------------------------------------------------------------------------------------------------------------------------


# Some spatial entity texts are very detailed information, and this makes the geocoding task difficult
#  That is why, if we do not succeed in geocding the original text, we split into a numerous smaller texts, see an example below
# example: "India: Dey Para,Nadia,WB"
# alternative_spatial_entity_list = ["India: Dey Para", "India: Nadia", "India: WB"]
def geocode_with_geonames_aux(spatial_entity_text, alternative_spatial_entity_list, country_bias, geonames_results_in_batch_filepath):

  if not os.path.exists(geonames_results_in_batch_filepath):
    # create an empty file
    columns = ["text"] + [str(i) for i in range(consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST)]
    df = pd.DataFrame(columns=columns)
    df.to_csv(geonames_results_in_batch_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  #
  df_batch_geonames_results = pd.read_csv(geonames_results_in_batch_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_geonames_results = dict(zip(df_batch_geonames_results["text"], df_batch_geonames_results.index))

  relevant_locations = []

  if spatial_entity_text in inverted_indices_for_df_batch_geonames_results:
    print("!!!!!! DB ACCESS for:", spatial_entity_text)
    gindx = inverted_indices_for_df_batch_geonames_results[spatial_entity_text]
    row_locs = df_batch_geonames_results.iloc[gindx]
    relevant_locations = [json.loads(row_locs[str(i)]) for i in range(consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST)] # we access the columns

  else:
    geonames_api_username_list = [
                                  consts.GEONAMES_API_USERNAME1, consts.GEONAMES_API_USERNAME2, \
                                  consts.GEONAMES_API_USERNAME4, consts.GEONAMES_API_USERNAME3, \
                                  consts.GEONAMES_API_USERNAME5, consts.GEONAMES_API_USERNAME6, \
                                  consts.GEONAMES_API_USERNAME7, consts.GEONAMES_API_USERNAME8
                                  ]
    random.shuffle(geonames_api_username_list)


    try:
      print(spatial_entity_text, country_bias)

      geonames_api_username = geonames_api_username_list[0] # get the first one after shuffling
      print("---- account ", geonames_api_username)
      client_geonames = GeoNames(username=geonames_api_username)

      geonames_locations = []
      for text in [spatial_entity_text]+alternative_spatial_entity_list:
        print("text for geonames:", text)
        if ":" in text:
          text_without_country = text.split(":")[1].strip()
          text = text_without_country + ", " + text.split(":")[0].strip()
        curr_geonames_locations = client_geonames.geocode(text, exactly_one=False, timeout=20)
        if curr_geonames_locations is not None:
          count = 0
          for i in range(len(curr_geonames_locations)):
            curr_res = curr_geonames_locations[i]
            #print("fcode", curr_res.raw["fcode"])
            if curr_res.raw["fcode"] in ["PCLI", "PPL", "PPLA", "PPLA2", "PPLA3", "PPLC", "ADM1", "ADM1H", "ADM2", "ADM2H", "ADM3", "ADM3H"]:
              geonames_locations.append(curr_res)
              count+=1
              if count == 3:
                break

      #print("len(geonames_locations):" , len(geonames_locations))
      if len(geonames_locations) > consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST:
        geonames_locations = geonames_locations[:consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST]

      for loc in geonames_locations:
        # example: Gauteng in South Africa

        if loc != -1 and "countryCode" in loc.raw: # TODO: do we really need this: "loc != -1" ??
          country_code_alpha2 = loc.raw["countryCode"]
          if country_code_alpha2 == country_bias:
            res = prepare_geonames_loc_data(loc, country_code_alpha2, geonames_api_username)

            relevant_locations.append(res)
    except GeocoderTimedOut as e:
      print('GeocoderTimedOut')
    except GeocoderQueryError as e:
      print('GeocoderQueryError')
      print(e)
    except GeocoderQuotaExceeded as e:
      print('GeocoderQuotaExceeded')
      print(e)
    except GeocoderUnavailable as e:
      print('GeocoderUnavailable')
      print(e)
    except:
      print("error in geocode_batch_with_geonames() with name=", spatial_entity_text)
      pass
    
    if spatial_entity_text not in inverted_indices_for_df_batch_geonames_results:  # update DB
      #print(relevant_locations)
      df_batch, inverted_indices = update_geocoding_results_in_DB_aux(geonames_results_in_batch_filepath, spatial_entity_text, relevant_locations)
      df_batch_geonames_results = df_batch
      inverted_indices_for_df_batch_geonames_results = inverted_indices
  
  
  if len(relevant_locations) == 0: # for instance, when it is a continent
    print("not relevant_locations")
    relevant_locations = ["-1" for i in range(consts.MAX_NB_LOCS_PER_GEOCODING_REQUEST)]
    
  return relevant_locations
    
    
    

def prepare_geonames_loc_data_aux(loc, country_code_alpha2, geonames_api_username):
  res = {"name": "-1", "geoname_id": "-1", "country_name": "-1", "country_code": "-1", "region_name": "-1",
         "city_name": "-1", "hierarchy": "-1", "raw_data": "-1"}

  res["country_code"] = country_code_alpha2
  res["raw_data"] = loc.raw

  if 'toponymName' in loc.raw:
    toponymName = loc.raw['toponymName']
    res["name"] = toponymName
  if 'geonameId' in loc.raw:
    geonameId = loc.raw['geonameId']
    res["geoname_id"] = geonameId
  if 'countryName' in loc.raw:
    country_name = loc.raw['countryName']
    res["country_name"] = country_name
  if 'adminName1' in loc.raw:
    region_name = loc.raw['adminName1']
    res["region_name"] = region_name

  if res["geoname_id"] != "-1":
    try:
      res_hier = {"name": [], "geoname_id": []}
      g = geocoder.geonames(geonameId, method='hierarchy', key=geonames_api_username)
      if g.status == "OK":
        # print("status: ", g.status)
        for result in g:  # iterate the hierarchy from upper level to lower level
          if "countryName" in result.raw:
            res_hier["name"].append(result.address)
            res_hier["geoname_id"].append(result.geonames_id)
        res["hierarchy"] = res_hier

        if len(res_hier["name"])>2:
          res["city_name"] = res["name"]
    except:
      print("error in get_loc_hierarchy() with geoname_id=", geonameId)

  return res
    
 
def geocode_batch_with_geonames_aux(spatial_entity_list, alternative_spatial_entity_list_of_list, country_bias_list, geonames_results_in_batch_filepath):
  result_list = []
  tot = len(spatial_entity_list)
  for i in range(len(spatial_entity_list)):
    print("--- i:",i, "/", tot)
    spatial_entity_text = spatial_entity_list[i]
    alternative_spatial_entity_list = alternative_spatial_entity_list_of_list[i]
    country_bias = country_bias_list[i]
    relevant_locations = geocode_with_geonames_aux(spatial_entity_text, alternative_spatial_entity_list, country_bias, geonames_results_in_batch_filepath)
    first_relevant_location = relevant_locations[0]
    result_list.append(first_relevant_location)
  return result_list   
    
    
    
    
    
    
    
    
    
    
    