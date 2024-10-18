

from geopy.geocoders import Nominatim, GeoNames, ArcGIS
import time
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

import src.consts as consts
import random

from iso3166 import countries




def geocode_batch_with_nominatim(spatial_entity_list, country_bias_list):
  client_nominatim = Nominatim(user_agent="geoapi")
  
  result_list = []
  for i in range(len(spatial_entity_list)):
    #time.sleep(1)
    time.sleep(0.10)
    res = {"name": "-1", "country_code": "-1", "raw_data": "-1"}
    try:
      name = spatial_entity_list[i]
      country_bias = country_bias_list[i]
      if country_bias == "-1":
        country_bias=None
      else:
        country_bias = countries.get(country_bias).alpha2
      print(i, name, country_bias)
      nominatim_locations = client_nominatim.geocode(name, exactly_one=False, addressdetails=True, language="en", timeout=20)
      best_loc = None
      if nominatim_locations is None:
        nominatim_locations = []
      processed = False
      for loc in nominatim_locations:
        # example: Gauteng in South Africa
        if loc != -1 and "country_code" in loc.raw["address"]:
          toponymName = loc.raw['display_name'].split(", ")[0]
          if loc.raw["address"]["country_code"] == country_bias and toponymName == name:
            best_loc = loc
            processed = True
            break
      if not processed:
        for loc in nominatim_locations:
          if loc != -1 and "country_code" in loc.raw["address"]:
            if loc.raw["address"]["country_code"] == country_bias:
              best_loc = loc
              processed = True
              break
      if not processed and len(nominatim_locations)>0:
        best_loc = nominatim_locations[0] # by deafult
        
      if best_loc is not None:
        details = best_loc.raw["address"].copy()
        del details["country"]
        del details["country_code"]
        if len(details) != 0: # we do not want to get a country, if it is not a country, there will be some other attributes
          best_loc_name = best_loc.raw['display_name'].split(", ")[0]
          best_loc_country_code = best_loc.raw["address"]["country_code"].upper()
          best_loc_country_code = countries.get(best_loc_country_code).alpha3 
          res = {"name": best_loc_name, "country_code": best_loc_country_code, "raw_data": best_loc.raw}
    except:
      print("error in geocode_batch_with_nominatim() with name=", spatial_entity_list[i])
      pass
    result_list.append(res)

  return result_list


  
def geocode_batch_with_arcgis(spatial_entity_list, country_bias_list):
  client1_arcgis = ArcGIS(username="arinik9", password="Na271992", referer="AAPK82fb6deaf02541588bf004728607e880z1SJBCIFjsz9QncRCrvnLpTJPRchMUEYUrVXSJo6OgZGCQvkS5EXeSOuWpTeJ_aF", user_agent="application")
  client2_arcgis = ArcGIS(username="dodorag226", password="Na428361Na", referer="AAPKdc46e77ca38e450b8723d35afb8355a6aI9_YSSuXT1RGiPVZOcXASEOc5HCS_mzwnEsst5-aGRuiH9Bo8J-VXyOTaQX8SRu", user_agent="application") #   >> dodorag226@yubua.com
  clients_arcgis = [client1_arcgis, client2_arcgis]
  random.shuffle(clients_arcgis)

  result_list = []
  for i in range(len(spatial_entity_list)):
    res = {"name": "-1", "country_code": "-1", "raw_data": "-1"}
    try:
      name = spatial_entity_list[i]
      country_bias = country_bias_list[i]
      print(i, name, country_bias)
      
      arcgis_client_index = int(i/2000) # 500 query per account
      print("---- account index ", arcgis_client_index)
      client_arcgis = clients_arcgis[arcgis_client_index]
  
      arcgis_locations = client_arcgis.geocode(name, exactly_one=False, out_fields="*", timeout=10)
      if arcgis_locations is None:
        arcgis_locations = []
      best_loc = None
      processed = False
      for loc in arcgis_locations:
        # example: Gauteng in South Africa
        if loc != -1 and "Country" in loc.raw["attributes"]:
          country_codes_alpha3 = loc.raw["attributes"]["Country"]
          toponymName = loc.raw["attributes"]['PlaceName'].split(", ")[0]
          if country_codes_alpha3 == country_bias and toponymName == name:
            best_loc = loc
            processed = True
            break
      if not processed:
        for loc in arcgis_locations:
          if loc != -1 and "Country" in loc.raw["attributes"]:
            country_codes_alpha3 = loc.raw["attributes"]["Country"]
            if country_codes_alpha3 == country_bias:
              best_loc = loc
              processed = True
              break
      if not processed and len(arcgis_locations)>0:
        best_loc = arcgis_locations[0] # by deafult
        
      if best_loc is not None:
        best_loc_name = best_loc.raw["attributes"]['PlaceName'].split(", ")[0]
        best_loc_country_code = best_loc.raw["attributes"]["Country"]
        if best_loc.raw["attributes"]["Type"] != "Country": # TODO: it is a bug from arcgis, ir retuns the country name ..
          # ex of the arcgis bug: loc_name  = "Telangana, Nigeria" ==> Telangana is in India
          res = {"name": best_loc_name, "country_code": best_loc_country_code, "raw_data": best_loc.raw}
          
    except:
      print("error in geocode_batch_with_arcgis() with name=", spatial_entity_list[i])
      pass
    result_list.append(res)
    
  return result_list


def geocode_batch_with_geonames(spatial_entity_list, country_bias_list):

  geonames_api_username_list = [consts.GEONAMES_API_USERNAME, consts.GEONAMES_API_USERNAME2, \
                                consts.GEONAMES_API_USERNAME4, consts.GEONAMES_API_USERNAME3, \
                                consts.GEONAMES_API_USERNAME5, consts.GEONAMES_API_USERNAME6, \
                                consts.GEONAMES_API_USERNAME7,\
                                consts.GEONAMES_API_USERNAME8
                                ]
  random.shuffle(geonames_api_username_list)
    
  result_list = []
  for i in range(len(spatial_entity_list)):
    res = {"name": "-1", "country_code": "-1", "raw_data": "-1"}
    try:
      name = spatial_entity_list[i]
      country_bias = country_bias_list[i]
      if country_bias == "-1":
        country_bias=None
      else:
        country_bias = countries.get(country_bias).alpha2
      print(i, name, country_bias)
      
      geonames_username_index = int(i/500) # 500 query per account
      geonames_api_username = geonames_api_username_list[geonames_username_index]
      print("---- account ", geonames_api_username)
      client_geonames = GeoNames(username=geonames_api_username)
      geonames_locations = client_geonames.geocode(name, exactly_one=False, timeout=10)
      if geonames_locations is None:
        geonames_locations = []
      best_loc = None
      processed = False
      for loc in geonames_locations:
        # example: Gauteng in South Africa
        if loc != -1 and  "countryCode" in loc and "toponymName" in loc and loc["countryCode"] == country_bias and loc["toponymName"] == name:
          best_loc = loc
          processed = True
          break
      if not processed:
        for loc in geonames_locations:
          if loc != -1 and "countryCode" in loc and loc["countryCode"] in country_bias:
            best_loc = loc
            processed = True
            break
      if not processed and len(geonames_locations)>0:
        best_loc = geonames_locations[0] # by deafult
      
      if best_loc is not None:
        if best_loc.raw["geonameId"] != best_loc.raw["countryId"]: # we do not want to get a country
          print(best_loc.raw)
          best_loc_name = best_loc.raw["toponymName"]
          best_loc_country_code = best_loc.raw["countryCode"]
          print(best_loc_country_code)
          best_loc_country_code = countries.get(best_loc_country_code).alpha3 
          res = {"name": best_loc_name, "country_code": best_loc_country_code, "raw_data": best_loc.raw}
    except:
      print("error in geocode_batch_with_geonames() with name=", spatial_entity_list[i])
      pass
    result_list.append(res)
    
  return result_list



def geocode_with_geonames(spatial_entity_value, country_bias_list, db_locs=None, country_limit=None, only_country=False):
  spatial_entity_value = spatial_entity_value.strip()
  text = spatial_entity_value
  if country_limit is not None:
    text = spatial_entity_value + ", " + country_limit
    #print(text)
  process_country_bias = (len(country_bias_list) > 0)
  country_bias_list_alpha2 = [countries.get(country_code_alpha3).alpha2 for country_code_alpha3 in country_bias_list]
  geonames_api_username_list =  [
                                consts.GEONAMES_API_USERNAME, consts.GEONAMES_API_USERNAME2, \
                                consts.GEONAMES_API_USERNAME3, consts.GEONAMES_API_USERNAME4, \
                                consts.GEONAMES_API_USERNAME5, consts.GEONAMES_API_USERNAME6, \
                                consts.GEONAMES_API_USERNAME7, 
                                consts.GEONAMES_API_USERNAME8                             # consts.GEONAMES_API_USERNAME5, 
                                # consts.GEONAMES_API_USERNAME7, , 
                                # consts.GEONAMES_API_USERNAME3, consts.GEONAMES_API_USERNAME2, consts.GEONAMES_API_USERNAME4, consts.GEONAMES_API_USERNAME6
                                ]
  random.shuffle(geonames_api_username_list)
  #print(spatial_entity_value, country_bias_list)
  #
  res = {"geonamesId": "-1", "name": "-1", "country_code": "-1", "raw_data": "-1"}
  #try:
  geonames_username_index = random.randint(0, len(geonames_api_username_list)-1)
  geonames_api_username = geonames_api_username_list[geonames_username_index]
  
  client_geonames = GeoNames(username=geonames_api_username)
  geonames_locations = db_locs
  if db_locs is None:
    print("---- account ", geonames_api_username, "COST COST!!!!", text)
    geonames_locations = client_geonames.geocode(text, exactly_one=False, country_bias=None, timeout=10)
    if geonames_locations is None:
      geonames_locations = []
    geonames_locations = [loc.raw for loc in geonames_locations]
  #print(len(geonames_locations))
  if len(geonames_locations) > 0:
    # best_loc = geonames_locations[0] # by default: the first result
    best_loc = res
    processed = False
    if process_country_bias:
      for loc in geonames_locations:
        # example: Gauteng in South Africa
        if loc != -1 and  "countryCode" in loc and loc["countryCode"] in country_bias_list_alpha2:
          if ("toponymName" in loc and loc["toponymName"] == spatial_entity_value) \
           or ("name" in loc and loc["name"] == spatial_entity_value):
            best_loc = loc
            processed = True
            break
      if not processed:
        for loc in geonames_locations:
          if loc != -1 and "countryCode" in loc and loc["countryCode"] in country_bias_list_alpha2:
            best_loc = loc
            processed = True
            break
    else:
      if only_country:
        for loc in geonames_locations:
          if loc != -1 and "fcode" in loc and (loc["fcode"] == 'PCLI' or loc["fcode"] == 'PCLS'):
            best_loc = loc
            processed = True
            break
      else:
        if country_limit is not None:
          for loc in geonames_locations:
            if loc != -1 and "fcode" in loc and loc["fcode"] != 'PCLI':
              best_loc = loc
              processed = True
              break
        else:
          for loc in geonames_locations:
            if loc != -1:
              if ("toponymName" in loc and loc["toponymName"] == spatial_entity_value) \
                or ("name" in loc and loc["name"] == spatial_entity_value):
                best_loc = loc
                processed = True
                break
          if not processed:
            processed = True
            best_loc = geonames_locations[0]
    if processed and best_loc["geonameId"] != best_loc["countryId"]: # we do not want to get a country
      best_loc_country_code_alpha3 = countries.get(best_loc["countryCode"]).alpha3 
      res = {"geonamesId": best_loc["geonameId"], "name": best_loc["toponymName"], "country_code": best_loc_country_code_alpha3, "raw_data": best_loc}
  # except:
  #   print("error in geocode_batch_with_geonames() with name=", spatial_entity_value)
  #   pass
    
  return res



def geocode_raw_with_geonames(spatial_entity_value):
  spatial_entity_value = spatial_entity_value.strip()
  geonames_api_username_list = [
                                consts.GEONAMES_API_USERNAME5, consts.GEONAMES_API_USERNAME1, \
                                consts.GEONAMES_API_USERNAME3, \
                                consts.GEONAMES_API_USERNAME4, \
                                consts.GEONAMES_API_USERNAME8, \
                                consts.GEONAMES_API_USERNAME2 
                                ] #consts.GEONAMES_API_USERNAME5, consts.GEONAMES_API_USERNAME, consts.GEONAMES_API_USERNAME8
                              # consts.GEONAMES_API_USERNAME7, consts.GEONAMES_API_USERNAME3, \
                               # consts.GEONAMES_API_USERNAME4, consts.GEONAMES_API_USERNAME6, consts.GEONAMES_API_USERNAME2,
  random.shuffle(geonames_api_username_list)
  #try:
  geonames_username_index = random.randint(0, len(geonames_api_username_list)-1)
  geonames_api_username = geonames_api_username_list[geonames_username_index]
  print("---- account ", geonames_api_username)
  client_geonames = GeoNames(username=geonames_api_username)
  geonames_locations = []
  try:
    geonames_locations = client_geonames.geocode(spatial_entity_value, exactly_one=False, country_bias=None, timeout=10)
    if geonames_locations is None:
        geonames_locations = []
    geonames_locations = [g.raw for g in geonames_locations]
  except GeocoderTimedOut as e:
    print("Read timed out or max retries exceeded with this Geonames account:", geonames_api_username)
    pass # default, it is an empty list
  return geonames_locations
  # except:
  #   print("error in geocode_batch_with_geonames() with name=", spatial_entity_value)
  #   pass
  #  
  
