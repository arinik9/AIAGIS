'''
Created on Sep 1, 2022

@author: nejat
'''


from iso3166 import countries
import re
import pycountry_convert as pc
from geopy.geocoders import Nominatim
import time
import json
import pandas as pd
import csv

from src.geocoding.geocode import geocode_raw_with_geonames
import src.consts as consts


  
  
def retrieve_continent_from_country_code(country_code):
  continent_name = ""
  try:
    if len(country_code) == 3: # alpha3
      country_code = pc.country_alpha3_to_country_alpha2(country_code)
    continent_name = pc.country_alpha2_to_continent_code(country_code)
  except KeyError:
    print(country_code, "is a key error")
  return continent_name



def get_country_apolitical_name_from_alpha3_code(country_alpha3_code):
  country_name = ""
  if country_alpha3_code == "KOR":
    country_name = "South Korea"
  elif country_alpha3_code == "GBR":
    country_name = "United Kingdom"
  elif country_alpha3_code == "RUS":
    country_name = "Russia"
  elif country_alpha3_code == "IRN":
    country_name = "Iran"
  elif country_alpha3_code == "PRK":
    country_name = "North Korea"
  else:
    country_name = countries.get(country_alpha3_code).apolitical_name
  return country_name


# There are 2 possibilities:
#  1) we process the abbreviation
#  2) if convert_spatial_entity_abbreviation_into_name != "-1", on top of step 1, we also go a step further and converts the abbreviation into a name
#
# In any case, we treat four abbreviation cases:
# 1) if all the letters are in uppercase (from 2 to 4 letters)
# 2) the text with dots, e.g. U.K or U.S.A. etc.
# 3) the first letter is in uppercase and the second and third ones (from 2 to 3 letters) are in lowercase, e.g. Fla >> Floria, Ill >> Illinois
# 4) the text ended with a dot, e.g., Minn. >> Minnesota
def process_spatial_entity_abbreviation(text, country_code_alpha2="-1"):
  # The reason why we do not treat multi words, event though there can be an abbreviation
  # good example: LA country
  # bad example: St. Louis
  if text.lower() in consts.BAN_LIST_FOR_SPATIAL_ENTITY_EXTRACTION: # source: https://www.dictionary.com/e/dr-vs-phd-vs-md/
    return text
  
  init_text = text
  rgx = "^([A-Z][.])+" # e.g., turn "U.K." into "UK"
  if re.sub(rgx, "", text) == "" and len(text)<=8: #1 letter + 1 dot, for 4 times
    text = text.replace(".", "")
  rgx = "^([A-Z][ ])+[A-Z]$" # e.g., turn "U K" or "U S A" into "UK"
  if re.sub(rgx, "", text) == "":
    text = text.replace(" ", "")
  rgx = "^[A-Z]&[A-Z]$" # e.g., turn "U&K" into "UK"
  if re.sub(rgx, "", text) == "":
    text = text.replace("&", "")
  rgx = "^[A-Z]-[A-Z]$" # e.g., turn "U-K" into "UK"
  if re.sub(rgx, "", text) == "":
    text = text.replace("-", "")
  # rgx = "^[A-Z][a-z]{1,2}$" # e.g., turn "Pa" into "PA"
  # if re.sub(rgx, "", text) == "" and text.lower() not in consts.BAN_LIST_FOR_SPATIAL_ENTITY_EXTRACTION:
  #   text = text.upper()
  rgx = "^[A-Z][a-z]{1,3}[.]{1,1}$" # e.g., turn "Pa." into "PA"
  if re.sub(rgx, "", text) == "":
    text = text.upper().replace(".", "")
  
  process_nominatim = False
  if init_text == init_text.upper() and len(init_text)<=4:
    process_nominatim = True
  elif init_text != init_text.upper() and init_text != text and country_code_alpha2 != "-1":
    process_nominatim = True
    
  if process_nominatim:
    time.sleep(0.1)
    # use Nominatim to get entity names (based on my observations, Nominatim outperforms Geonames and Arcgis geocoders)
    client_nominatim = Nominatim(user_agent="geoapi")
    #client_nominatim = Nominatim(user_agent="arinik9")
    nominatim_locations = client_nominatim.geocode(text, exactly_one=False, addressdetails=True, language="en", timeout=20)
    if nominatim_locations is None:
      print("None for ", text)
    if nominatim_locations is not None:
      if len(nominatim_locations)>5: # limit the search with 5 results
        nominatim_locations = nominatim_locations[0:5]
      for loc in nominatim_locations:
        if loc.raw["address"]["country_code"].upper() == country_code_alpha2:
          text = loc.raw['display_name'].split(", ")[0]
          #print(text)
          return text
    return init_text
  return text



def resolve_manually_loc_name_ambiguity(loc_name, source_country_code, article_text):
  loc_type = "N/A"
  country_code = "N/A"
  processed = False
  fcode = "N/A"
  # ----------------------   
  if loc_name.lower() == "eu":
    loc_type = "continent"
    country_code = "-1"
    fcode = "-1"
    processed = True
  # ----------------------     
  if loc_name == "UK":
    loc_type = "country"
    country_code = "GBR"
    fcode = "PCLI"
    processed = True    
  # ----------------------      
  if loc_name == "US" or loc_name == "USA":
    loc_type = "country"
    country_code = "USA"
    fcode = "PCLI"
    processed = True
  # ----------------------
  if loc_name == "Northern Ireland" or loc_name == "Scotland":
    loc_type = "state"
    country_code = "GBR"
    fcode = "ADM1"
    processed = True
  # ---------------------- 
  if (loc_name == "NI" or loc_name == "Northern Ireland") and (source_country_code == "GBR" or source_country_code == "IRL"):
    loc_type = "state"
    country_code = "GBR"
    fcode = "ADM1"
    processed = True
  # ----------------------   
  if loc_name == "Hong Kong":
    loc_type = "country"
    country_code = "HK"
    fcode = "PCLS"
    processed = True
  # ----------------------   
  if loc_name == "Macao" or loc_name == "Macau":
    loc_type = "country"
    country_code = "MO"
    fcode = "PCLS"
    processed = True      
  # ----------------------   
  if loc_name == "Niger" and source_country_code == "NGA":
    loc_type = "state"
    country_code = "NGA"
    fcode = "ADM1"
    processed = True    
  # ----------------------  
  if loc_name == "SA":
    processed = True
    if source_country_code != "-1" and retrieve_continent_from_country_code(source_country_code) == "AF":
      loc_type = "country"
      country_code = "ZAF"
    elif loc_name == "SA" and source_country_code == "-1":
      if "South Africa" in article_text:
        loc_type = "country"
        country_code = "ZAF"
      elif "Saudi Arabia" in article_text:
        loc_type = "country"
        country_code = "SAU"
  # ----------------------      
  if loc_name == "Zim" and source_country_code != "-1" and source_country_code == "ZWE":
    loc_type = "country"
    country_code = "ZWE"
    processed = True
  # ----------------------      
  if loc_name == "MA":
    processed = True
    if source_country_code != "-1" and source_country_code == "MAR":
      loc_type = "country"
      country_code = "MAR"
      fcode = "PCLI"
    elif source_country_code == "-1" and "Morocco" in article_text:
      loc_type = "country"
      country_code = "MAR"
      fcode = "PCLI"
  # ----------------------  
  if loc_name == "TN" and source_country_code == "-1":
    processed = True
    if "Tunisia" in article_text:
        loc_type = "country"
        country_code = "TUN"
        fcode = "PCLI"
  # ----------------------  
  if loc_name == "PA":
    processed = True
    if source_country_code != "-1" and source_country_code == "PAN":
      loc_type = "country"
      country_code = "PAN"
      fcode = "PCLI"
    elif source_country_code == "-1" and "Panama" in article_text:
      loc_type = "country"
      country_code = "PAN"
      fcode = "PCLI"
  # ----------------------  
  if loc_name == "Meuse": # there is a bug in Nominatim about it
    if source_country_code != "-1" and source_country_code == "EGY":
      loc_type = "country"
      country_code = "EGY"
      fcode = "PCLI"
      processed = True
    elif source_country_code == "-1" and "Egypt" in article_text:
      loc_type = "country"
      country_code = "EGY"
      fcode = "PCLI"
      processed = True

  return(processed, loc_type, country_code, fcode)
        
        

def retrieve_in_batch_geocoding_results(spatial_entities_list, geonames_results_in_batch_filepath):

  results = {} # each key corresponds to a column in the future file
  for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST):
    results[i] = []
  results["text"] = []
  for i in range(len(spatial_entities_list)):
    text = spatial_entities_list[i]
    locs = geocode_raw_with_geonames(text)
    if locs is not None:
      for i in range(len(locs)):
        results[i].append(json.loads(locs[i].raw))
      for i in range(len(locs), consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST):
        results[i].append("-1")
      results["text"].append(text)
  df = pd.DataFrame(results)
  df.to_csv(geonames_results_in_batch_filepath, index=True, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
    
        