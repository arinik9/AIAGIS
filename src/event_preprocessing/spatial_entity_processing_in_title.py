'''
Created on Nov 16, 2021

@author: nejat
'''

import os
import json
import pandas as pd
import numpy as np
import string
from collections import Counter
import time
import csv
import re
import spacy

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from geopy.geocoders import Nominatim
from iso3166 import countries

import src.consts as consts
from src.geocoding.relative_spatial_entity import is_relative_spatial_entity, idenitfy_relative_spatial_cardinal
from src.geocoding.geocode import geocode_batch_with_nominatim, geocode_batch_with_arcgis, geocode_batch_with_geonames, geocode_with_geonames
from src.util_gis import retrieve_continent_from_country_code, get_country_apolitical_name_from_alpha3_code
from src.geocoding.prepare_spatial_entity_lookup_table import build_spatial_entity_lookup_table
from src.util_gis import process_spatial_entity_abbreviation, resolve_manually_loc_name_ambiguity
from src.event_normalization.spatial_entity_normalization import update_geocoding_results_in_DB



########################################################################
# This method retrieves the location names from raw text 
#  (e.g. title of a news article) by using SpaCy.
#
#
# df_articles: the file containing the input raw event informations,
#              i.e. articles with raw text, title, etc.
# media_source_resolution_filepath: the file path, in which the association between
#                                    the news sources and their affiliated publication
#                                    country information (e.g. "Le Monde" et "France") 
#                                    is stored (when possible).
# title_spacy_raw_locations_filepath: output file path, where the results will be stored
########################################################################
def retrieve_spacy_raw_locations_from_title(df_articles, media_source_resolution_filepath, title_spacy_raw_locations_filepath):
  ### named entity types: https://www.dataknowsall.com/ner.html
  nlp_lg = spacy.load("en_core_web_lg") # an IDE can give an error, but it works fine
  nlp_trf = spacy.load("en_core_web_trf") # an IDE can give an error, but it works fine
  
  df_media_source = pd.read_csv(media_source_resolution_filepath, sep=",", keep_default_na=False)
  article_id_to_source_country_code_assoc = dict(zip(df_media_source[consts.COL_ARTICLE_ID], df_media_source["pub_country_code"]))
  
  loc_name_list = []
  loc_label_list = []
  article_id_list = []
  title_list = []
  article_text_list = []
  for index, row in df_articles.iterrows():
    article_id = row[consts.COL_ID]
    print(index, article_id)
    title = row[consts.COL_TITLE]
    source_country_code = article_id_to_source_country_code_assoc[article_id]
    source_country_code_alpha2 = "-1"
    if source_country_code != "-1":
      source_country_code_alpha2 = countries.get(source_country_code).alpha2
    article_text = row[consts.COL_TEXT]
    
    # --------------
    # apply a simple event_preprocessing for Japan and Serbia
    title = title.replace("pref.", "prefecture")
    title = title.replace("Pref.", "Prefecture")
    title = title.replace("APTOPIX", "") # APTOPIX SERBIA AND MONTENEGRO BIRD FLU
    title = title.replace("aptopix", "") # APTOPIX SERBIA AND MONTENEGRO BIRD FLU
    # --------------
  
    ent_found = False
    ents = nlp_trf(title).ents # we find more spatial entities with 'trf' model
    # but, it is possible it misses some texts, so we add additional model 'lg'
    for ent in ents: # iterate over the named entities of the text
      if ent.label_ == "GPE" or ent.label_ == "LOC" or ent.label_ == "FAC":
        ent_found = True
        break
    if not ent_found: 
      ents = nlp_lg(title).ents
      for ent in ents: # iterate over the named entities of the text
        if ent.label_ == "GPE" or ent.label_ == "LOC" or ent.label_ == "FAC":
          ent_found = True
          break
    if ent_found: 
      for ent in ents: # iterate over the named entities of the text
        if ent.label_ == "GPE" or ent.label_ == "LOC" or ent.label_ == "FAC":
          text = ent.text
          if ".." in text:
            rgx = r"(?i)[.]{2,}$" # remove multiple dots in the end
            text = re.sub(rgx, "", text)
          rgx = "(?i)(^the )"
          text = re.sub(rgx, "", text)
          rgx = "(?i)( region)"
          text = re.sub(rgx, "", text)
          rgx = "(?i)(^upper)|(^lower)" # >> for Upper and Lower Rhine => it is a complicated situation 
          text = re.sub(rgx, "", text)
          rgx = "(^-)|(-$)|(:$)"
          text = re.sub(rgx, "", text)
          # if text != text.upper(): # remove a dot point from the end, e.g. Pa. >> we do not treat U.K.
          #   rgx = "(\.$)"
          #   text = re.sub(rgx, "", text)
          text = text.strip()
                
          process = False
          if text.lower().replace(",","").strip() not in consts.BAN_LIST_FOR_SPATIAL_ENTITY_EXTRACTION:
            text = text.replace(",","").strip()
            if "west nile" not in text.lower():
              if not ((text.lower() == "dc" and source_country_code == "IND") or (text.lower() == "cm" and source_country_code == "IND")): # >> 'DC' stands for "Deputy commissioner", CM: Chief minister
                #if not (is_relative_spatial_entity(ent.text) and idenitfy_relative_spatial_cardinal(ent.text) == "-1"):
                #print(text, idenitfy_relative_spatial_cardinal(text))
                if not (idenitfy_relative_spatial_cardinal(text) == "-1"):
                  if text != "" and len(text)>1 and not text.isnumeric():
                    process = True
          if process:
            # for abbreviations: we do it here, because we check above the ban list
            text = trim_and_adjust_spatial_entity_name(text)
            if is_a_spatial_entity_abbreviation(text):
              text = process_spatial_entity_abbreviation(text, source_country_code_alpha2)
            rgx = "(?i)('s|’s)" # good example: "Africa's", bad example: "St John's"
            text = re.sub(rgx, "", text)
            # -------
            # to prevent from having "North of France"
            if is_relative_spatial_entity(text) and " of " in text:
              text = text.split(" of ")[1]
            # -------
            # to prevent from having "West Africa" or "East Europe"
            ent_tokens = text.split(" ")
            if is_relative_spatial_entity(text) and len(ent_tokens) == 2: # if there are 2 words/tokens 
              if not (text == "North America" or text == "South America" or text == "South Africa"): 
                if ent_tokens[1] in consts.CONTINENTS_NAME or ent_tokens[1] in consts.CONTINENTS_CODE:
                  text = ent_tokens[1]
            
            #text = idenitfy_relative_spatial_cardinal(text)
            #print("!!!! ", ent.text, " => ", text)
            loc_name_list.append(text)
            loc_label_list.append(ent.label_)
            title_list.append(title)
            article_id_list.append(article_id)
            article_text_list.append(article_text)

  source_country_code_list = [article_id_to_source_country_code_assoc[a_id] for a_id in article_id_list]
          
          
  df = pd.DataFrame(data={consts.COL_ARTICLE_ID: article_id_list, "text": article_text_list, consts.COL_TITLE: title_list,\
                        "label": loc_label_list,  "name": loc_name_list, "media_source_country_code": source_country_code_list})

  df.to_csv(title_spacy_raw_locations_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  

  
def enrich_raw_spacy_locations_with_country_code_and_loc_type(title_spacy_raw_locations_filepath, title_spacy_raw_locations_with_type_filepath):
  df_title_spacy_raw_locations = pd.read_csv(title_spacy_raw_locations_filepath, sep=";", keep_default_na=False)
      
  loc_name_list = df_title_spacy_raw_locations["name"]
  loc_label_list = df_title_spacy_raw_locations["label"]
  article_id_list = df_title_spacy_raw_locations[consts.COL_ARTICLE_ID]
  article_text_list = df_title_spacy_raw_locations["text"]
  title_list = df_title_spacy_raw_locations[consts.COL_TITLE]
  source_country_code_list = df_title_spacy_raw_locations["media_source_country_code"]
  
  client_nominatim = Nominatim(user_agent="geoapi")
  
  loc_type_list = []
  country_code_list = []
  for i in range(len(loc_name_list)):
    loc_name = loc_name_list[i]
    source_country_code = source_country_code_list[i]
    article_text = article_text_list[i]
    print("----- ", i, loc_name, source_country_code)
    
    processed = False
    # ----------------------   
    processed, loc_type, country_code, fcode = resolve_manually_loc_name_ambiguity(loc_name, source_country_code, article_text)      
    #loc_type = "other" >> by default
    #country_code = "N/A" >> by default
    if not processed and loc_name != "-1" and len(loc_name)>1: # it has at least 2 characters
      try:
        time.sleep(0.1)
        res = client_nominatim.geocode(loc_name, exactly_one=True, addressdetails=True, language="en", timeout=20)
        if res is not None:
          if "country" not in res.raw["address"]:
            loc_type = "continent or sea"
            country_code = "-1"
          else:
            details = res.raw["address"].copy()
            del details["country"]
            del details["country_code"]
            if len(details) == 0:
              loc_type = "country"
              country_code = countries.get(res.raw["address"]["country_code"]).alpha3
              
              # check if it is a country abbreviation
              if len(loc_name) <= 3:
                rgx = "^([A-Z])+"
                if re.sub(rgx, "", loc_name) != "": # it is not a country abbreviation, since there are some lowercase letter
                  loc_type = "other"
                  country_code = "N/A"
                else: # if it is a country abbreviation
                  # check if the text country's continent is the same as the source media's continent
                  # e.g. "France" > "EU" and "Italy" > "EU"
                  if source_country_code != "-1" and retrieve_continent_from_country_code(country_code) != retrieve_continent_from_country_code(source_country_code):
                    loc_type = "other"
                    country_code = "N/A"
          
      except:
        print("error in retrieve_location_info_from_title() with name=", loc_name)
        pass
    loc_type_list.append(loc_type)
    country_code_list.append(country_code)
  # ------------------------------------------------------
    
  df = pd.DataFrame(data={consts.COL_ARTICLE_ID: article_id_list, consts.COL_TITLE: title_list,\
                        "label": loc_label_list,  "name": loc_name_list, "media_source_country_code": source_country_code_list, \
                        "type": loc_type_list, "country_code": country_code_list})

  df.to_csv(title_spacy_raw_locations_with_type_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
   
   
    
def retrieve_titles_without_spacy_locations(df_articles, title_spacy_raw_locations_with_type_filepath,\
                                             titles_without_spacy_locations_filepath, media_source_resolution_filepath):
  df_title_spacy_locs = pd.read_csv(title_spacy_raw_locations_with_type_filepath, sep=";", keep_default_na=False) # .iloc[0:500]
 
  df_media_source = pd.read_csv(media_source_resolution_filepath, sep=",", keep_default_na=False)
  #source_to_country_assoc = dict(zip(df_media_source["url"].str.replace("www.",""), df_media_source["pub_country"]))
  article_id_to_source_country_code_assoc = dict(zip(df_media_source[consts.COL_ARTICLE_ID], df_media_source["pub_country_code"]))

  df_titles_without_spacy_locations = df_articles[~df_articles[consts.COL_ID].isin(df_title_spacy_locs[consts.COL_ARTICLE_ID])]
  df_titles_without_spacy_locations["media_source_country_code"] = df_titles_without_spacy_locations[consts.COL_ID].apply(lambda x: article_id_to_source_country_code_assoc[x])
  df_titles_without_spacy_locations.to_csv(titles_without_spacy_locations_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
    



def retrieve_location_info_from_title_with_knowledge_base(data_folder, title_spacy_raw_locations_with_type_filepath, \
                                                          titles_without_spacy_locations_filepath, \
                                                          title_locations_from_knowledge_base_auxiliary_filepath, \
                                                          title_locations_from_knowledge_base_final_filepath):
  
  df_titles_without_spacy_locations = pd.read_csv(titles_without_spacy_locations_filepath, sep=";", keep_default_na=False, encoding='utf8')
  
  # ===========================================
  # SPACY
  df_title_locations = pd.read_csv(title_spacy_raw_locations_with_type_filepath, sep=";", keep_default_na=False, encoding='utf8')
  extr_spacy_spatial_entity_dict = dict(zip(df_title_locations["name"], df_title_locations["country_code"]))
  extr_spacy_spatial_entity_list = np.unique(df_title_locations["name"].to_list())
  
  # ===========================================
  # LOOKUP TABLE
  extr_lookup_spatial_entity_dict, lookup_country_to_entites_dict  = build_spatial_entity_lookup_table(data_folder)
  extr_lookup_spatial_entity_list = extr_lookup_spatial_entity_dict.keys()
  extr_lookup_spatial_entity_list = [ent.split("(")[0] if "(" in ent else ent for ent in extr_lookup_spatial_entity_list]
  extr_lookup_spatial_entity_list = [ent.replace(")","") if ")" in ent else ent for ent in extr_lookup_spatial_entity_list]
  extr_lookup_spatial_entity_list = [ent.replace("\\", " ") if "\\" in ent else ent for ent in extr_lookup_spatial_entity_list]
  
  # ===========================================
  # COUNTRY/NATIONALITY LIST
  filepath = os.path.join(data_folder, "nationalities", "Nationality_to_Country_Assoc.csv")
  df_country_nationality = pd.read_csv(filepath, sep=";", keep_default_na=False)
  extr_country_nationality_spatial_entity_dict = dict(zip(df_country_nationality["nationality"], df_country_nationality["country_code"]))
  extr_country_nationality_spatial_entity_dict2 = dict(zip(df_country_nationality["country_name"], df_country_nationality["country_code"]))
  extr_country_nationality_spatial_entity_dict.update(extr_country_nationality_spatial_entity_dict2)
  extr_country_nationality_spatial_entity_list = extr_country_nationality_spatial_entity_dict.keys()
  
  # -------------------------------------------
  nlp_trf = spacy.load("en_core_web_trf") # an IDE can give an error, but it works fine
  
  punctuations = string.punctuation
  nltk.download('punkt')
  stop_words = set(stopwords.words('english'))
  
  new_spacy_spatial_entity_list = []
  new_spacy_spatial_country_list = []
  new_lookup_spatial_entity_list = []
  new_lookup_spatial_country_list = []
  new_country_nationality_spatial_entity_list = []
  new_country_nationality_spatial_country_list = []
  
  for index, row in df_titles_without_spacy_locations.iterrows():
    title = row[consts.COL_TITLE]
    source_media_country_code = row["media_source_country_code"]
    # print(index, source_media_country_code)
    
    curr_extr_lookup_spatial_entity_list = []
    if source_media_country_code in lookup_country_to_entites_dict:
      curr_extr_lookup_spatial_entity_list = lookup_country_to_entites_dict[source_media_country_code]
    
    w_tokens = word_tokenize(title)
    rgx = "(^["+punctuations+"])|(["+punctuations+"]$)"
    tokens_adj = [re.sub(rgx, "", w) for w in w_tokens]
    tokens = []
    for t in tokens_adj:
      r = re.compile(r'[\s{}]+'.format(re.escape(punctuations)))
      tokens.extend(r.split(t))
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    bigram = [" ".join(x) for x in ngrams(filtered_tokens, 2)]
    trigram = [" ".join(x) for x in ngrams(filtered_tokens, 3)]
    # word_grams = tokens_adj + filtered_tokens + bigram + trigram
    word_grams = filtered_tokens + bigram + trigram
    
    # ===========================================
    # SPACY
    curr_spacy_spatial_entity_list = [t for t in word_grams if t in extr_spacy_spatial_entity_list]
    curr_spacy_spatial_entity_list_str = "-1"
    if len(curr_spacy_spatial_entity_list) > 0:
      #curr_spacy_spatial_entity_list_str = ", ".join(np.unique(curr_spacy_spatial_entity_list))
      curr_spacy_spatial_entity_list_str = ", ".join(curr_spacy_spatial_entity_list)
    new_spacy_spatial_entity_list.append(curr_spacy_spatial_entity_list_str)
    
    curr_spacy_spatial_country_list = [extr_spacy_spatial_entity_dict[t] for t in word_grams if t in extr_spacy_spatial_entity_list]
    curr_spacy_spatial_country_list_str = "-1"
    if len(curr_spacy_spatial_country_list) > 0:
      #curr_spacy_spatial_country_list_str = ", ".join(np.unique(curr_spacy_spatial_country_list))
      curr_spacy_spatial_country_list_str = ", ".join(curr_spacy_spatial_country_list)
    new_spacy_spatial_country_list.append(curr_spacy_spatial_country_list_str)
      
    # ===========================================
    # LOOKUP TABLE
    curr_lookup_spatial_entity_list = [t for t in word_grams if t in curr_extr_lookup_spatial_entity_list]
    curr_lookup_spatial_entity_list_str = "-1"
    if len(curr_lookup_spatial_entity_list) > 0:
      #curr_lookup_spatial_entity_list_str = ", ".join(np.unique(curr_lookup_spatial_entity_list))
      curr_lookup_spatial_entity_list_str = ", ".join(curr_lookup_spatial_entity_list)
    new_lookup_spatial_entity_list.append(curr_lookup_spatial_entity_list_str)
    
    curr_lookup_spatial_country_list = [source_media_country_code for t in word_grams if t in curr_extr_lookup_spatial_entity_list]
    curr_lookup_spatial_country_list_str = "-1"
    if len(curr_lookup_spatial_country_list) > 0:
      #curr_lookup_spatial_country_list_str = ", ".join(np.unique(curr_lookup_spatial_country_list))
      curr_lookup_spatial_country_list_str = ", ".join(curr_lookup_spatial_country_list)
    new_lookup_spatial_country_list.append(curr_lookup_spatial_country_list_str)
    
    # ===========================================
    # COUNTRY/NATIONALITY LIST
    curr_country_nationality_spatial_entity_list = [t for t in word_grams if t in extr_country_nationality_spatial_entity_list]
    curr_country_nationality_spatial_entity_list_str = "-1"
    if len(curr_country_nationality_spatial_entity_list) > 0:
      #curr_country_nationality_spatial_entity_list_str = ", ".join(np.unique(curr_country_nationality_spatial_entity_list))
      curr_country_nationality_spatial_entity_list_str = ", ".join(curr_country_nationality_spatial_entity_list)
    new_country_nationality_spatial_entity_list.append(curr_country_nationality_spatial_entity_list_str)
    
    curr_country_nationality_spatial_country_list = [extr_country_nationality_spatial_entity_dict[t] for t in word_grams if t in extr_country_nationality_spatial_entity_list]
    curr_country_nationality_spatial_country_list_str = "-1"
    if len(curr_country_nationality_spatial_country_list) > 0:
      #curr_country_nationality_spatial_country_list_str = ", ".join(np.unique(curr_country_nationality_spatial_country_list))
      curr_country_nationality_spatial_country_list_str = ", ".join(curr_country_nationality_spatial_country_list)
    new_country_nationality_spatial_country_list.append(curr_country_nationality_spatial_country_list_str)      
          
  # --------------------------------------------
  
  df_titles_without_spacy_locations["spacy_spatial_entity_name"] = new_spacy_spatial_entity_list
  df_titles_without_spacy_locations["spacy_spatial_country_code"] = new_spacy_spatial_country_list
  df_titles_without_spacy_locations["lookup_spatial_entity_name"] = new_lookup_spatial_entity_list
  df_titles_without_spacy_locations["lookup_spatial_country_code"] = new_lookup_spatial_country_list
  df_titles_without_spacy_locations["country_nationality_spatial_entity_name"] = new_country_nationality_spatial_entity_list
  df_titles_without_spacy_locations["country_nationality_spatial_country_code"] = new_country_nationality_spatial_country_list
  
  df_titles_without_spacy_locations.to_csv(title_locations_from_knowledge_base_auxiliary_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  
  
  # =============================================
  # construct the final data structure
  article_id_list = []
  title_list = []
  label_list = []
  type_list = []
  name_list = []
  country_code_list = []
  media_source_country_code_list = []
  for index, row in df_titles_without_spacy_locations.iterrows():
    article_id = row[consts.COL_ID]
    #print("-------", index, article_id)
    title = row[consts.COL_TITLE]
    source_media_country_code = row["media_source_country_code"]
    
    spacy_spatial_entity_name_list = []
    spacy_spatial_country_code_list = []
    if row["spacy_spatial_entity_name"] != "-1":
      spacy_spatial_entity_name_list = row["spacy_spatial_entity_name"].split(", ")
      spacy_spatial_country_code_list = row["spacy_spatial_country_code"].split(", ")
      if row["spacy_spatial_country_code"] == "N/A" and len(spacy_spatial_entity_name_list) != len(spacy_spatial_country_code_list):
        spacy_spatial_country_code_list = ["N/A"]*len(spacy_spatial_entity_name_list)
      elif row["spacy_spatial_country_code"] == "-1" and len(spacy_spatial_entity_name_list) != len(spacy_spatial_country_code_list):
        spacy_spatial_country_code_list = ["N/A"]*len(spacy_spatial_entity_name_list)
    lookup_spatial_entity_name_list = []
    lookup_spatial_country_code_list = []
    if row["lookup_spatial_entity_name"] != "-1":
      lookup_spatial_entity_name_list = row["lookup_spatial_entity_name"].split(", ")
      lookup_spatial_country_code_list = row["lookup_spatial_country_code"].split(", ")
      if len(lookup_spatial_entity_name_list) != len(lookup_spatial_country_code_list):
        lookup_spatial_country_code_list = lookup_spatial_country_code_list*len(lookup_spatial_entity_name_list)
        
    country_nationality_spatial_entity_name_list = []
    country_nationality_spatial_country_code_list = []
    if row["country_nationality_spatial_entity_name"] != "-1":
      country_nationality_spatial_entity_name_list = row["country_nationality_spatial_entity_name"].split(", ")
      country_nationality_spatial_country_code_list = row["country_nationality_spatial_country_code"].split(", ")
      if len(country_nationality_spatial_entity_name_list) != len(country_nationality_spatial_country_code_list):
        country_nationality_spatial_country_code_list = country_nationality_spatial_country_code_list*len(country_nationality_spatial_entity_name_list)
        
    spatial_entity_name_list = spacy_spatial_entity_name_list + lookup_spatial_entity_name_list\
                             + country_nationality_spatial_entity_name_list
    spatial_entity_country_list = spacy_spatial_country_code_list + lookup_spatial_country_code_list\
                             +country_nationality_spatial_country_code_list
                             
    nb_entity = len(spatial_entity_name_list)
    name_list.extend(spatial_entity_name_list)
    country_code_list.extend(spatial_entity_country_list)
    article_id_list.extend([article_id]*nb_entity)
    title_list.extend([title]*nb_entity)
    label_list.extend(["-1"]*nb_entity)
    type_list.extend(["other"]*len(spacy_spatial_entity_name_list))
    type_list.extend(["state/city"]*len(lookup_spatial_entity_name_list))
    type_list.extend(["country"]*len(country_nationality_spatial_entity_name_list))
    media_source_country_code_list.extend([source_media_country_code]*nb_entity)
  data = {"article_id": article_id_list, "title": title_list, "label": label_list, \
          "name": name_list, "media_source_country_code": media_source_country_code_list, 
          "type": type_list, "country_code": country_code_list}  
    
  df = pd.DataFrame(data)
  df.to_csv(title_locations_from_knowledge_base_final_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)





def combine_all_title_locations(title_spacy_raw_locations_with_type_filepath, title_locations_from_knowledge_base_final_filepath, \
                                title_locations_final_filepath):
  df_title_locs_spacy = pd.read_csv(title_spacy_raw_locations_with_type_filepath, sep=";", keep_default_na=False) # .iloc[0:500]
  df_title_locs_from_knowledge_base = pd.read_csv(title_locations_from_knowledge_base_final_filepath, sep=";", keep_default_na=False) # .iloc[0:500]
  
  df_title_locs = pd.concat([df_title_locs_spacy.reset_index(), df_title_locs_from_knowledge_base.reset_index()])
  print(df_title_locs_spacy.shape, df_title_locs_from_knowledge_base.shape, df_title_locs.shape)
  df_title_locs.to_csv(title_locations_final_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  
      


def apply_different_gecoders_on_spatial_entities_in_title_for_normalization(title_locations_final_filepath, title_locations_final_with_geocoding_results_filepath):
  df_title_locs = pd.read_csv(title_locations_final_filepath, sep=";", keep_default_na=False) # .iloc[0:500]
  
  df_title_locs["country_bias"] = df_title_locs["media_source_country_code"] # temporary column
  df_title_locs_na = df_title_locs[df_title_locs["country_code"] == "N/A"]
  df_title_locs_ok = df_title_locs[df_title_locs["country_code"] != "N/A"]
  
  df_title_locs_na.reset_index(drop=True, inplace=True)
  df_title_locs_ok.reset_index(drop=True, inplace=True)

  article_id_list = np.unique(df_title_locs[consts.COL_ARTICLE_ID].to_numpy())
  
  print(df_title_locs_na["country_bias"])
  # ---------------------------------------------------------
  # adjust country bias column in 'df_title_locs_na'
  for article_id in article_id_list:
    #print("----------- ", article_id)
    res = df_title_locs_ok[df_title_locs_ok[consts.COL_ARTICLE_ID] == article_id]
    if res.shape[0] == 1: # there is one country mention in the title
      # TODO: counter example: B.1.525 variant, linked to Nigeria, now found in Telangana >> Telangana is in India, not in Nigeria
      #print(res)
      country_code = res.iloc[0]["country_code"]
      if country_code != "-1":
        indx = (df_title_locs_na[consts.COL_ARTICLE_ID] == article_id)
        #if df_title_locs_na.loc[indx].shape[0] > 0: # if there is any other spatial entities other than country and continent
        df_title_locs_na.loc[indx, "country_bias"] = country_code
        # workaround: we put the country name, after a comma, instead of limiting directly the search within a country
        df_title_locs_na.loc[indx, "name"] = df_title_locs_na.loc[indx, "name"] + ", " + countries.get(country_code).name
  # ---------------------------------------------------------
  
  spatial_entity_list = df_title_locs_na["name"].to_numpy()
  country_bias_list = df_title_locs_na["country_bias"].to_numpy()

  print("--------- geonames ")
  geonames_result_list = geocode_batch_with_geonames(spatial_entity_list, country_bias_list)
  df_geonames = pd.DataFrame(geonames_result_list)
  df_geonames.columns= ["name_geonames", "country_code_geonames", "raw_data_json_geonames"]
  df_geonames["raw_data_json_geonames"] = df_geonames["raw_data_json_geonames"].apply(lambda x: json.dumps(x, ensure_ascii=False))
  print(df_geonames.shape)
  
  print("--------- arcgis ")
  arcgis_result_list = geocode_batch_with_arcgis(spatial_entity_list, country_bias_list)
  df_arcgis = pd.DataFrame(arcgis_result_list)
  df_arcgis.columns= ["name_arcgis", "country_code_arcgis", "raw_data_json_arcgis"]
  df_arcgis["raw_data_json_arcgis"] = df_arcgis["raw_data_json_arcgis"].apply(lambda x: json.dumps(x, ensure_ascii=False))
  print(df_arcgis.shape)
  
  print("--------- nominatim ")
  nominatim_result_list = geocode_batch_with_nominatim(spatial_entity_list, country_bias_list)
  df_nominatim = pd.DataFrame(nominatim_result_list)
  df_nominatim.columns= ["name_nominatim", "country_code_nominatim", "raw_data_json_nominatim"]
  df_nominatim["raw_data_json_nominatim"] = df_nominatim["raw_data_json_nominatim"].apply(lambda x: json.dumps(x, ensure_ascii=False))
  print(df_nominatim.shape)

  
  df_title_locs_na["name"] = df_title_locs_na["name"].apply(lambda x: x.split(", ")[0]) # return back from workaround solution
  title_locations_new = pd.concat([df_title_locs_na, df_nominatim, df_arcgis, df_geonames], axis=1)
      
  df_title_locs_ok["name_nominatim"] = "-1"
  df_title_locs_ok["country_code_nominatim"] = "-1"
  df_title_locs_ok["raw_data_json_nominatim"] = "-1"
  df_title_locs_ok["name_arcgis"] = "-1"
  df_title_locs_ok["country_code_arcgis"] = "-1"
  df_title_locs_ok["raw_data_json_arcgis"] = "-1"
  df_title_locs_ok["name_geonames"] = "-1"
  df_title_locs_ok["country_code_geonames"] = "-1"
  df_title_locs_ok["raw_data_json_geonames"] = "-1"
  
  title_locations_new_total = pd.concat([title_locations_new, df_title_locs_ok], axis=0)
  title_locations_new_total = title_locations_new_total.drop("country_bias", axis=1)
  
  title_locations_new_total.to_csv(title_locations_final_with_geocoding_results_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
    

def is_a_spatial_entity_abbreviation(text):
  rgx = "^[A-Z][A-Z]{0,2}[A-Z]$"
  if re.search(rgx, text) is not None: # if it is an abbreviation
    return True
  return False



def trim_and_adjust_spatial_entity_name(text):
  if "Co " in text:
    text = text.replace("Co ", "County ").strip()
    
  for ban_kw in consts.BAN_KEYWORDS_LIST_FOR_SPATIAL_ENTITIES:
    if " "+ban_kw in text.lower():
      text = text.replace(" "+ban_kw, "").strip()
    if " "+ban_kw.lower() in text.lower():
      text = text.replace(" "+ban_kw.lower(), "").strip()
    if " "+ban_kw.upper() in text.lower():
      text = text.replace(" "+ban_kw.lower(), "").strip()
    if " "+ban_kw.title() in text.lower():
      text = text.replace(" "+ban_kw.title(), "").strip()
      
  if text.upper() == text and len(text)>4:
    text = text.title()  
  return text


    
def normalize_spatial_entities_in_title_from_geocoding_results(title_locations_final_with_geocoding_results_filepath,\
                                                                geonames_results_in_batch_filepath, \
                                                                title_locations_final_with_geonames_filepath):
  df_title_geocoding_results = pd.read_csv(title_locations_final_with_geocoding_results_filepath, sep=";", keep_default_na=False) # .iloc[0:500]
  
  if not os.path.exists(geonames_results_in_batch_filepath):
    # create an empty file 
    columns = [str(i) for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST)] + ["text"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(geonames_results_in_batch_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  #
  df_batch_geonames_results = pd.read_csv(geonames_results_in_batch_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_geonames_results = dict(zip(df_batch_geonames_results["text"], df_batch_geonames_results.index))


  article_ids = []
  media_source_country_code_list = []
  value_list = []
  country_bias_list = []
  type_list = []
  for index, row in df_title_geocoding_results.iterrows():
    print("----------", index, "/", df_title_geocoding_results.shape[0])
    article_id = row["article_id"]
    media_source_country_code = row["media_source_country_code"]
    text = row["name"]
    
    text = trim_and_adjust_spatial_entity_name(text)
    if is_a_spatial_entity_abbreviation(text):
      text = row["name_nominatim"]

    processed = False
    if row["country_code"] != "N/A" and row["country_code"] != "-1": # if it is a country
      # nationalities and country names are mixed up in this category
      country_name = get_country_apolitical_name_from_alpha3_code(row["country_code"])
      value_list.append(country_name)
      country_bias_list.append(row["country_code"])
      type_list.append("country")
      processed = True
    elif idenitfy_relative_spatial_cardinal(text) == "-1":
      pass
    else:
        country_codes = [row["country_code_nominatim"], row["country_code_arcgis"], row["country_code_geonames"]]
        country_codes = [country_code for country_code in country_codes if country_code != "-1" and country_code != ""]
        # this step is really a bit restrictive in order to get a high precision score
        ##if len(country_codes)>1 and len(np.unique(country_codes)) == 1: # at least 2, all from the same country
          ##country_bias_list.append(country_codes[0])
        if len(country_codes)>1: # majority vote, but we check equlity cases, for intanec ["UK", "FR"] >> here, there is no majority
          counter_res = Counter(country_codes)
          counts_dict = dict(counter_res)
          maj_country_code = max(country_codes, key=country_codes.count)
          max_count = counts_dict[maj_country_code]
          nb_max = len([val for val in counts_dict.values() if val == max_count])
          if nb_max == 1: # both country codes point to the same entity
            country_bias_list.append(maj_country_code)
            value_list.append(text)
            type_list.append("other")
            processed = True

    if processed:
      article_ids.append(article_id)
      media_source_country_code_list.append(media_source_country_code)
      
  # ------------------------------------------------------------------------
  
  geonames_json_list = [] 
  for i, spatial_entity in enumerate(value_list):
    print("----------", i, "/", len(value_list))
    #time.sleep(1)
    country_bias = country_bias_list[i]
    country_limit = None
    spatial_entity_adj = spatial_entity
    if type_list[i] != "country":
      country_limit = get_country_apolitical_name_from_alpha3_code(country_bias)
      spatial_entity_adj = spatial_entity + ", " + country_limit
    print(spatial_entity_adj)
    # ==================================================
    # retrieve the results from our database
    locs = None
    if spatial_entity_adj in inverted_indices_for_df_batch_geonames_results:
      #print("DB ACCESS!!!!")
      gindx = inverted_indices_for_df_batch_geonames_results[spatial_entity_adj]
      row_locs = df_batch_geonames_results.iloc[gindx]
      locs = [json.loads(row_locs[str(i)]) for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST)] # we access the columns
    # ==================================================
    res = geocode_with_geonames(spatial_entity, [country_bias], locs, country_limit)
    if locs is None and res["geonamesId"] != '-1':  # update DB
      df_batch, inverted_indices = update_geocoding_results_in_DB(geonames_results_in_batch_filepath, spatial_entity_adj)
      df_batch_geonames_results = df_batch
      inverted_indices_for_df_batch_geonames_results = inverted_indices
    geonames_json_list.append(json.dumps(res["raw_data"], ensure_ascii=False))
    
  df = pd.DataFrame({"article_id": article_ids, "media_source_country_code": media_source_country_code_list,\
       "country_bias": country_bias_list, "final_value": value_list, "final_geonames_json": geonames_json_list}) 
  
  df2 = df[df["final_geonames_json"] != '-1']

  df2.to_csv(title_locations_final_with_geonames_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)




