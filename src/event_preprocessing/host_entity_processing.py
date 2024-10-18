'''
Created on Nov 16, 2021

@author: nejat
'''


import re
import json
import os
import pandas as pd
import csv
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import string
from collections import OrderedDict

import src.consts as consts
import src.user_consts as user_consts
from src.geocoding.relative_spatial_entity import is_relative_spatial_entity
from src.event_normalization.host_entity_normalization import update_host_results_in_DB
from src.event_preprocessing.pattern_en import pluralize, singularize # https://www.geeksforgeeks.org/python-program-to-convert-singular-to-plural/



def retrieve_host_entity_list(raw_host_info_list, host_results_DB_filepath):
    preprocessed_host_list = []
    preprocessed_host_type_list = []
    for host_text in raw_host_info_list:
      host_list, host_type_list = retrieve_host_from_raw_sentence(host_text, host_results_DB_filepath, True) # it can be an empty list
      print(host_list, host_type_list)
      if len(host_list) == 0:
        preprocessed_host_list.append("-1")
        preprocessed_host_type_list.append("-1")
      else:
        preprocessed_host_list.append(host_list)
        preprocessed_host_type_list.append(host_type_list)
    return(preprocessed_host_list, preprocessed_host_type_list)




################################################################################################
# This method aims to extract host entities from (1) structured or (2) unstructured data.
# 1) If it takes in input a raw sentence text, i.e. structured,
#  (e.g. "a new case found in a poultry farm"), it retrieves a token list.
#  This token list consists of one-gram and bi-gram tokens after have removed stopwords.
#  (e.g. [new", "case", "new case", .... ])
# 2) If it is a structured text data (e.g. "dog, cat") with a well separated keywords,
#  it turns the text into a token list as well (e.g. ["dog", "cat"])
#
# It performs the processing in 2 steps:
# 1) turning the text into a token list
# 2) verifying if a token is found in the NCBI data. This step can be time consuming.
#    That is why it registers the search results into an auxiliary file in order to use it later as a lookup strategy.
#
# More details:
# - If a token have been already processed before (i.e. already found in the auxiliary file),
#    it just takes the result from this file. Otherwise, it verifies if a given token is found in the NCBI data
# - When it verifies if a given token is found in the NCBI data, it tries three forms of a token:
#    >> singular form
#    >> plural form
#    >> initial form (without change)
# - When it verifies if a given token is found in the NCBI data, it iterates over all data in the NCBI.
#    During this process, it first iterates over the most specific host keywords. If nothing found,
#    then it iterates over one level more generic host keywords, and so on.
#    This is why it uses this code line: "for lvl in list(range(20,0,-1)):" >> from more specific to more generic
#    For instance "racoon dog" is more specific compared to "dog".
#    However, the method finds a given keyword at a specific level, it continues to search other keywords of the same level,
#    but it stops searching other keywords for more generic levels.
#    This keyword genericity is determined by the levels.
#    Each host keyword is associated with a level.
# - NCBI data is currently structured by 3 host types: "avian", "mammal", "human"
#  The method can find host keywords from the NCBI data for both types, if they coexist in the same text.
#    
#
# PARAMETERS
# - sentence_text: structured or unstrucuted text data
# - NCBI_host_results_filepath: the auxiliary result filepath
# - is_raw_sentence: whether the sentence_text represents a sentence. If so, 
#                    it constitutes a list of one-gram and bi-gram tokens after have removed stopwords.
#
# OUTPUT:
# - host_list: host entities found in the NCBI data
# - host_type_list: the corresponding host types of the host list
#
# Example for the output:
# - host_list: ["dog", "poultry", "woman"]
# - host_type_list: ["mammal", "avian", "human"]
################################################################################################ 
def retrieve_host_from_raw_sentence(sentence_text, NCBI_host_results_filepath, is_raw_sentence=False):
  print(NCBI_host_results_filepath)
  if not os.path.exists(NCBI_host_results_filepath):
    # create an empty file 
    columns = ["result", "text"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(NCBI_host_results_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  df_batch_NCBI_results = pd.read_csv(NCBI_host_results_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_NCBI_results = dict(zip(df_batch_NCBI_results["text"], df_batch_NCBI_results.index))
  
  host_list = []
  host_type_list = []
  sentence_text = sentence_text.lower()
  
  sentence_keywords = [sentence_text]
  if not is_raw_sentence:
    if ";" in sentence_text:
      sentence_keywords = sentence_text.split(";")
      sentence_keywords = [s.strip() for s in sentence_keywords]
    elif "," in sentence_text:
      sentence_keywords = sentence_text.split(",")
      sentence_keywords = [s.strip() for s in sentence_keywords]
  elif is_raw_sentence:
    punctuations = string.punctuation
    stop_words = set(stopwords.words('english'))
    
    w_tokens = word_tokenize(sentence_text)
    rgx = "(^["+punctuations+"])|(["+punctuations+"]$)"
    tokens_adj = [re.sub(rgx, "", w) for w in w_tokens]
    tokens = []
    for t in w_tokens:
      r = re.compile(r'[\s{}]+'.format(re.escape(punctuations)))
      tokens.extend(r.split(t))
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words and re.search('[a-zA-Z]', w) is not None and not is_relative_spatial_entity(w)]
    filtered_tokens = [w for w in tokens if w.lower() not in consts.stopwords_ext]
    #filtered_tokens = [singularize(w) for w in filtered_tokens]
    bigram = [" ".join(x).strip() for x in ngrams(filtered_tokens, 2) if " ".join(x).strip() != ""]
    #trigram = [" ".join(x).strip() for x in ngrams(filtered_tokens, 3) if " ".join(x).strip() != ""]
    tokens_adj = [t for t in tokens_adj if t != ""]
    filtered_tokens = [t for t in filtered_tokens if t != ""]
    
    #grams = trigram + bigram + filtered_tokens + tokens_adj
    grams = bigram + filtered_tokens
    if ("lion" in grams and "sea lion" in grams) or ("lions" in grams and "sea lions" in grams):
      grams = [w for w in grams if "lion" != w and "lions" != w]
    sentence_keywords = list(OrderedDict.fromkeys(grams))
    # post-processing: remove the words with at 1 or 2 characters
    sentence_keywords = [sk for sk in sentence_keywords if len(sk)>2]
  
  print("--", sentence_keywords)
  for sentence_text in sentence_keywords:
    sing_text = singularize(sentence_text)
    plu_text = pluralize(sentence_text)
    #print("------- sentence_text", sentence_text)
    text_found = ""
    found_in_DB = False
    if sentence_text in inverted_indices_for_df_batch_NCBI_results:
      text_found = sentence_text
      found_in_DB = True
    elif sing_text in inverted_indices_for_df_batch_NCBI_results:
      text_found = sing_text
      found_in_DB = True
    elif plu_text in inverted_indices_for_df_batch_NCBI_results:
      text_found = plu_text
      found_in_DB = True
      
    if found_in_DB:
      host_indx = inverted_indices_for_df_batch_NCBI_results[text_found]
      row_host_result = df_batch_NCBI_results.iloc[host_indx]
      host_data = json.loads(row_host_result["result"])
      print(host_data)
      if host_data["text"] != "-1":
        host_list.append(host_data["text"])
        host_type_list.append(host_data["type"])
    else:
      any_found = False
      for host_type in list(consts.HOST_KEYWORDS_HIERARCHY_DICT.keys()): # hierarchy level 0
        found = False
        #if host_type in user_consts.USER_HOST_TYPES:
        host_subtype_data = consts.HOST_KEYWORDS_HIERARCHY_DICT[host_type] # it is a list of dicts with two keys: "text" and "hierarchy"
        # each entry in 'host_subtype_data["hierarchy"]' is organized from general to specialized tuple_data
        for lvl in list(range(20,0,-1)):
          #if h.is_host_info_empty(): #if a human case already found, or another host type, we do not treat a new one
          if found:
            break
          for dict_entry in host_subtype_data:
            # print(lvl, dict_entry)
            curr_lvl = dict_entry["level"]
            if lvl == curr_lvl:
              kw = dict_entry["text"].lower()
              text_found = ""
              if sentence_text == kw:
                found = True
                text_found = sentence_text
              elif sing_text == kw:
                found = True
                text_found = sing_text
              elif plu_text == kw:
                found = True
                text_found = plu_text
              if found:
                any_found = True
                txt = kw
                txt = txt.lower()
                # --------------------------------------------------------------------------
                # some manual processing
                txt = txt.replace("sealion","sea lion")
                txt = txt.replace("north american river otter", "northern american river otter")
                if txt == "vison":
                  txt = txt.replace("vison","neovison vison")
                txt = txt.replace("bottlenose dolphin","common bottlenose dolphin")
                txt = txt.replace("black bear","bear")
                txt = txt.replace("harbour porpoise","harbor porpoise")
                # --------------------------------------------------------------------------
                host_list.append(txt)
                host_type_list.append(host_type)
                result = {"text": txt, "type": host_type}
                df_batch, inverted_indices = update_host_results_in_DB(NCBI_host_results_filepath, text_found, result)
                df_batch_NCBI_results = df_batch
                inverted_indices_for_df_batch_NCBI_results = inverted_indices
                #
                break
      if not any_found:
        result = {"text": "-1", "type": "-1"}
        update_host_results_in_DB(NCBI_host_results_filepath, sentence_text, result)
        if sentence_text != sing_text:
          update_host_results_in_DB(NCBI_host_results_filepath, sing_text, result)
        if sentence_text != plu_text:
          update_host_results_in_DB(NCBI_host_results_filepath, plu_text, result)    
     
  # post-processing: keep only desired host types
  filt_host_list = []
  filt_host_type_list = []
  for i in range(len(host_list)):
     if host_type_list[i] in user_consts.USER_HOST_TYPES:
       filt_host_list.append(host_list[i])
       filt_host_type_list.append(host_type_list[i])
  return filt_host_list, filt_host_type_list
