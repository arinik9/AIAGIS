'''
Created on Nov 16, 2021

@author: nejat
'''

import src.consts as consts
import src.user_consts as user_consts
import re
import unicodedata
import numpy as np

  
def retrieve_disease_entity_list(raw_disease_info_list, disease_name):
  preprocessed_disease_list = []
  preprocessed_disease_type_list = []
  for disease_text in raw_disease_info_list:
    disease_list, disease_type_list = retrieve_disease_from_raw_sentence(disease_text, disease_name) # it can be an empty list
    if len(disease_list) == 0:
      preprocessed_disease_list.append("-1")
      preprocessed_disease_type_list.append("-1")
    else:
      preprocessed_disease_list.append(disease_list)
      preprocessed_disease_type_list.append(disease_type_list)
  return(preprocessed_disease_list, preprocessed_disease_type_list)



  ################################################################################################
  # This method retrieves the spatial entities from titles (i.e. raw small text) with 2 strategies:
  # 1) SpaCy
  # 2.1) Lookup table based on existing spatial entities of a database (up to ADM3 level)
  # 2.2) Lookup table based nationalities (e.g. french farm etc.)
  # It first tries to get the spatial entities from a title with SpaCy, if any.
  # - If it succeeds, then it stop to search other spatial entities.
  # - Otherwise (i.e. if it finds nothing with SpaCy), it uses the lookup table strategy.
  # All extracted spatial entities from the titles are stored in 
  #  "out/event_preprocessing-results/padiweb/titles_final_locations.csv"
  #
  #
  # PARAMETERS
  # None (the variables used in this method are the class variables)
  ################################################################################################ 
def retrieve_disease_from_raw_sentence(sentence_text, disease_name):
  disease_list = []
  
  sentence_text = sentence_text.lower()
  clean_text = unicodedata.normalize("NFKD",sentence_text)

  # "hpai" and "lpai" are the first two elements in DISEASE_KEYWORDS_DICT >> intentionally
  disease_keywords = consts.DISEASE_KEYWORDS_DICT[disease_name].keys()
  #print(disease_keywords)
  for kw in disease_keywords:
    #print(kw)
    parts = kw.split(" ")
    kw_pattern = ' '.join([p+".{0,2}" for p in parts])
    # I dont add a space in the beginning, maybe a phrase can start with the keyword
    #kw_pattern = " "+ kw_pattern # to ensure that our keyword is not contained in a string >> the found string should start with our keyword
    res = re.findall(kw_pattern, clean_text)
    if len(res)>0:
      add = True
      # ---------------------------------------------------------------------------
      # multiple keywords can be found in a text (e.g. "avian" and "avian flu")
      # In this case, we need to return the long keyword (e.g. "avian flu" instead of "avian")
      for exist_kw in disease_list:
        if kw in exist_kw: # if a keyword is a substring of another keyword
          add = False
          break
        elif exist_kw in kw:
          disease_list.remove(exist_kw)
          break
      if add:
        disease_list.append(kw)
      # ---------------------------------------------------------------------------  
  if disease_name == user_consts.DISEASE_AVIAN_INFLUENZA:
    res = re.findall("h[0-9]{1,2}n[0-9]{1,2}", clean_text)
    res = [r.strip() for r in res]
    res2 = re.findall("h[0-9]{1,2}", clean_text)
    res2 = [r2.strip() for r2 in res2]
    res3 = re.findall("n[0-9]{1,2}", clean_text)
    res3 = [r2.strip() for r2 in res3]
    # an event must have a single AI subtype
    # for now, we do not treat complex sentences
    disease_list = disease_list + res + res2 + res3
    #disease_list = disease_list + res2
  
  disease_list = list(np.unique(disease_list))
  disease_type_list = [disease_name]*len(disease_list)
  return(disease_list, disease_type_list)


  
