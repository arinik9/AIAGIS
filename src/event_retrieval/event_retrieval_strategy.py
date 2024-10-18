'''
Created on Feb 18, 2022

@author: nejat
'''

from abc import ABC, abstractmethod
import pandas as pd

from typing import List
import numpy as np
from copy import deepcopy
import json
import dateutil.parser as parser

import src.consts as consts
from src.event.event import Event
from src.event.location import Location
from src.event.temporality import Temporality
from src.event.symptom import Symptom
from src.event.host import Host
from src.event.hosts import Hosts
from src.event.disease import Disease
from src.util_gis import retrieve_continent_from_country_code
from src.event.timex import convert_timex_str_to_datetime

from sutime import SUTime # temporal entity normalization

import re


class EventRetrievalStrategy(ABC):
  """
  The Strategy interface declares operations common to all supported versions
  of some algorithm.
  
  The Context uses this interface to call the algorithm defined by Concrete
  Strategies.
  """
  
    
  @abstractmethod
  def get_description(self) -> str:
      pass
  
  
  @abstractmethod
  def estimate(self, row:pd.Series, sentence_id_to_article_id:dict, df_initial_info_extr: pd.DataFrame, df_articles: pd.DataFrame, \
               geonames_ass:dict, sentence_id_to_text:dict, article_id_to_title:dict) -> List:
      pass
    
    
  
  




  
class EventRetrievalStrategyStructuredData(EventRetrievalStrategy):
  
  def get_description(self) -> str:
    return(consts.EVENT_RETRIEVAL_STRATEGY_STRUCTURED_DATA)


  def estimate(self, article_id:int, pub_date, df_norm_spatial_entities_by_article:pd.DataFrame, \
               df_norm_disease_entities_by_article:pd.DataFrame, df_norm_host_entities_by_article:pd.DataFrame) -> List:
    
      if df_norm_spatial_entities_by_article.shape[0]>0 and \
        df_norm_disease_entities_by_article.shape[0]>0 and \
        df_norm_host_entities_by_article.shape[0]>0:
        print("====== ENTERED !")
        
        df_norm_spatial_entities_by_article.reset_index(inplace=True)
        df_norm_disease_entities_by_article.reset_index(inplace=True)
        df_norm_host_entities_by_article.reset_index(inplace=True)
  
        # ===========================================================================================
        # PART 1: SPATIAL ENTITIES 
        # ===========================================================================================      
        print(df_norm_spatial_entities_by_article)
        name = df_norm_spatial_entities_by_article.loc[0,consts.KEY_GEONAMES_NAME]
        geoname_id = df_norm_spatial_entities_by_article.loc[0,consts.COL_GEONAMES_ID]
        geoname_json_str = df_norm_spatial_entities_by_article.loc[0,consts.COL_GEONAMS_JSON]
        geoname_json = json.loads(geoname_json_str)
        lat = geoname_json[consts.COL_LAT]
        lng = geoname_json[consts.COL_LNG]
        country_code = df_norm_spatial_entities_by_article.loc[0,consts.COL_COUNTRY_ALPHA3_CODE]
        continent = df_norm_spatial_entities_by_article.loc[0,consts.COL_LOC_CONTINENT]
        hierarchy_data = eval(df_norm_spatial_entities_by_article.loc[0,consts.COL_GEONAMES_HIERARCHY_INFO])
        loc = Location(name, geoname_id, geoname_json, lat, lng, country_code, continent, hierarchy_data)
        print(loc)
 
        # ===========================================================================================
        # PART 2: DISEASE ENTITIES 
        # ===========================================================================================
        dis_data = eval(df_norm_disease_entities_by_article.loc[0,consts.COL_DISEASE_ENTRY])
        d_serotype = dis_data["serotype"]
        d_subtype = dis_data["subtype"]
        d_type = dis_data["type"]
        d_pathogenicity = dis_data["pathogenicity"]
        disease_info = Disease(d_serotype, d_subtype, d_type, d_pathogenicity)
        print(disease_info)
        
        # ===========================================================================================
        # PART 3: HOST ENTITIES 
        # ===========================================================================================
        host_entry_list = eval(df_norm_host_entities_by_article.loc[0,consts.COL_HOST_ENTRY])
        host_list = [Host(host_entry) for host_entry in host_entry_list]
        hosts_info = Hosts(host_list)
        print(hosts_info)
        
        # ===========================================================================================
        # PART 4: SYMPTOM ENTITIES 
        #        We do nothing here, it is for the future
        # ===========================================================================================
        symptom_info = Symptom()

        # ===========================================================================================
        # PART 5: EVENT CREATION
        # ===========================================================================================
        e = Event(-1, article_id, "", "", loc, pub_date, disease_info, hosts_info, symptom_info, "", "")
        
        return e
      
   
   
   
   
    
class EventRetrievalStrategyRelevantSentence(EventRetrievalStrategy):
  
  def get_description(self) -> str:
    return(consts.EVENT_RETRIEVAL_STRATEGY_RELEVANT_SENTENCE)



  def rebuild_location_list_with_spatial_inclusion(self, location_list):
    geonameId_to_loc = {}
    for loc in location_list:
      geonameId_to_loc[loc.get_geoname_id()] = loc
      
    geonameId_list = [loc.get_geoname_id() for loc in location_list]
    unique_geonameId_list = np.unique(geonameId_list)
    
    location_list = [geonameId_to_loc[id] for id in unique_geonameId_list]
        
    geoname_ids_to_remove = []
    if len(location_list)>1:
      for i in range(len(location_list)):
        loc1 = location_list[i]
        for j in range(len(location_list)):
          loc2 = location_list[j]
          if i != j:
            if loc1.get_geoname_id() not in geoname_ids_to_remove and loc2.get_geoname_id() not in geoname_ids_to_remove:
              if loc1.is_spatially_included(loc2):
                geoname_ids_to_remove.append(loc2.get_geoname_id())
    #
    final_location_list = []
    for loc in location_list:
      if loc.get_geoname_id() not in geoname_ids_to_remove:
        final_location_list.append(loc)
    return final_location_list




  def merge_events_with_same_spatial_entity(self, event_list):
    #print("merge_events_with_same_spatial_entity")
    #print(event_list)
    first_event = event_list[0]
    other_event_list = event_list[1:]
    e_final = deepcopy(first_event)
    
    for e in other_event_list:
      # we suppose that there is not any inconsistency for hierarchy in all host info of 'e'
      if not e.is_identical(e_final):
      
        # 1) improve disease info 
        # if e.disease.is_hierarchically_included(e_final.disease):
        if e.disease.hierarchically_includes(e_final.disease):
          e_final.disease = e.disease
          e_final.sentences = e_final.sentences + "||" + e.sentences
        if e_final.disease.pathogenicity == "unknown-pathogenicity" and e.disease.pathogenicity != "unknown-pathogenicity":
          e_final.disease.pathogenicity = e.disease.pathogenicity
              
        # 2) improve host info
        # if e.host.is_hierarchically_included(e_final.host):
        #   e_final.host = e.host
        #   e_final.sentences = e_final.sentences + "||" + e.sentences
        host_to_remove = [] # we do not change 'host_info' and 'e.host' during the for iteration
        host_to_add = []
        new_host = []
        for h_new in e.host.get_entry():
          for h_exis in e_final.host.get_entry():
            # if host_info.is_hierarchically_included(e.host) or (list(host_info.get_entry().keys())[0] == "human" and list(host_info.get_entry().keys())[0] != list(e.host.get_entry().keys())[0]):
            if h_new.hierarchically_includes(h_exis):
              host_to_remove.append(h_exis)
              host_to_add.append(h_new)
            else:
              new_host.append(h_new)
        for h_exis in host_to_remove:
          e_final.host.remove_host_info(h_exis)
        for h_new in host_to_add:
          e_final.host.add_host_info_if_success(h_new)
        for h_new in new_host:
          e_final.host.add_host_info_if_success(h_new)
        # TODO: verify if there are any existing sentence id ?
        e_final.sentences = e_final.sentences + "||" + e.sentences
          
    return e_final  

          
  # we also take into account disease name and host type
  def handle_event_location_redundancy_and_spatial_inclusion(self, event_list):
    final_event_list = []
    #print("handle_event_location_redundancy_and_spatial_inclusion")
    #print(event_list)
    
    # first, regroup by geoname_id
    dict_by_geoname_id = {}
    for e in event_list:
      key = str(e.loc.get_geoname_id())+"_"+e.disease.d_type # +"_"+  ",".join([h.get_entry()["common_name"] for h in e.host.get_entry()])
      if key not in dict_by_geoname_id:
        dict_by_geoname_id[key] = []
      dict_by_geoname_id[key].append(e)
       
    # second, compress the event occurring in the same place
    #print("dict_by_geoname_id")
    #print(dict_by_geoname_id)
    for key in dict_by_geoname_id.keys():
      #print("-- key", key)
      e_list = dict_by_geoname_id[key]
      e = e_list[0]
      if len(e_list)>1:
        e = self.merge_events_with_same_spatial_entity(e_list)
      #print(e)
      final_event_list.append(e)
      
    final_event_list.sort(key=lambda e: e.loc.get_hierarchy_level(), reverse=True)
    #print("AFTER SORT")
    #print(final_event_list)
    
    # third, check spatial inclusion
    final_event_list2 = []
    event_ids_to_remove = [] # list of lists
    for i in range(len(final_event_list)):
      e_i = final_event_list[i]
      if e_i.e_id not in event_ids_to_remove:
        loc1 = e_i.loc
        for j in range(len(final_event_list)):
          if i != j:
            loc2 = final_event_list[j].loc
            if loc1.is_spatially_included(loc2) or loc1.get_geoname_id() == loc2.get_geoname_id():
              list_to_merge = [final_event_list[i], final_event_list[j]]
              e_i = self.merge_events_with_same_spatial_entity(list_to_merge) # deepcopy
              event_ids_to_remove.append(final_event_list[j].e_id)
      if e_i.e_id not in event_ids_to_remove:
        final_event_list2.append(e_i)
            
            
    #print("END handle_event_location_redundancy_and_spatial_inclusion")
    #print(final_event_list2)
    return(final_event_list2)
  
  
      
  # example of the "row" object. It is of type 'pd.Series'.
  # -----------------------------------
  # id_articleweb                                                       3WWDYJD6ZB
  # id                                                        [52751799, 52751815]
  # position                                                             [489, 43]
  # token_index                                                            [41, 9]
  # length                                                                [11, 11]
  # type                         ['outbreak-related-location', 'outbreak-relate...
  # label                                                   ['correct', 'correct']
  # value                                           ['South Korea', 'South Korea']
  # keyword_id                                                [154461.0, 154461.0]
  # from_automatic_extraction                                           [1.0, 1.0]
  # geonames_id                                             [1835841.0, 1835841.0]
  # country_feature_code                                          ['PCLI', 'PCLI']
  # sentence_id                                                 [9493873, 9493869]
  # s_classificationlabel_id                                    [[11, 12], [11, 12]]
  # country                                                            South Korea
  # country_freq                                                                 2
  # nb_country                                                                   1
  # Name: 0, dtype: object      
     
      
  def estimate(self, disease_name:str, df_sentences_by_article:pd.DataFrame, df_norm_spatial_entities_by_article:pd.DataFrame, \
               df_norm_disease_entities_by_article:pd.DataFrame, df_norm_host_entities_by_article:pd.DataFrame, \
                df_articles:pd.DataFrame, df_initial_info_extr: pd.DataFrame, \
                geonames_hierarchy_ass:dict) -> List:
    # first, assign input params to local variables
    article_id = df_norm_spatial_entities_by_article.iloc[0][consts.COL_ARTICLE_ID_RENAMED]
    article_info = df_articles[df_articles[consts.COL_ID] == article_id].iloc[0]
    self.disease_name = disease_name
    self.df_sentences_by_article = df_sentences_by_article
    self.df_norm_spatial_entities_by_article = df_norm_spatial_entities_by_article
    self.df_norm_disease_entities_by_article = df_norm_disease_entities_by_article
    self.df_norm_host_entities_by_article = df_norm_host_entities_by_article
    self.df_initial_info_extr = df_initial_info_extr
    self.geonames_hierarchy_ass = geonames_hierarchy_ass
    
    title = article_info[consts.COL_TITLE]
    published_time = article_info[consts.COL_PUBLISHED_TIME]
    t = Temporality(published_time)
    source = article_info[consts.COL_SOURCE]
    url = article_info[consts.COL_URL]
    
    print("------------------------!! estimate !!------------------------")
    SLIDING_WINDOW_LENGTH = 4 # 4 next sentences 
    sliding_window_for_history = [] # e.g. 
    # local_sentence_id_list = np.unique(self.df_sentences_by_article["local_sentence_id"].to_numpy())
    # #print(self.sentence_id_to_local_sentence_id, sliding_window_for_history)
    # print("local_sentence_id_list", local_sentence_id_list)
    
    final_event_candidates = []
    #local_sentence_id_list = [s_id for s_id in local_sentence_id_list if s_id != -1] # -1 from those coming from title
    for index, row in df_sentences_by_article.iterrows():
      local_sentence_id = row["local_sentence_id"]
      sentence_id = row["id"]
      if local_sentence_id != -1:
        sentence_text = row["text"]
        print("----- local sentence id", local_sentence_id, sentence_text)
        df_norm_spatial_entities_by_sentence = \
                  self.df_norm_spatial_entities_by_article[\
                                       self.df_norm_spatial_entities_by_article["local_sentence_id"] == local_sentence_id]
        df_norm_disease_entities_by_sentence = \
                  self.df_norm_disease_entities_by_article[\
                                       self.df_norm_disease_entities_by_article["local_sentence_id"] == local_sentence_id]
        df_norm_host_entities_by_sentence = \
                  self.df_norm_host_entities_by_article[\
                                       self.df_norm_host_entities_by_article["local_sentence_id"] == local_sentence_id]
                  
        # if a new spatial entity is found, the enter
        # later, we chek other entities
        if df_norm_spatial_entities_by_sentence.shape[0]>0:
          print("====== ENTERED !")
          
          #print(df_extr_spatial_entities_by_sentence["value"])
          if len(sliding_window_for_history)>0: 
            # remove the old entries based on time limit
            completed_event_candidates = [entry[1] for entry in sliding_window_for_history if (local_sentence_id-entry[0])>=SLIDING_WINDOW_LENGTH]
            sliding_window_for_history = [entry for entry in sliding_window_for_history if (local_sentence_id-entry[0])<SLIDING_WINDOW_LENGTH]
            final_event_candidates.extend(completed_event_candidates)
                    
          if not self.contains_ban_related_keyword(sentence_text):
            
            #print("-- no ban related keyword")
            print("BEFORE sliding_window_for_history")
            print(sliding_window_for_history)
            
            print(df_norm_spatial_entities_by_sentence[["value", "id_articleweb", "local_sentence_id"]])

            has_disease_keywords = True if df_norm_disease_entities_by_sentence.shape[0]>0 else False
            has_host = True if df_norm_host_entities_by_sentence.shape[0]>0 else False
          
            disease_info = None
            if has_disease_keywords:
              disease_info_list = []
              for index, row in df_norm_disease_entities_by_sentence.iterrows():
                #"disease_text"
                dis_data = eval(row["disease"])
                d_serotype = dis_data["serotype"]
                d_subtype = dis_data["subtype"]
                d_type = dis_data["type"]
                d_pathogenicity = dis_data["pathogenicity"]
                disease_info = Disease(d_serotype, d_subtype, d_type, d_pathogenicity)
                disease_info_list.append(disease_info)
                # we keep only the first disease info, otherwise it would be complicated
              disease_info = disease_info_list[0]
              print(disease_info)
            
            hosts_info = None
            if has_host:
              host_info_list = []
              for index, row in df_norm_host_entities_by_sentence.iterrows():
                #"disease_text"
                host_hier = eval(row["host_hierarchy"])
                host_id = row["host_id"]
                host_type = row["host_type"]
                host_common_name = row["host_text"]
                host_data = {"id": host_id, "type": host_type, "common_name": host_common_name, "level":len(host_hier), "hierarchy": host_hier}
                host_info = Host(host_data)
                host_info_list.append(host_info)
              hosts_info = Hosts(host_info_list)            
              print(hosts_info)
            
            symptom_info = Symptom()
            #print(host_info.get_entry().keys())
            #contains_forbidden_hosts = np.any([True if h not in consts.DISEASE_HOST_RELATION[self.disease_name] else False \
            #                          for h in host_info.get_entry().keys()])
            contains_forbidden_hosts = False
            #print("contains_forbidden_hosts", contains_forbidden_hosts)
            #print("has_disease_keywords", has_disease_keywords)
            has_new_event_info = (has_disease_keywords and has_host and not contains_forbidden_hosts)
            #print("has_new_event_info", has_new_event_info)  
            elaborate_existing_events = ((disease_info is None or has_disease_keywords) \
                                          and not contains_forbidden_hosts and len(sliding_window_for_history)>0)
            print("elaborate_existing_events", elaborate_existing_events)
            if elaborate_existing_events:
              print("---- update with elaborate_existing_events")
              sliding_window_for_history = self.update_existing_event_candidates_from_sentence(sentence_id, local_sentence_id,
                                                                   sliding_window_for_history, \
                                                                  df_norm_spatial_entities_by_sentence, \
                                                                  disease_info, hosts_info, symptom_info)
              print("AFTER sliding_window_for_history")
              print(sliding_window_for_history)
              
            if has_new_event_info:
              print("---- new")
              print(sentence_text)
              print(df_norm_spatial_entities_by_sentence["value"])
              event_candidates_by_sentences, has_old_date = self.retrieve_new_event_candidates_from_sentence(article_id, title, published_time, \
                                                               sentence_id, local_sentence_id, t, disease_name, source, url,
                                                               sentence_text, df_norm_spatial_entities_by_sentence, \
                                                               disease_info, hosts_info, symptom_info)
              if has_old_date:
                sliding_window_for_history = [] # reset
              else:
                for e in event_candidates_by_sentences:
                  print(e)
                  sliding_window_for_history.append((local_sentence_id, e))
            
    print("END sliding_window_for_history")
    print(sliding_window_for_history)
    if len(sliding_window_for_history)>0:
      completed_event_candidates = [entry[1] for entry in sliding_window_for_history]
      final_event_candidates.extend(completed_event_candidates)
    # finally, check redundancy, if so, compress them        
    #print("before handle_event_location_redundancy_and_spatial_inclusion")
    #print(final_event_candidates)
    final_event_candidates2 = self.handle_event_location_redundancy_and_spatial_inclusion(final_event_candidates)
    #print("FINAL")
    #print(final_event_candidates2)
    return final_event_candidates2



  def retrieve_new_event_candidates_from_sentence(self, article_id, title, published_time, sentence_id, local_sentence_id , \
                                                  t, disease_name, source, url,
                                                  sentence_text, df_extr_spatial_entities_by_sentence,
                                                   disease_info, host_info, symptom_info):
    event_candidates = []
    # --------------------------------------------------------
    #has_annual_statement = False
    has_annual_statement = ("this year so far" in sentence_text.lower()) or ("so far this year" in sentence_text.lower())
    # sutime source: https://nlp.stanford.edu/software/sutime.html
    # annotation guideline on Timex: https://timeml.github.io/site/publications/timeMLdocs/annguide_1.2.1.pdf
    #print(sentence_text)
    # converting timex format to datetime: https://stackoverflow.com/questions/55066094/timex-strings-to-datetime-in-python
    pub_date = parser.parse(published_time)
    sutime = SUTime(mark_time_ranges=False, include_range=False)
    res_list = sutime.parse(sentence_text, published_time)
    has_old_date = False
    #print(published_time)
    #print(res_list)
    res_list_date = [ent for ent in res_list if 'timex-value' in ent and ent['type'] == 'DATE']
    timex_str_list = [ent['timex-value'] for ent in res_list_date]
    # {'timex-value': '2022-09-14', 'start': 18, 'end': 26, 'text': 'tomorrow', 'type': 'DATE', 'value': '2022-09-14'}
    found_dates = convert_timex_str_to_datetime(timex_str_list)
    #print("found_dates", found_dates)
    for found_date in found_dates:
      # diff_days = (pub_date-found_date).days
      #print(abs(pub_date.day-found_date.day))
      #if abs(pub_date.day-found_date.day) > 360: # 1 year of difference >> to be sure that it is an old date
      if abs((pub_date-found_date).days) > 30: # 1 year of difference >> to be sure that it is an old date
        has_old_date = True
        #print("has_old_date")
        break
    # --------------------------------------------------------
    
    if not has_old_date and not has_annual_statement:
      spatial_entity_list = df_extr_spatial_entities_by_sentence["value"].to_list()
      #country_list = df_extr_spatial_entities_by_sentence["country_code"].to_list()
      #spatial_entity_list = [spatial_entity_list[i]+" ("+country_list[i]+")" for i in range(len(spatial_entity_list))]
      #print(spatial_entity_list)
      location_list = [] # TODO: we can regroup locations by country
      for index, row in df_extr_spatial_entities_by_sentence.iterrows():
        name = row[consts.COL_VALUE]
        geoname_id = row[consts.COL_GEONAMES_ID]
        geoname_json_str = row[consts.COL_GEONAMS_JSON]
        geoname_json = json.loads(geoname_json_str)
        lat = geoname_json["lat"]
        lng = geoname_json["lng"]
        fcode = row[consts.COL_FEATURE_CODE]
        country_code = row["country_code"]
        continent = retrieve_continent_from_country_code(country_code)
        hierarchy_data = eval(self.geonames_hierarchy_ass[geoname_id]) # it converts into a list
        
        loc = Location(name, geoname_id, geoname_json, lat, lng, country_code, continent, hierarchy_data)
        location_list.append(loc)
        
      #print(location_list)
      final_location_list = self.rebuild_location_list_with_spatial_inclusion(location_list)
      print(final_location_list)
      # create an event object for each location
  
      sentences = str(sentence_id) + "("+str(local_sentence_id)+")"
      event_candidates = []
      for loc in final_location_list:
        if host_info is None:
          host_info = Host()
        e = Event(-1, article_id, url, source, loc, t, disease_info, host_info, symptom_info, title, sentences)
        event_candidates.append(e)
        
    return event_candidates, has_old_date
  
  

  def update_existing_event_candidates_from_sentence(self, sentence_id, local_sentence_id, sliding_window_for_history, \
                                                              df_extr_spatial_entities_by_sentence, \
                                                              disease_info, hosts_info, symptom_info):
    # TODO: check if spatialy inclusion does not exist
    
    # 1) improve locations 
    ids_to_remove = []
    for index, row in df_extr_spatial_entities_by_sentence.iterrows():
      name = row[consts.COL_VALUE]
      geoname_id = row[consts.COL_GEONAMES_ID]
      geoname_json_str = row[consts.COL_GEONAMS_JSON]
      # JSON only allows enclosing strings with double quotes 
      #p = re.compile('(?<!\\\\)\'') # source: https://stackoverflow.com/questions/39491420/python-jsonexpecting-property-name-enclosed-in-double-quotes
      #geoname_json_str = p.sub('\"', geoname_json_str)
      geoname_json = json.loads(geoname_json_str)
      lat = geoname_json["lat"]
      lng = geoname_json["lng"]
      fcode = row[consts.COL_FEATURE_CODE]
      country_code = row["country_code"]
      continent = retrieve_continent_from_country_code(country_code)
      hierarchy_data = eval(self.geonames_hierarchy_ass[geoname_id])
      
      loc = Location(name, geoname_id, geoname_json, lat, lng, country_code, continent, hierarchy_data)
      # check if loc is an improvement w.r.t existing ones
      
      entries_to_add = []
      for (id, e) in sliding_window_for_history:
        if loc.is_spatially_included(e.loc):
          e2 = Event(-1, e.article_id, e.url, e.source, loc, e.date, e.disease, e.host, e.symptom, e.title, e.sentences)
          if str(sentence_id) not in e.sentences:
            e2.sentences = e2.sentences + ", " + str(sentence_id) + "("+str(local_sentence_id)+")"
          entries_to_add.append((id, e2))
          if e.loc.get_geoname_id() not in ids_to_remove:
            ids_to_remove.append(e.loc.get_geoname_id())
      sliding_window_for_history.extend(entries_to_add)
    sliding_window_for_history = [(id, e) for (id, e) in sliding_window_for_history if e.loc.get_geoname_id() not in ids_to_remove] 
          
    # 2) improve disease info in terms of hierarchy
    if disease_info is not None:
      for (id, e) in sliding_window_for_history:
        #if disease_info.is_hierarchically_included(e.disease):
        if disease_info.hierarchically_includes(e.disease): # i.e. if disease_info is more specific than e.disease
          e.disease = disease_info
          if str(sentence_id) not in e.sentences:
            e.sentences = e.sentences + ", " + str(sentence_id) + "("+str(local_sentence_id)+")"
            
    # 3) improve host info in terms of hierarchy
    if hosts_info is not None: # not host_info.is_host_info_empty():
      for (id, e) in sliding_window_for_history:
        # for each event, there can be multiple host candidates
        host_to_remove = [] # we do not change 'host_info' and 'e.host' during the for iteration
        host_to_add = []
        for h_new in hosts_info.get_entry():
          for h_exis in e.host.get_entry():
            # if host_info.is_hierarchically_included(e.host) or (list(host_info.get_entry().keys())[0] == "human" and list(host_info.get_entry().keys())[0] != list(e.host.get_entry().keys())[0]):
            if h_new.hierarchically_includes(h_exis):
              host_to_remove.append(h_exis)
              host_to_add.append(h_new)
        for h_exis in host_to_remove:
          e.host.remove_host_info(h_exis)
        for h_new in host_to_add:
          e.host.add_host_info_if_success(h_new)
        if str(sentence_id) not in e.sentences:
          e.sentences = e.sentences + ", " + str(sentence_id) + "("+str(local_sentence_id)+")"
          
    return sliding_window_for_history
  
  
  def contains_ban_related_keyword(self, text):
    ban = False
    for kw in consts.BAN_KEYWORDS_LIST:
      parts = kw.split(" ")
      kw_pattern = ' '.join([" "+p+".{0,2}" for p in parts])
      if len(re.findall(kw_pattern, text))>0:
        ban = True
        break
    return ban  
