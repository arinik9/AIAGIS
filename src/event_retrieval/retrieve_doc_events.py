'''
Created on Nov 16, 2021

@author: nejat
'''


import os
import pandas as pd
import numpy as np
import csv
from abc import ABC, abstractmethod

import src.util as util
import src.consts as consts
from src.event_retrieval.event_retrieval_strategy import EventRetrievalStrategy

from src.util_event import get_df_from_events

import re



class AbstractEventRetrieval(ABC):

  def __init__(self, event_retrieval_strategy: EventRetrievalStrategy):
    print("--- using sentence identification strategy: ", event_retrieval_strategy.get_description())
    self.event_retrieval_strategy = event_retrieval_strategy
    
    self.out_events_filename = "doc_events" + "_task1=" +  self.event_retrieval_strategy.get_description() + "." + consts.FILE_FORMAT_CSV
    
    
  @abstractmethod
  def read_csv_files_as_dataframe(self):
    pass
  
    
  @abstractmethod
  def retrieve_doc_events(self):
    pass  
  
    
  def perform_event_retrieval(self, disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder):
    print("...... beginning of event retrieval for " + disease_name + "!")
    self.disease_name = disease_name
    self.in_preprocessing_result_folder = in_preprocessing_result_folder
    self.in_norm_result_folder = in_norm_result_folder
    self.data_folder = data_folder
    self.out_doc_events_folder = out_doc_events_folder
    
    self.read_csv_files_as_dataframe()
    
    events = self.retrieve_doc_events(disease_name)
    df_events = get_df_from_events(events)
    
    if isinstance(df_events, int) and df_events == -1:
      return;
    
    if len(df_events)>0:
      # --------------------------------
      # add time relate columns
      dates = pd.to_datetime(df_events[consts.COL_PUBLISHED_TIME]).to_list()
      day_no_list, week_no_list, biweek_no_list, month_no_list, year_list, season_list = util.retrieve_time_related_info(dates)
      df_events["day_no"] = day_no_list
      df_events["week_no"] = week_no_list
      df_events["biweek_no"] = biweek_no_list
      df_events["month_no"] = month_no_list
      df_events["year"] = year_list
      df_events["season"] = season_list
      # --------------------------------

      out_events_filepath = os.path.join(out_doc_events_folder, self.out_events_filename)
      df_events.to_csv(out_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
    print("...... end of event retrieval !")







class EventRetrievalPadiweb(AbstractEventRetrieval):
# for each selected article, find the corresponding sentence ids
# identify the sentence id describing an event
# find keywords based on this sentence id
# construct the tuple
# try to find previous and next sentences to complete the tuple

# a_id = 'ZW76AEWEPX'
# s_indx = S_agg_reverted_indices[a_id] 
# s = S_agg[s_indx:s_indx+1]


# possible entity type:
# - keyword (e.g. "case", "virus", "Japan")
# - outbreak-related-location
# - GPE (e.g. "Hokkaido")
# - ORG
# - disease
# - host
# - outbreak-related-date (e.g. "December")
# - TIME (e.g. "Monday evening") >> no entry for Avian Influenza
# - DATE (e.g. "Monday", "Monday, November 1") >> no entry for Avian Influenza
# - cases (e.g. "23,000")
# - NORP (e.g. "Italian")
# - LOC (e.g. "Hokkaido East")
# - symptom (e.g. "dead")
# - (TODO) published time

# TODO https://datascience.stackexchange.com/questions/12575/similarity-between-two-words

# https://nlp.stanford.edu/software/sutime.shtml
# https://stanfordnlp.github.io/CoreNLP/sutime.html
# https://stanfordnlp.github.io/CoreNLP/pipeline.html
# Heideltime


# A-ADM1, i.e. première division administrative (e.g. région française, état aux USA
# la classe P-PPL, associée le plus souvent à des petites villes, villages, etc.

# The order of the PPL* features goes like : PPLC > PPLA[2-4] > PPL > PPLX > PPLL



  def __init__(self, strategy: EventRetrievalStrategy):
    super().__init__(strategy)
    self.out_events_filename = "doc_events_" + consts.NEWS_SURVEILLANCE_PADIWEB + "_task1=" +  self.event_retrieval_strategy.get_description() + "." + consts.FILE_FORMAT_CSV
    self.input_raw_events_filename = "raw_events_"+consts.NEWS_SURVEILLANCE_PADIWEB
    self.input_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
    self.input_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
    self.input_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
            

  def read_csv_files_as_dataframe(self):
    articles_filepath = os.path.join(self.in_preprocessing_result_folder, \
                           self.input_raw_events_filename + "." + consts.FILE_FORMAT_CSV)
    cols_articles = [consts.COL_ID, consts.COL_TITLE, consts.COL_URL, consts.COL_SOURCE, \
                      consts.COL_PUBLISHED_TIME]
    # if signal_info_exists:
    #   cols_articles.append(consts.COL_SIGNAL_ID)
    self.df_articles = pd.read_csv(articles_filepath, usecols=cols_articles, sep=";", keep_default_na=False)
    
    sentences_filepath = os.path.join(self.in_preprocessing_result_folder, \
                           consts.PADIWEB_EXT_SENTENCES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    cols_sentences = [consts.COL_ID,consts.COL_TEXT,consts.COL_START,consts.COL_END, \
                      consts.COL_ARTICLE_ID, consts.COL_LANG, consts.COL_SENTENCE_CLASSIF_LABEL_ID, "local_sentence_id"]
    self.df_sentences = pd.read_csv(sentences_filepath, usecols=cols_sentences, sep=";", keep_default_na=False)
    
    norm_spatial_entities_filepath = os.path.join(self.in_norm_result_folder, \
                            self.input_norm_spatial_entities_filename + "." + consts.FILE_FORMAT_CSV)
    cols_extr_spatial_entities = [consts.COL_ID, consts.COl_POS, consts.COl_TOKEN_INDEX, consts.COl_LENGTH, \
                      consts.COL_TYPE, consts.COL_LABEL, consts.COL_VALUE, \
                      consts.COL_ARTICLE_ID_RENAMED, consts.COL_KEYW_ID, consts.COL_FROM_AUTO_EXTR, \
                      consts.COL_GEONAMES_ID, consts.COL_GEONAMS_JSON, consts.COL_SENTENCE_ID, \
                      consts.COL_FEATURE_CODE, consts.COL_SENTENCE_CLASSIF_LABEL_ID, "local_sentence_id", \
                      "country_code", "updated", "ref_countries_for_geocoding"] 
    self.df_norm_spatial_entities = pd.read_csv(norm_spatial_entities_filepath, usecols=cols_extr_spatial_entities, sep=";", keep_default_na=False)
    #self.df_extr_spatial_entities[consts.COL_SENTENCE_ID] = self.df_extr_spatial_entities[consts.COL_SENTENCE_ID].astype(int)
    #self.df_extr_spatial_entities["local_sentence_id"] = self.df_extr_spatial_entities["local_sentence_id"].astype(int)

    norm_disease_entities_filepath = os.path.join(self.in_norm_result_folder, \
                            self.input_norm_disease_entities_filename + "." + consts.FILE_FORMAT_CSV)    
    self.df_norm_disease_entities = pd.read_csv(norm_disease_entities_filepath, sep=";", keep_default_na=False)
    
    norm_host_entities_filepath = os.path.join(self.in_norm_result_folder, \
                            self.input_norm_host_entities_filename + "." + consts.FILE_FORMAT_CSV)
    self.df_norm_host_entities = pd.read_csv(norm_host_entities_filepath, sep=";", keep_default_na=False)
    

    geonames_hierarchy_filepath = os.path.join(self.in_norm_result_folder, \
                            consts.GEONAMES_HIERARCHY_INFO_FILENAME + "." + consts.FILE_FORMAT_CSV)
    cols_geonames_hierarcy = [consts.COL_GEONAMES_ID, "hierarchy_name", "hierarchy_geoname_id"] 
    self.df_geonames_hierarchy = pd.read_csv(geonames_hierarchy_filepath, usecols=cols_geonames_hierarcy, sep=";", keep_default_na=False)

    # keywords_filepath = os.path.join(self.in_preprocessing_result_folder, \
    #                    consts.PADIWEB_KEYWORD_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    # cols_keywords = [consts.COL_ID,consts.COL_NAME,consts.COL_LANG,consts.COL_KEYW_TYPE_ID]
    # self.df_keywords = pd.read_csv(keywords_filepath, usecols=cols_keywords, sep=";", keep_default_na=False)
    #
    # keywords_align_filepath = os.path.join(self.in_preprocessing_result_folder, \
    #                      consts.PADIWEB_KEYW_ALIGN_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    # cols_keywords_align = [consts.COL_ID, consts.COL_ALIGN_KEYW_ID, consts.COL_KEYW_ID]
    # self.df_keywords_align = pd.read_csv(keywords_align_filepath, usecols=cols_keywords_align, sep=";", keep_default_na=False)
    #

    # this one is needed for geonames json
    intitial_info_extr_filepath = os.path.join(self.in_preprocessing_result_folder, \
                        consts.PADIWEB_EXT_INFO_EXTR_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    cols_intitial_info_extr = [consts.COL_ID, consts.COl_POS, consts.COl_TOKEN_INDEX, consts.COl_LENGTH, \
                      consts.COL_TYPE, consts.COL_LABEL, consts.COL_VALUE, \
                      consts.COL_ARTICLE_ID_RENAMED, consts.COL_KEYW_ID, consts.COL_FROM_AUTO_EXTR, \
                      consts.COL_SENTENCE_ID, consts.COL_GEONAMES_ID, consts.COL_GEONAMS_JSON] 
    self.df_initial_info_extr = pd.read_csv(intitial_info_extr_filepath, usecols=cols_intitial_info_extr, sep=";", keep_default_na=False)



  def perform_event_retrieval(self, disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder):
    super().perform_event_retrieval(disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder)



  def retrieve_doc_events(self, disease_name):
    all_doc_events = []
    
    self.sentence_id_to_article_id = dict(zip(self.df_sentences[consts.COL_ID], self.df_sentences[consts.COL_ARTICLE_ID]))
    self.sentence_id_to_text = dict(zip(self.df_sentences[consts.COL_ID], self.df_sentences[consts.COL_TEXT]))
    self.article_id_to_title = dict(zip(self.df_articles[consts.COL_ID], self.df_articles[consts.COL_TITLE]))
    self.sentence_id_to_local_sentence_id = dict(zip(self.df_sentences[consts.COL_ID], self.df_sentences["local_sentence_id"]))
    self.geonames_hierarchy_ass = dict(zip(self.df_geonames_hierarchy[consts.COL_GEONAMES_ID], self.df_geonames_hierarchy["hierarchy_geoname_id"]))
   
    
    article_id_list = np.unique(self.df_norm_spatial_entities[consts.COL_ARTICLE_ID_RENAMED])

    #article_id_list = ['0704MFBINF', '0AIB1VZESR']
    #article_id_list = ["7QZJNVLHTS", "L7CG25TT3N", "RA419Z60HU", "JMO1JRJDJW", "5PF9FM6U2Q", "MDQA110VY0", "4SJJ0Y1X0D", "WAYOPZHA5J", "03WD3TAW5C", "02TL7DYIWC", "3UFCKR6BOV", "3TB9FO1ENQ", "3Z3PDHGCJX", "7V4FWGZU0D", "3SJZPBCWT8"]
    #article_id_list = ['0704MFBINF', '0AIB1VZESR', '0GJ5LF54J6', '0O87F8LCK8', '0PDPT5PYS9', '0Q67XKFHXV', '0QI0DU2GNU', '0R0N5UMR7V', '0T726GLBEA']
    relevant_article_id_list = self.remove_ban_related_articles(article_id_list)
    #relevant_article_id_list = ['A86TY2VKFO', 'P6CXS552R4', '4TP9S9MRHD', '7CGBSW6XVD', 'HO377ZC5Q8', 'O0ZSJHTLRX', '5RZ4N0CHCV', 'L2056OXVZM', 'CKYEHIHDUG', '7CGBSW6XVD', '7WWPUS79DA', '5YLQ4E5MI0', 'C1XP44ULHW', 'O1VMLG1RGK', 'GQ53J3P5SL', 'TD76M6JV83', 'SQ3ZUSBNSW', '4QWA91MBT6', 'DEPVU4GMA7']
    
    for indx, article_id in enumerate(relevant_article_id_list):
      print("----------- ", indx, article_id)
      #time.sleep(0.5)
      
      
      df_sentences_by_article = \
                self.df_sentences[self.df_sentences[consts.COL_ARTICLE_ID] == article_id]
      
      df_norm_spatial_entities_by_article = \
                self.df_norm_spatial_entities[self.df_norm_spatial_entities[consts.COL_ARTICLE_ID_RENAMED] == article_id]
                
      df_norm_disease_entities_by_article = \
            self.df_norm_disease_entities[self.df_norm_disease_entities[consts.COL_ARTICLE_ID] == article_id]
            
      df_norm_host_entities_by_article = \
            self.df_norm_host_entities[self.df_norm_host_entities[consts.COL_ARTICLE_ID] == article_id]
       
      #print(df_extr_spatial_entities_by_article)
      if df_norm_spatial_entities_by_article.shape[0]>0:
        # =================================================================
        # EVENT RETRIEVAL STRATEGY
        
        curr_doc_events = self.event_retrieval_strategy.estimate(disease_name, df_sentences_by_article, \
                                                    df_norm_spatial_entities_by_article, \
                                                    df_norm_disease_entities_by_article, df_norm_host_entities_by_article, \
                                                    self.df_articles, self.df_initial_info_extr, \
                                                    self.geonames_hierarchy_ass)
        print(curr_doc_events)
        # =================================================================
        #print("--------------------------------- We extracted ", len(curr_doc_events), " doc events on "+disease_name+".")
        if curr_doc_events is not None and len(curr_doc_events) > 0:
            all_doc_events.extend(curr_doc_events)
    
    # df_selected_sentence_ids = pd.DataFrame(sentence_ids_list, columns =[consts.COL_SENTENCE_ID])
    # df_selected_sentence_ids[consts.COL_ARTICLE_ID_RENAMED] = self.df_info_extr_single_event_est[consts.COL_ARTICLE_ID_RENAMED]
    # selected_sentence_ids_filepath = os.path.join(self.out_csv_folder, \
    #                         consts.PADIWEB_SELECTED_SENTENCE_IDS_FILENAME + "." + consts.FILE_FORMAT_CSV)
    # df_selected_sentence_ids.to_csv(selected_sentence_ids_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
    return(all_doc_events)
  
  
  def contains_ban_related_keyword(self, text):
    ban = False
    for kw in consts.BAN_KEYWORDS_LIST:
      parts = kw.split(" ")
      kw_pattern = ' '.join([" "+p+".{0,2}" for p in parts])
      if len(re.findall(kw_pattern, text))>0:
        ban = True
        break
    return ban
  
    
  # it also verifies if an article id is in df_articles or not >> some inconsistency is possible
  def remove_ban_related_articles(self, article_id_list):
    relevant_article_id_list = []
    for article_id in article_id_list:
      if article_id in self.article_id_to_title:
        title = self.article_id_to_title[article_id]
        if not self.contains_ban_related_keyword(title):
          relevant_article_id_list.append(article_id)
    return relevant_article_id_list
        
      



class EventRetrievalEmpresi(AbstractEventRetrieval):

  def __init__(self, strategy: EventRetrievalStrategy):
    super().__init__(strategy)
    # input
    self.input_raw_events_filename = "raw_events_"+consts.NEWS_DB_EMPRESS_I
    self.input_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_DB_EMPRESS_I
    self.input_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_DB_EMPRESS_I
    self.input_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_DB_EMPRESS_I
    # output
    self.out_events_filename = "doc_events_" + consts.NEWS_DB_EMPRESS_I + "_task1=" +  self.event_retrieval_strategy.get_description() + "." + consts.FILE_FORMAT_CSV
    
   

  def read_csv_files_as_dataframe(self):
    articles_filepath = os.path.join(self.in_preprocessing_result_folder, \
                           self.input_raw_events_filename + "." + consts.FILE_FORMAT_CSV)
    self.df_articles = pd.read_csv(articles_filepath, sep=";", keep_default_na=False)
    
    norm_spatial_entities_filepath = os.path.join(self.in_norm_result_folder, \
                            self.input_norm_spatial_entities_filename + "." + consts.FILE_FORMAT_CSV)
    self.df_norm_spatial_entities = pd.read_csv(norm_spatial_entities_filepath, sep=";", keep_default_na=False)

    norm_disease_entities_filepath = os.path.join(self.in_norm_result_folder, \
                            self.input_norm_disease_entities_filename + "." + consts.FILE_FORMAT_CSV)    
    self.df_norm_disease_entities = pd.read_csv(norm_disease_entities_filepath, sep=";", keep_default_na=False)
    
    norm_host_entities_filepath = os.path.join(self.in_norm_result_folder, \
                            self.input_norm_host_entities_filename + "." + consts.FILE_FORMAT_CSV)
    self.df_norm_host_entities = pd.read_csv(norm_host_entities_filepath, sep=";", keep_default_na=False)
    
     

  def perform_event_retrieval(self, disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder):
    super().perform_event_retrieval(disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder)


  def retrieve_doc_events(self, disease_name):
    all_events = []
    
    #article_id_list = np.unique(self.df_articles[consts.COL_ID])
    for indx, row in self.df_articles.iterrows():
      article_id = row[consts.COL_ARTICLE_ID]
      t = row[consts.COL_PUBLISHED_TIME]
      print("----------- ", indx, article_id, t)
      
      df_norm_spatial_entities_by_article = \
                self.df_norm_spatial_entities[self.df_norm_spatial_entities[consts.COL_ARTICLE_ID] == article_id]
                
      df_norm_disease_entities_by_article = \
            self.df_norm_disease_entities[self.df_norm_disease_entities[consts.COL_ARTICLE_ID] == article_id]
            
      df_norm_host_entities_by_article = \
            self.df_norm_host_entities[self.df_norm_host_entities[consts.COL_ARTICLE_ID] == article_id]
       
      if df_norm_spatial_entities_by_article.shape[0]>0:
        # =================================================================
        # EVENT RETRIEVAL STRATEGY
        
        curr_event = self.event_retrieval_strategy.estimate(article_id, t, \
                                                    df_norm_spatial_entities_by_article, \
                                                    df_norm_disease_entities_by_article, \
                                                    df_norm_host_entities_by_article)
        # =================================================================
        #print("--------------------------------- We extracted: ", curr_event)
        if curr_event is not None:
            all_events.append(curr_event)
    
    return(all_events)
    
    
    
    
class EventRetrievalWahis(EventRetrievalEmpresi):

  def __init__(self, strategy: EventRetrievalStrategy):
    super().__init__(strategy)
    # input
    self.input_raw_events_filename = "raw_events_"+consts.NEWS_DB_WAHIS
    self.input_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_DB_WAHIS
    self.input_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_DB_WAHIS
    self.input_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_DB_WAHIS
    # output
    self.out_events_filename = "doc_events_" + consts.NEWS_DB_WAHIS + "_task1=" +  self.event_retrieval_strategy.get_description() + "." + consts.FILE_FORMAT_CSV
    
    
  def read_csv_files_as_dataframe(self):
    super().read_csv_files_as_dataframe()
    
     
  def perform_event_retrieval(self, disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder):
    super().perform_event_retrieval(disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder)


  def retrieve_doc_events(self, disease_name):
    return super().retrieve_doc_events(disease_name)
    
    
    
class EventRetrievalPromed(EventRetrievalEmpresi):

  def __init__(self, strategy: EventRetrievalStrategy):
    super().__init__(strategy)
    # input
    self.input_raw_events_filename = "raw_events_"+consts.NEWS_SURVEILLANCE_PROMED
    self.input_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_SURVEILLANCE_PROMED
    self.input_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_SURVEILLANCE_PROMED
    self.input_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_SURVEILLANCE_PROMED
    # output
    self.out_events_filename = "doc_events_" + consts.NEWS_SURVEILLANCE_PROMED + "_task1=" +  self.event_retrieval_strategy.get_description() + "." + consts.FILE_FORMAT_CSV
    
    
  def read_csv_files_as_dataframe(self):
    super().read_csv_files_as_dataframe()
    
     
  def perform_event_retrieval(self, disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder):
    super().perform_event_retrieval(disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder)


  def retrieve_doc_events(self, disease_name):
    return super().retrieve_doc_events(disease_name)
    
    
    
class EventRetrievalApha(EventRetrievalEmpresi):

  def __init__(self, strategy: EventRetrievalStrategy):
    super().__init__(strategy)
    # input
    self.input_raw_events_filename = "raw_events_"+consts.NEWS_DB_APHA
    self.input_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_DB_APHA
    self.input_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_DB_APHA
    self.input_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_DB_APHA
    # output
    self.out_events_filename = "doc_events_" + consts.NEWS_DB_APHA + "_task1=" +  self.event_retrieval_strategy.get_description() + "." + consts.FILE_FORMAT_CSV


  def read_csv_files_as_dataframe(self):
    super().read_csv_files_as_dataframe()
    
     
  def perform_event_retrieval(self, disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder):
    super().perform_event_retrieval(disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder)


  def retrieve_doc_events(self, disease_name):
    return super().retrieve_doc_events(disease_name)
    
        
    
class EventRetrievalAphis(EventRetrievalEmpresi):

  def __init__(self, strategy: EventRetrievalStrategy):
    super().__init__(strategy)
    # input
    self.input_raw_events_filename = "raw_events_"+consts.NEWS_DB_APHIS
    self.input_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_DB_APHIS
    self.input_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_DB_APHIS
    self.input_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_DB_APHIS
    # output
    self.out_events_filename = "doc_events_" + consts.NEWS_DB_APHIS + "_task1=" +  self.event_retrieval_strategy.get_description() + "." + consts.FILE_FORMAT_CSV
    
    
  def read_csv_files_as_dataframe(self):
    super().read_csv_files_as_dataframe()
    
     
  def perform_event_retrieval(self, disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder):
    super().perform_event_retrieval(disease_name, in_preprocessing_result_folder, in_norm_result_folder, out_doc_events_folder, data_folder)


  def retrieve_doc_events(self, disease_name):
    return super().retrieve_doc_events(disease_name)




class EventRetrievalBVBRC(EventRetrievalEmpresi):

  def __init__(self, strategy: EventRetrievalStrategy):
    super().__init__(strategy)
    # input
    self.input_raw_events_filename = "raw_events_" + consts.NEWS_DB_BVBRC
    self.input_norm_spatial_entities_filename = "norm_spatial_entities_" + consts.NEWS_DB_BVBRC
    self.input_norm_disease_entities_filename = "norm_disease_entities_" + consts.NEWS_DB_BVBRC
    self.input_norm_host_entities_filename = "norm_host_entities_" + consts.NEWS_DB_BVBRC
    # output
    self.out_events_filename = "doc_events_" + consts.NEWS_DB_BVBRC + "_task1=" + self.event_retrieval_strategy.get_description() + "." + consts.FILE_FORMAT_CSV

  def read_csv_files_as_dataframe(self):
    super().read_csv_files_as_dataframe()

  def perform_event_retrieval(self, disease_name, in_preprocessing_result_folder, in_norm_result_folder,
                              out_doc_events_folder, data_folder):
    super().perform_event_retrieval(disease_name, in_preprocessing_result_folder, in_norm_result_folder,
                                    out_doc_events_folder, data_folder)

  def retrieve_doc_events(self, disease_name):
    return super().retrieve_doc_events(disease_name)