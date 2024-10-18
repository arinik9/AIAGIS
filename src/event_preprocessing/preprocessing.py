'''
Created on Nov 16, 2021

@author: nejat
'''

import os
import json
import math

import pandas as pd
import numpy as np

import src.consts as consts

import dateutil.parser as parser

import shutil

from abc import ABC, abstractmethod


import csv
import re

from src.event_preprocessing.disease_entity_processing import retrieve_disease_from_raw_sentence
                                
from src.event_preprocessing.host_entity_processing import retrieve_host_from_raw_sentence
                                
from tldextract import extract

from src.media_sources.news_source_geo_coverage import perform_media_source_resolution

from datetime import date


from src.event_preprocessing.spatial_entity_processing_in_title import retrieve_spacy_raw_locations_from_title,\
                       enrich_raw_spacy_locations_with_country_code_and_loc_type, retrieve_titles_without_spacy_locations, \
                       retrieve_location_info_from_title_with_knowledge_base, combine_all_title_locations
from src.event_preprocessing.host_entity_processing import retrieve_host_entity_list

from src.event_preprocessing.disease_entity_processing import retrieve_disease_entity_list

from src.event_preprocessing.spatial_entity_processing import retrieve_loc_list_searchable_on_geonames



class AbstractPreprocessing(ABC):
  ''' Abstract class for event_preprocessing. This class contains all the common methods.
  used in the extended classes (e.g. PreprocessingPadiweb). 
  Currently, there is a single method: "trim_url_custom()'''
  
  def __init__(self):
    pass

  @abstractmethod
  def perform_preprocessing(self):
    pass
  
  
  ########################################################################
  # This method trims the URL text in input to get a proper URL. This URL
  #  can be a website link or a twitter link. It is currently used only 
  #  for PADI-web, but it is useful for any EBS platform.
  #
  # Example:
  # Input:
  # url = "http://news.google.com/news/url?sa=t&fd=R&ct2=us&usg=AFQjCNGVMSrg2dLxRQjoY7IsVq4KnyhJSQ&
  #   clid=c3a7d30bb8a4878e06b80cf16b898331&cid=52780061392433&ei=GZjBW4C7FqPbzAa3qpCIBg
  #   &url=https://www.channelnewsasia.com/news/asia/china-reports-new-h5n6-bird-flu-outbreak-in-hunan-province-10808126"
  # Output:
  # url = "https://www.channelnewsasia.com/news/asia/china-reports-new-h5n6-bird-flu-outbreak-in-hunan-province-10808126"
  #
  # PARAMETERS
  # - url: URL as string
  ########################################################################
  def trim_url_custom(self, url):
    http_count = url.count("http")
    if http_count > 0:
      url_part_by_http = url.split("http")
      part_of_interest = url_part_by_http[http_count]
      part_of_interest = part_of_interest.replace("%2F","/")
      part_of_interest = part_of_interest.split("//")[1]
      twitter_count = url.count("twitter")
      if twitter_count>0:
        # https://twitter.com/FluTrackers/statuses/1077629051630702593
        if "/status" in part_of_interest:
          final_source = part_of_interest.split("/status")[0]
        else:
          final_source = part_of_interest.split("/statuses")[0]
      else:
        final_source = part_of_interest.split("/")[0]
        
      # https://redirect.viglink.com?u=sdsfsdfds
      # The ? indicates that there are variables coming in.
      res = re.match(r"^(.)+\?", final_source)
      if res is not None:
        final_source = final_source.split("?", 1)[0]
        
      if "wwwnc." in final_source:
        final_source = final_source.split("wwwnc.")[1]
      if "www." in final_source:
        final_source = final_source.split("www.")[1]
      

      if final_source == "oie.int":
        final_source = "wahis.oie.int"

      # BEGIN ---------------------------
      # some manual cleaning rules
      final_source = re.sub('^e\.', '', final_source)
      final_source = re.sub('^m\.', '', final_source)
      final_source = re.sub('^english\.', '', final_source)
      final_source = re.sub('^eng\.', '', final_source)
      final_source = re.sub('^en\.', '', final_source)
      final_source = re.sub('^fr\.', '', final_source)
      final_source = re.sub('^french\.', '', final_source)
      final_source = re.sub('^it\.', '', final_source)
      final_source = re.sub('^blog\.', '', final_source)
      final_source = re.sub('^blogs\.', '', final_source)
      final_source = re.sub('^world\.', '', final_source)
      final_source = re.sub('^in\.', '', final_source)
      final_source = re.sub('^www01\.', '', final_source)
      final_source = re.sub('^www1\.', '', final_source)
      final_source = re.sub('^www2\.', '', final_source)
      final_source = re.sub('^www3\.', '', final_source)
      final_source = re.sub('^wap\.', '', final_source)
      if "indiatimes.com" in final_source:
        final_source = "indiatimes.com"
      if "yna.co.kr" in final_source:
        final_source = "yna.co.kr"
      if "kbs.co.kr" in final_source:
        final_source = "kbs.co.kr"
      if "china.org.cn" in final_source:
        final_source = "china.org.cn"        
      if "reuters.com" in final_source:
        final_source = "reuters.com"
      if "sputniknews.com" in final_source:
        final_source = "sputniknews.com"   
      if "dvm360.com" in final_source:
        final_source = "dvm360.com"
      if "yahoo.com" in final_source:
        final_source = "yahoo.com"  
      if "who.int" in final_source:
        final_source = "who.int" 
      if "francetvinfo.fr" in final_source:
        final_source = "francetvinfo.fr"
      if "cbslocal.com" in final_source:
        final_source = "cbslocal.com"
      if "gelocal.it" in final_source:
        final_source = "gelocal.it"
      if "ctvnews.ca" in final_source:
        final_source = "ctvnews.ca"  
      if "floridaweekly.com" in final_source:
        final_source = "floridaweekly.com"      
      if "advocatemag.com" in final_source:
        final_source = "advocatemag.com"
      if "euronews.com" in final_source:
        final_source = "euronews.com"
      if "repubblica.it" in final_source:
        final_source = "repubblica.it"
      if "corriere.it" in final_source:
        final_source = "corriere.it"
      if "timesreview.com" in final_source:
        final_source = "timesreview.com"
      if "news12.com" in final_source:
        final_source = "news12.com"
      if "newschannelnebraska.com" in final_source:
        final_source = "newschannelnebraska.com"
      if "springer.com" in final_source:
        final_source = "springer.com"
      # END ---------------------------

      return(final_source)
    return(url)
  
  
  
  



class PreprocessingPadiweb(AbstractPreprocessing):
  '''
    PADI-web concrete class for event_preprocessing. This class contains all the common methods.
    used in the extended classes (e.g. PreprocessingPadiweb). 
    The main trigger method for event_preprocessing is perform_preprocessing()
  '''  

  def __init__(self):
    self.input_events_filename = consts.PADIWEB_ARTICLES_CSV_FILENAME
    self.output_raw_events_filename = "raw_events_"+consts.NEWS_SURVEILLANCE_PADIWEB
    self.output_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
    self.output_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
    self.output_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
    
    
  ################################################################################################
  # This method is the main trigger method for event_preprocessing PADI-web input data.
  # It performs the event_preprocessing in 10 steps:
  # 1) reading the input files and copying into a new folder.
#      All new files created in this method will be stored in this new folder.
  # 2) retrieve source names from the corresponding URLs (e.g. "lemonde" from "www.lemonde.fr")
  # 3) Remove duplicated articles
  # 4) retrieve the country information regarding media sources publishing news articles
  #    (e.g. France for "www.lemonde.fr")
  # 5) Extract spatial entities from the titles and perform geocoding.
  # 6) add "local sentence id" into "sentences_with_labels.csv" and create a new file 
  #    "extended_sentences_with_labels.csv". "local sentence id" represents the sentence order 
  #    in an article. If we treat the first (resp. 1). sentence of an article, 
  #    its "local sentence id" is (resp. 1).
  # 7) create an auxiliary data structure to make an association between
  #      "extracted_information.csv" and "extended_sentences_with_labels.csv"
  # 8) keep only relevant entities/keywords of relevant sentences from relevant articles
  # 9) Retrieve host entities from the relevant sentences of an relevant article. 
  #    We do not use the host entities provided by PADI-web 
  # 10) Retrieve disease entities from the relevant sentences of an relevant article. 
  #    We do not use the host entities provided by PADI-web 
  #
  # REMARK: We use the spatial entities extracted from PADI-web
  #
  #
  # PARAMETERS
  # - disease_name: the disease for which we perform the normalized event extraction task
  # - input_folder: input folder containing the input files (i.e. epidnews2event/in/padiweb)
  # - out_preprocessing_result_folder: output folder for storing the event_preprocessing results
  #                                    (i.e. epidnews2event/out/event_preprocessing-results)
  # - data_folder: data folder containing external data (i.e. epidnews2event/data)
  ################################################################################################
  def perform_preprocessing(self, disease_name, input_folder, out_preprocessing_result_folder, data_folder):
    print("...... beginning of event_preprocessing for padiweb !")
    self.disease_name = disease_name
    self.input_folder = input_folder
    self.out_preprocessing_result_folder = out_preprocessing_result_folder
    self.data_folder = data_folder

    # ===========================================================================================
    # PART 1: reading the input files and copying into a new folder
    # ===========================================================================================   
        
    self.read_csv_files_as_dataframe()
    
    self.copy_csv_files_from_input() # for DEBUG

    # ===========================================================================================
    # PART 2: Enhancing and event_preprocessing the following information:
    #            - URL
    #            - media source name
    #            - publication date
    # ===========================================================================================      
    
    # retrieve source names from the corresponding urls, and overwrite the csv event file
    source_names = [self.trim_url_custom(url) for url in self.df_articles[consts.COL_URL]]
    self.df_articles[consts.COL_SOURCE] = source_names
    articles_filepath = os.path.join(self.out_preprocessing_result_folder, \
                       self.output_raw_events_filename + "." + consts.FILE_FORMAT_CSV)
    
    self.df_articles["p_date"] = self.df_articles[consts.COL_PUBLISHED_TIME].apply(parser.parse)
    self.df_articles["p_date2"] = self.df_articles["p_date"].apply(lambda x: str(x.day)+"/"+str(x.month)+"/"+str(x.year))
    self.df_articles.groupby([consts.COL_TITLE, "p_date2"])[consts.COL_ID, consts.COL_URL, consts.COL_SOURCE, consts.COL_PUBLISHED_TIME, \
                     consts.COL_TEXT, consts.COL_PROCESSED_TEXT, consts.COL_LANG].aggregate(lambda x: list(x)[0])
    
    self.df_articles["source_td"] = self.df_articles["source"].apply(lambda x: extract(x)[1])
    self.df_articles.loc[self.df_articles["source_td"] == "www", "source_td"] = self.df_articles.loc[self.df_articles["source_td"] == "www", "source"]
    self.df_articles.loc[self.df_articles["source_td"] == "gov.uk", "source_td"] = self.df_articles.loc[self.df_articles["source_td"] == "gov.uk", "source"]
    self.df_articles.loc[self.df_articles["source_td"] == "gov.ie", "source_td"] = self.df_articles.loc[self.df_articles["source_td"] == "gov.ie", "source"]
    self.df_articles.loc[self.df_articles["source_td"] == "gov.scot", "source_td"] = self.df_articles.loc[self.df_articles["source_td"] == "gov.scot", "source"]
    
    # ===========================================================================================
    # PART 3: Remove duplicates based on:
    #            - title
    #            - url base
    #            - publication date
    # ===========================================================================================
    
    df_groupby = self.df_articles.groupby([consts.COL_TITLE, "source_td", "p_date2"])[consts.COL_ID, consts.COL_URL, consts.COL_SOURCE, consts.COL_PUBLISHED_TIME, \
                     consts.COL_TEXT, consts.COL_PROCESSED_TEXT, consts.COL_LANG].aggregate(lambda x: list(x)[0])
    
    df_groupby.to_csv(articles_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
    
    # ===========================================================================================
    # PART 4: We use the following dataset to retrieve the country information regarding media sources
    #          publishing news articles: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/4F7TZW
    #        In this dataset, if I remember correctly, it basically lists the national news sources, 
    #          but there is no distinction between regional/local or national. 
    # ===========================================================================================
    
    #self.read_csv_files_as_dataframe() # reload them, especially for articles
    self.df_articles = pd.read_csv(articles_filepath, sep=";", keep_default_na=False, encoding='utf8') # for DEBUG

    
    
    news_source_url_list = self.df_articles[consts.COL_URL].to_numpy()
    out_filepath = os.path.join(self.out_preprocessing_result_folder, "media_source_resolution" + "." + consts.FILE_FORMAT_CSV)
    df_media_source = perform_media_source_resolution(news_source_url_list, self.data_folder)
    df_media_source[consts.COL_ARTICLE_ID] = self.df_articles[consts.COL_ID]
    df_media_source.to_csv(out_filepath, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    
    
    # ===========================================================================================
    # PART 5: Extract spatial entities from the titles and perform geocoding.
    #          These spatial entities will serve as an initial bias 
    #          for correctly geocoding difficult spatial entities found in the main text.
    #          In the end, it produced the file "title_locations_final_with_geonames.csv". 
    #          This file has the following columns:
    #          - article_id
    #          - media_source_country_code
    #          - country_bias
    #          - final_value
    #          - final_geonames_json
    #            (Note that there might be multiple spatial entities in the same title)
    # ===========================================================================================
    
    self.retrieve_spatial_entities_from_title_for_all_articles()
    
    
    
    # ===========================================================================================
    # PART 6: Add local sentence id into "extended_sentences_with_labels.csv"
    #        Normally, PADI-web uses global sentence ids. But, it is also useful to to know 
    #          the first sentence of a given article. This is why we extract the local ids
    # ===========================================================================================
    
    a_ids = self.df_sentences[consts.COL_ARTICLE_ID].to_numpy()
    nb_row = self.df_sentences.shape[0]
    article_ending_sentence_indexes = {}
    for i in range(0,nb_row):
      a_id = a_ids[i]
      if a_id not in article_ending_sentence_indexes:
        article_ending_sentence_indexes[a_id] = i # i: starting index
    
    article_starting_sentence_indexes = {}
    for i in range(0,nb_row):
      a_id = a_ids[i]
      article_starting_sentence_indexes[a_id] = i # i: ending index       
    
    self.df_sentences[consts.COL_LOCAL_SENTENCE_ID] = -1
    unique_a_ids = np.unique(a_ids)
    for a_id in unique_a_ids:
      start_index = article_starting_sentence_indexes[a_id]
      end_index = article_ending_sentence_indexes[a_id]
      nb_sents = start_index-end_index+1 # they are in desc order
      print(a_id, start_index, end_index, nb_sents)
      self.df_sentences.iloc[end_index:(start_index+1)][consts.COL_LOCAL_SENTENCE_ID] = range(nb_sents) # range(nb_sents-1, -1, -1) # decreasing order, until we reach 0 index.
    
    self.df_sentences = self.df_sentences.sort_values(by=[consts.COL_ARTICLE_ID, consts.COL_LOCAL_SENTENCE_ID], ascending=True)
    
    filepath = os.path.join(self.out_preprocessing_result_folder, \
                          consts.PADIWEB_EXT_SENTENCES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    self.df_sentences.to_csv(filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC) # TODO not used in the rest of the code
    
    
    # after the 'groupby' operation, I would not find a solution to access the desired column
    # as a workaround, I reread the dataframe.
    ##df_sentences_agg = pd.read_csv(filepath, sep=";", keep_default_na=False)
    self.sentence_id_to_local_id_ass = dict(zip(self.df_sentences[consts.COL_ID], self.df_sentences[consts.COL_LOCAL_SENTENCE_ID]))
    
    
    # ===========================================================================================
    # PART 7: 1) Create "inverted_article_indices_for_sentences"
    #         2) It extends the initial "extracted_information.csv" file with these information:
    #            -sentence local id
    #            - geonames_info
    #            - sentence classification label
    # All these information help identifying relevant disease related keywords, which already 
    #    extracted by PADI-web
    # ===========================================================================================
    
    nb_row = self.df_sentences.shape[0]
    a_ids = self.df_sentences[consts.COL_ARTICLE_ID].to_numpy()
    
    # inverted indexes for sentences: this is before filtering
    # because the sentences with ids != 12 can have some additional information 
    self.inverted_article_indices_for_sentences = {}
    for i in range(0,nb_row):
      a_id = a_ids[i]
      if a_id not in self.inverted_article_indices_for_sentences:
        self.inverted_article_indices_for_sentences[a_id] = i # i: starting index
    
    print(np.unique(a_ids).shape)    
    # >>> an exampel for self.inverted_article_indices_for_sentences
    # {'DEOL28QKIP': 0, 'MO5GLFVP9X': 2, '8QSAIG2LAH': 8, 'P068SXF7LZ': 14, 'M9Y31PW9YJ': 19, '8Q1P443EDC': 43, 'R9K3R99XTF': 63, 'PONHFDNH5G': 67, 'M04B9PUD06': 79, 'QLYNYYQOFH': 97, 'G7QZQ1KAN0': 105, '9IXRWPVJR1': 115, 'XJ90HGTPYT': 121, 'JQG87NSQ6Q': 134, 'NDUJ0T87P2': 147, 'I0MN9I2TR8': 160, 'D59DLD0ODO': 174, ... }
    
    
    self.extend_info_extr_with_sentence_id_and_geonames_and_classif_label()
    # DEBUG purposes ------------------------------------
    # >> In order to comment the method above:
    # extr_info_filepath = os.path.join(self.out_preprocessing_result_folder, \
    #            "extr_info_improved" + "." + consts.FILE_FORMAT_CSV)
    # self.df_info_extr = pd.read_csv(extr_info_filepath, sep=";", keep_default_na=False)
    # ---------------------------------------------------
    
    
    # ===========================================================================================
    # PART 8: Keep only relevant entities/keywords of relevant sentences from relevant articles
    #          In the end, it creates the file 'extended_extracted_information.csv'
    #            - sentence global id
    #            - sentence local id
    #            - sentence classification id
    #            - feature code
    # ===========================================================================================
    
    # scenario: the classification label of a given sentence is not "11 12", but it contains outbreak related loc 
    df_ext_info_outbreaks = self.df_info_extr[self.df_info_extr["type"] == "outbreak-related-location"]
    # oubtreaks_sentence_ids_11_12 = df_ext_info_outbreaks[df_ext_info_outbreaks[consts.COL_SENTENCE_CLASSIF_LABEL_ID] == "11 12"]
    oubtreaks_sentence_ids_other = df_ext_info_outbreaks[df_ext_info_outbreaks[consts.COL_SENTENCE_CLASSIF_LABEL_ID] != "11 12"]
    sentence_ids_to_adjust = oubtreaks_sentence_ids_other[consts.COL_SENTENCE_ID].to_list()
    #
    self.df_info_extr.loc[self.df_info_extr[consts.COL_SENTENCE_ID].isin(sentence_ids_to_adjust), [consts.COL_SENTENCE_CLASSIF_LABEL_ID]] = "11 12"
    #
    self.df_sentences.loc[self.df_sentences[consts.COL_ID].isin(sentence_ids_to_adjust), [consts.COL_SENTENCE_CLASSIF_LABEL_ID]] = "11 12"
    
    # filter aggregated sentences by classification label
    drop_vals = [k for k in self.df_sentences.index \
                 if str(consts.CLASSIF_LABEL_CURR_EVENT_ID)+" "+str(consts.CLASSIF_LABEL_DESCR_EPID_ID) \
                  != self.df_sentences.loc[k, consts.COL_SENTENCE_CLASSIF_LABEL_ID]\
                and str(consts.CLASSIF_LABEL_CURR_EVENT_ID)+" "+str(consts.CLASSIF_LABEL_GENERAL_EPID_ID)\
                  != self.df_sentences.loc[k, consts.COL_SENTENCE_CLASSIF_LABEL_ID]\
                and str(consts.CLASSIF_LABEL_CURR_EVENT_ID)+" "+str(consts.CLASSIF_LABEL_TRANS_PATHWAY_ID)\
                  != self.df_sentences.loc[k, consts.COL_SENTENCE_CLASSIF_LABEL_ID]\
    ]
    df_sentences_filtered = self.df_sentences.drop(drop_vals)
    filepath = os.path.join(self.out_preprocessing_result_folder, \
                          consts.PADIWEB_RELEVANT_SENTENCES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    df_sentences_filtered.to_csv(filepath, index=True, sep=";", quoting=csv.QUOTE_NONNUMERIC) # TODO not used in the rest of the code
    
    filepath = os.path.join(self.out_preprocessing_result_folder, consts.PADIWEB_EXT_INFO_EXTR_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    self.df_info_extr.to_csv(filepath, index=True, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
    
    # # ===========================================================================================
    # # PART 9: Retrieve host entities
    # # ===========================================================================================   
    extr_host_info_filepath = os.path.join(self.out_preprocessing_result_folder, \
                                     self.output_preprocessed_host_entities_filename + "." + consts.FILE_FORMAT_CSV)
    NCBI_host_results_filepath = os.path.join(self.out_preprocessing_result_folder, \
                                     "NCBI_host_results_in_batch" + "." + consts.FILE_FORMAT_CSV)
    self.retrieve_host_entities_from_text_for_all_articles(extr_host_info_filepath, NCBI_host_results_filepath)
    
    # ===========================================================================================
    # PART 10: Retrieve disease entities
    # =========================================================================================== 
    extr_disease_info_filepath = os.path.join(self.out_preprocessing_result_folder, \
                                    self.output_preprocessed_disease_entities_filename + "." + consts.FILE_FORMAT_CSV)   
    self.retrieve_disease_entities_from_text_for_all_articles(extr_disease_info_filepath)
    
    # ===========================================================================================
    # PART 11: Retrieve spatial entities
    #          PADI-web already identifies the spatial entities. We just put them in a separated file
    # =========================================================================================== 
    
    drop_vals = [k for k in self.df_info_extr.index \
                 if str(consts.CLASSIF_LABEL_CURR_EVENT_ID)+" "+str(consts.CLASSIF_LABEL_DESCR_EPID_ID)\
                  != self.df_info_extr.loc[k, consts.COL_SENTENCE_CLASSIF_LABEL_ID]\
                 and str(consts.CLASSIF_LABEL_CURR_EVENT_ID)+" "+str(consts.CLASSIF_LABEL_GENERAL_EPID_ID)\
                  != self.df_info_extr.loc[k, consts.COL_SENTENCE_CLASSIF_LABEL_ID]\
                 and str(consts.CLASSIF_LABEL_CURR_EVENT_ID)+" "+str(consts.CLASSIF_LABEL_TRANS_PATHWAY_ID)\
                  != self.df_info_extr.loc[k, consts.COL_SENTENCE_CLASSIF_LABEL_ID]\
                ]
    df_ext_info_extr_only_relevant = self.df_info_extr.drop(drop_vals)
    df_only_relevant_spatial_entities = df_ext_info_extr_only_relevant[
                    (df_ext_info_extr_only_relevant[consts.COL_GEONAMES_ID] != "NULL") &
                    (df_ext_info_extr_only_relevant[consts.COL_GEONAMES_ID] != "-1")
                    ]
    filepath = os.path.join(self.out_preprocessing_result_folder, self.output_preprocessed_spatial_entities_filename + "." + consts.FILE_FORMAT_CSV)
    df_only_relevant_spatial_entities.to_csv(filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)


    print("...... end of event_preprocessing for padiweb !")




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
  def retrieve_spatial_entities_from_title_for_all_articles(self):
    
    media_source_resolution_filepath = os.path.join(self.out_preprocessing_result_folder, "media_source_resolution" + "." + consts.FILE_FORMAT_CSV)
    title_spacy_raw_locations_filepath = os.path.join(self.out_preprocessing_result_folder, \
                                  "title_locations_spacy" + "." + consts.FILE_FORMAT_CSV)
    retrieve_spacy_raw_locations_from_title(self.df_articles, media_source_resolution_filepath, title_spacy_raw_locations_filepath)
    
    title_spacy_raw_locations_with_type_filepath = os.path.join(self.out_preprocessing_result_folder, \
                                    "title_locations_spacy_with_type" + "." + consts.FILE_FORMAT_CSV)
    enrich_raw_spacy_locations_with_country_code_and_loc_type(title_spacy_raw_locations_filepath, title_spacy_raw_locations_with_type_filepath)
    
    titles_without_spacy_locations_filepath = os.path.join(self.out_preprocessing_result_folder, "titles_without_spacy_locations" + "." + consts.FILE_FORMAT_CSV)
    retrieve_titles_without_spacy_locations(self.df_articles, title_spacy_raw_locations_with_type_filepath, \
                                            titles_without_spacy_locations_filepath, media_source_resolution_filepath)
    
    title_locations_from_knowledge_base_auxiliary_filepath = os.path.join(self.out_preprocessing_result_folder, \
                                    "title_locations_auxiliary_from_knowledge_base" + "." + consts.FILE_FORMAT_CSV)
    title_locations_from_knowledge_base_final_filepath = os.path.join(self.out_preprocessing_result_folder, \
                                    "title_locations_from_knowledge_base" + "." + consts.FILE_FORMAT_CSV)
    retrieve_location_info_from_title_with_knowledge_base(self.data_folder, title_spacy_raw_locations_with_type_filepath, \
                                                          titles_without_spacy_locations_filepath, \
                                                          title_locations_from_knowledge_base_auxiliary_filepath, \
                                                          title_locations_from_knowledge_base_final_filepath)

    title_locations_final_filepath = os.path.join(self.out_preprocessing_result_folder, "titles_final_locations" + "." + consts.FILE_FORMAT_CSV)
    combine_all_title_locations(title_spacy_raw_locations_with_type_filepath, title_locations_from_knowledge_base_final_filepath, \
                                title_locations_final_filepath)
    


  ################################################################################################
  # It is the trigger method for retrieving the host entities from the content of the articles.
  # It does not rely on the host entities found by PADI-web. For this reason, it looks for 
  #  host entities with a lookup strategy based on NCBI taxonomy data.
  # REMARK: The goal here is not a normalization task. Rather, it is to check if there is any 
  #        host keywords. This is the preliminary step before normalization.
  #
  #
  # PARAMETERS
  # - extr_host_info_filepath: Output file.
  # - NCBI_host_results_filepath: Auxiliary file for storing the results.
  #                               Since the NCBI database is large enough, this speeds up 
  #                                the execution, when we rerun the code.
  #                                This file indicates if a specific keyword is found 
  #                                in the NCBI database or not.
  ################################################################################################ 
  def retrieve_host_entities_from_text_for_all_articles(self, extr_host_info_filepath, NCBI_host_results_filepath):
    
    articles_filepath = os.path.join(self.input_folder, \
                           consts.PADIWEB_ARTICLES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    cols_articles = [consts.COL_ID]
    df_articles = pd.read_csv(articles_filepath, usecols=cols_articles, sep=";", keep_default_na=False, encoding='utf8')
    article_id_list = df_articles[consts.COL_ID].tolist()
    
    sentences_filepath = os.path.join(self.out_preprocessing_result_folder, \
                          consts.PADIWEB_EXT_SENTENCES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    df_sentences =  pd.read_csv(sentences_filepath, sep=";", keep_default_na=False, encoding='utf8')

    df_list = []
    for indx, article_id in enumerate(article_id_list):
      df_sentences_by_article = df_sentences[df_sentences[consts.COL_ARTICLE_ID] == article_id]
      print("article_id:", article_id)
      df_host_info = self.retrieve_host_entities_in_sentences_by_article(article_id, df_sentences_by_article, NCBI_host_results_filepath)
      df_list.append(df_host_info)
    df_all_host_info = pd.concat(df_list)
    df_all_host_info.to_csv(extr_host_info_filepath, index=True, sep=";", quoting=csv.QUOTE_NONNUMERIC)


  ################################################################################################
  # It is the main method for retrieving the host entities from the content of a specific article.
  # The processing is done sentence by sentence.
  # It does not rely on the host entities found by PADI-web. For this reason, it looks for 
  #  host entities with a lookup strategy based on NCBI taxonomy data.
  # REMARK: The goal here is not a normalization task. Rather, it is to check if there is any 
  #        host keywords. This is the preliminary step before normalization.
  #
  #
  # PARAMETERS
  # - article_id: the article id for which we search for host entities
  # - df_sentences_by_article: sentences of the article with "article_id" in a dataframe format.
  # - NCBI_host_results_filepath: Auxiliary file for storing the results.
  #                               Since the NCBI database is large enough, this speeds up 
  #                                the execution, when we rerun the code.
  #                                This file indicates if a specific keyword is found in 
  #                                the NCBI database or not.
  ################################################################################################ 
  def retrieve_host_entities_in_sentences_by_article(self, article_id, df_sentences_by_article, NCBI_host_results_filepath):
    host_info_list = []
    host_type_list = []
    sentence_id_list = []
    text_list = []
    for index, row in df_sentences_by_article.iterrows():
      sentence_id = row[consts.COL_LOCAL_SENTENCE_ID]
      sentence_text = row[consts.COL_TEXT]
      print("---", sentence_id, sentence_text)
      host_list, host_t_list = retrieve_host_from_raw_sentence(sentence_text, NCBI_host_results_filepath, \
                                                is_raw_sentence=True) # host info can be empty
      # Example: host_list = ["dog"]
      print("result", host_list)
      host_info_list = host_info_list + host_list
      host_type_list = host_type_list + host_t_list
      text_list = text_list + [sentence_text]*len(host_list)
      sentence_id_list = sentence_id_list + [sentence_id]*len(host_list)
      
    df_host_info = pd.DataFrame(
    {consts.COL_ARTICLE_ID: article_id,
     consts.COL_LOCAL_SENTENCE_ID: sentence_id_list,
     consts.COL_SENT_TEXT: text_list, # for DEBUG purposes
     consts.COL_HOST: host_info_list,
     consts.COL_TYPE: host_type_list
    })
    
    return df_host_info
      
         


  ################################################################################################
  # It is the trigger method for retrieving the disease entities from the content of the articles.
  # It does not rely on the disease entities found by PADI-web. For this reason, it looks for 
  #  disease entities with a lookup strategy based on a custom simple taxonomy data that I constructed.
  # REMARK: The goal here is not a normalization task. Rather, it is to check if there is any 
  #        disease keywords. This is the preliminary step before normalization.
  #
  #
  # PARAMETERS
  # - extr_disease_info_filepath: Output file.
  # REMARK: Here, we do not need to store the results in an auxiliary file as in host entity processing,
  #         since the execution is fast enough.
  ################################################################################################  
  def retrieve_disease_entities_from_text_for_all_articles(self, extr_disease_info_filepath):
    
    articles_filepath = os.path.join(self.input_folder, \
                           consts.PADIWEB_ARTICLES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    cols_articles = [consts.COL_ID]
    df_articles = pd.read_csv(articles_filepath, usecols=cols_articles, sep=";", keep_default_na=False, encoding='utf8')
    article_id_list = df_articles[consts.COL_ID].tolist()
    
    sentences_filepath = os.path.join(self.out_preprocessing_result_folder, \
                          consts.PADIWEB_EXT_SENTENCES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    df_sentences =  pd.read_csv(sentences_filepath, sep=";", keep_default_na=False, encoding='utf8')

    df_list = []
    for indx, article_id in enumerate(article_id_list):
      df_sentences_by_article = df_sentences[df_sentences[consts.COL_ARTICLE_ID] == article_id]
      print("--- article_id:", article_id)
      df_disease_info = self.retrieve_disease_entities_in_sentences_by_article(article_id, df_sentences_by_article)
      print(df_disease_info)
      df_list.append(df_disease_info)
      
    df_all_disease_info = pd.concat(df_list)
    df_all_disease_info.to_csv(extr_disease_info_filepath, index=True, sep=";", quoting=csv.QUOTE_NONNUMERIC)



  ################################################################################################
  # It is the main method for retrieving the disease entities from the content of a specific article.
  # The processing is done sentence by sentence.
  # It does not rely on the disease entities found by PADI-web. For this reason, it looks for 
  #  disease entities with a lookup strategy based on a custom simple taxonomy data that I constructed.
  # REMARK: The goal here is not a normalization task. Rather, it is to check if there is any 
  #        host keywords. This is the preliminary step before normalization.
  #
  #
  # PARAMETERS
  # - article_id: the article id for which we search for host entities
  # - df_sentences_by_article: sentences of the article with "article_id" in a dataframe format.
  ################################################################################################ 
  def retrieve_disease_entities_in_sentences_by_article(self, article_id, df_sentences_by_article):
    disease_info_list = []
    disease_type_list = []
    sentence_id_list = []
    text_list = []
    for index, row in df_sentences_by_article.iterrows():
      sentence_id = row[consts.COL_LOCAL_SENTENCE_ID]
      sentence_text = row[consts.COL_TEXT]
      print("---", sentence_id, sentence_text)
      disease_list, disease_t_list = retrieve_disease_from_raw_sentence(sentence_text, self.disease_name)
      disease_info_list = disease_info_list + disease_list
      disease_type_list = disease_type_list + disease_t_list
      text_list = text_list + [sentence_text]*len(disease_list)
      sentence_id_list = sentence_id_list + [sentence_id]*len(disease_list)
      
    df_disease_info = pd.DataFrame(
    {consts.COL_ARTICLE_ID: article_id,
     consts.COL_LOCAL_SENTENCE_ID: sentence_id_list,
     consts.COL_SENT_TEXT: text_list, # for DEBUG purposes
     consts.COL_DISEASE: disease_info_list,
     consts.COL_TYPE: disease_type_list
    })
    return df_disease_info


  

  def read_csv_files_as_dataframe(self):
    articles_filepath = os.path.join(self.input_folder, \
                           self.input_events_filename + "." + consts.FILE_FORMAT_CSV)
    cols_articles = [consts.COL_ID, consts.COL_TITLE, consts.COL_URL, consts.COL_SOURCE, consts.COL_PUBLISHED_TIME, \
                     consts.COL_TEXT, consts.COL_PROCESSED_TEXT, consts.COL_LANG]
    self.df_articles = pd.read_csv(articles_filepath, usecols=cols_articles, sep=";", keep_default_na=False, encoding='utf8')
    
    sentences_filepath = os.path.join(self.input_folder, \
                           consts.PADIWEB_SENTENCES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    cols_sentences = [consts.COL_ID,consts.COL_TEXT,consts.COL_START,consts.COL_END, \
                      consts.COL_ARTICLE_ID, consts.COL_LANG, consts.COL_SENTENCE_CLASSIF_LABEL_ID]
    self.df_sentences = pd.read_csv(sentences_filepath, usecols=cols_sentences, sep=";", keep_default_na=False, encoding='utf8')
    
    info_extr_filepath = os.path.join(self.input_folder, \
                            consts.PADIWEB_INFO_EXTRAC_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    cols_info_extr = [consts.COL_ID, consts.COl_POS, consts.COl_TOKEN_INDEX, consts.COl_LENGTH, \
                      consts.COL_TYPE, consts.COL_LABEL, consts.COL_VALUE, \
                      consts.COL_ARTICLE_ID_RENAMED, consts.COL_KEYW_ID, consts.COL_FROM_AUTO_EXTR, \
                      consts.COL_GEONAMES_ID, consts.COL_GEONAMS_JSON] 
    self.df_info_extr = pd.read_csv(info_extr_filepath, usecols=cols_info_extr, sep=";", keep_default_na=False, encoding='utf8')
    



  def copy_csv_files_from_input(self):
    articles_curr_filepath = os.path.join(self.input_folder, \
                           consts.PADIWEB_ARTICLES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    articles_new_filepath = os.path.join(self.out_preprocessing_result_folder, \
                           consts.PADIWEB_ARTICLES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    shutil.copyfile(articles_curr_filepath, articles_new_filepath)
    
    info_extr_curr_filepath = os.path.join(self.input_folder, \
                           consts.PADIWEB_INFO_EXTRAC_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    info_extr_new_filepath = os.path.join(self.out_preprocessing_result_folder, \
                           consts.PADIWEB_INFO_EXTRAC_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    shutil.copyfile(info_extr_curr_filepath, info_extr_new_filepath)
    
    sentences_curr_filepath = os.path.join(self.input_folder, \
                           consts.PADIWEB_SENTENCES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    sentences_new_filepath = os.path.join(self.out_preprocessing_result_folder, \
                           consts.PADIWEB_SENTENCES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)    
    shutil.copyfile(sentences_curr_filepath, sentences_new_filepath)

    
  

  def exists_article_id_in_sentences(self, article_id):
    if article_id in self.inverted_article_indices_for_sentences:
      return True
    return False
  
  
  def retrieve_sentence_id_by_pos(self, article_id, pos):
    indx = self.inverted_article_indices_for_sentences[article_id]-1 # minus 1, because, we will increment in each iteration of While block
    #print(article_id, pos, indx)
    while True:
      indx = indx + 1
      row = self.df_sentences.iloc[indx]
      #print(indx)
      s_id = row[consts.COL_ID]
      start = int(row[consts.COL_START])
      end = int(row[consts.COL_END])
      if pos>=start and pos<end:
        return s_id
    
    
  # Adding sentence id into Extracted Information
  def extend_info_extr_with_sentence_id_and_geonames_and_classif_label(self):
    geonames_ass = dict(zip(self.df_info_extr[consts.COL_GEONAMES_ID],self.df_info_extr[consts.COL_GEONAMS_JSON]))
    #>>> geonames_ass
    #{1668284.0: '{"address": "Taiwan", "class_description": "country, state, region,...", "code": "PCLI", 
    # "country": "Taiwan", "country_code": "TW", "description": "independent political entity",
    # "feature_class": "A", "geonames_id": 1668284, "lat": "24", "lng": "121", "ok": true, "population": 22894384, 
    # "raw": {"adminCode1": "00", "lng": "121", "geonameId": 1668284, "toponymName": "Taiwan", "countryId": "1668284",
    #  "fcl": "A", "population": 22894384, "countryCode": "TW", "name": "Taiwan", "fclName": "country, state, region,...", 
    # "countryName": "Taiwan", "fcodeName": "independent political entity", "adminName1": "", "lat": "24", 
    # "fcode": "PCLI"}, "state_code": "00", "status": "OK"}', ....................

    # we do "reset_index()" because after the groupby operation the grouped column becomes index.
    self.inverted_indices_for_sentences = dict(zip(self.df_sentences[consts.COL_ID], self.df_sentences.index))
    # print("index, ", self.inverted_indices_for_sentences[114641604], df_sentences_agg.iloc[self.inverted_indices_for_sentences[114641604]]) 

    # - collect sentence ids
    # - collect country codes from geonames json
    # - collect classification labels
    s_ids = []
    feature_codes = []
    classif_labels = []
    sentence_local_id_list = []
    for index, row in self.df_info_extr.iterrows():
      article_id = row[consts.COL_ARTICLE_ID_RENAMED]
      pos = row[consts.COl_POS]
      if self.exists_article_id_in_sentences(article_id):
        print("article id: ", article_id)
        s_id = self.retrieve_sentence_id_by_pos(article_id, pos)
        s_ids.append(s_id)
        s_local_id = self.sentence_id_to_local_id_ass[s_id]
        sentence_local_id_list.append(s_local_id)
        # ----
        i = self.inverted_indices_for_sentences[s_id]
        classif_label = self.df_sentences.loc[i,consts.COL_SENTENCE_CLASSIF_LABEL_ID]
        classif_labels.append(classif_label)
      else:
        s_ids.append(-1)
        classif_labels.append(-1)
        sentence_local_id_list.append(-1)
      # --------------------
      code = "-1"
      if row[consts.COL_TYPE] in ["outbreak-related-location", "GPE"]:
        geo_id = row[consts.COL_GEONAMES_ID]
        if not math.isnan(int(geo_id)) and int(geo_id) != -1:
          json_data = json.loads(geonames_ass[geo_id])
          if 'code' in json_data:
            code = json_data['code'] # country code
      feature_codes.append(code)
    
    self.df_info_extr[consts.COL_SENTENCE_ID] = s_ids
    self.df_info_extr[consts.COL_FEATURE_CODE] = feature_codes
    self.df_info_extr[consts.COL_SENTENCE_CLASSIF_LABEL_ID] = classif_labels
    self.df_info_extr[consts.COL_LOCAL_SENTENCE_ID] = sentence_local_id_list
    
    self.df_info_extr = self.df_info_extr.sort_values(by=[consts.COL_ARTICLE_ID_RENAMED, consts.COL_LOCAL_SENTENCE_ID, consts.COl_TOKEN_INDEX], ascending=True)
    
    extr_info_filepath = os.path.join(self.out_preprocessing_result_folder, \
                   "extr_info_improved" + "." + consts.FILE_FORMAT_CSV)
    self.df_info_extr.to_csv(extr_info_filepath, index=True, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
    





 
   
class PreprocessingEmpresi(AbstractPreprocessing):
  '''
    PADI-web concrete class for event_preprocessing. This class contains all the common methods.
    used in the extended classes (e.g. PreprocessingPadiweb). 
    The main trigger method for event_preprocessing is perform_preprocessing()
  '''  


  def __init__(self):
    self.input_events_filename = "outbreaks.csv"
    self.output_raw_events_filename = "raw_events_"+consts.NEWS_DB_EMPRESS_I
    self.output_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_DB_EMPRESS_I
    self.output_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_DB_EMPRESS_I
    self.output_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_DB_EMPRESS_I
  
  
  def set_csv_separator(self):
    return(";")
  
  
  def determine_columns_types(self):
    dtypes = {consts.COL_ID: str}
    return(dtypes)
  
  
  def postprocess_input_event_dataframe(self, df_input_events):
    # DO NOTHING
    return df_input_events
  
  
  def determine_columns_to_rename(self):
    rename_dict = {
      consts.COL_ID: consts.COL_ARTICLE_ID,
      "Region": consts.COL_LOC_CONTINENT,
      "ReportDate": consts.COL_PUBLISHED_TIME,
      "Serotype": consts.COL_DISEASE_SUBTYPE
      }
    return rename_dict
  
  
  def determine_useful_columns(self):
    cols_events = [consts.COL_ARTICLE_ID, consts.COL_LOC_CONTINENT, consts.COL_LOC_COUNTRY, consts.COL_LOC_REGION, \
                   consts.COL_LOC_CITY, consts.COL_LAT, consts.COL_LNG, \
                   consts.COL_PUBLISHED_TIME, consts.COL_DISEASE, consts.COL_DISEASE_SUBTYPE, \
                   consts.COL_HOST]
    return(cols_events)
  

  ################################################################################################
  # This method is the main trigger method for event_preprocessing EMPRES-i input data.
  # It performs the event_preprocessing in 4 steps:
  # 1) reading the input structured event data, renaming some columns in order that
  #    the output event file of all EBS platforms has the same column names, 
  #    and writing it into a new folder
  # 2) Spatial entity processing
  # 3) disease entity processing
  # 4) host entity processing
  #
  #
  # PARAMETERS
  # - disease_name: the disease for which we perform the normalized event extraction task
  # - input_folder: input folder containing the input files (i.e. epidnews2event/in/empres-i)
  # - out_preprocessing_result_folder: output folder for storing the event_preprocessing results
  #                                    (i.e. epidnews2event/out/event_preprocessing-results/empres-i)
  # - data_folder: data folder containing external data (i.e. epidnews2event/data)
  ################################################################################################    
  def perform_preprocessing(self, disease_name, input_folder, out_preprocessing_result_folder, data_folder):
    self.disease_name = disease_name
    self.input_folder = input_folder
    self.out_preprocessing_result_folder = out_preprocessing_result_folder
    self.data_folder = data_folder
    
    # ====================================================================
    # PART 1: READ INPUT EVENT FILE & RENAME SOME COLUMNS
    #          & KEEP ONLY USEFUL COLUMNS FOR PROCESSING
    # ====================================================================
    csv_sep = self.set_csv_separator()
    dtypes = self.determine_columns_types()
    input_events_filepath = os.path.join(self.input_folder, self.input_events_filename)
    self.df_raw_events = pd.read_csv(input_events_filepath, sep=csv_sep, keep_default_na=False, dtype=dtypes)
    
    self.df_raw_events = self.postprocess_input_event_dataframe(self.df_raw_events)
    
    rename_dict = self.determine_columns_to_rename()
    self.df_raw_events = self.df_raw_events.rename(columns=rename_dict)
    
    cols_events = self.determine_useful_columns()
    self.df_raw_events = self.df_raw_events[cols_events]
    
    raw_events_filepath = os.path.join(self.out_preprocessing_result_folder, \
               self.output_raw_events_filename + "." + consts.FILE_FORMAT_CSV)
    self.df_raw_events.to_csv(raw_events_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
    article_id_list = self.df_raw_events[consts.COL_ARTICLE_ID].values.tolist()
    
    
    # ====================================================================
    # PART 2: SPATIAL ENTITY PROCESSING
    #    Goal: Verify that the raw spatial texts provided in the
    #          input event file is really spatial entities. Sometimes, 
    #          some place names do not exist in the geocoders. In the 
    #          literature, finding spatial entities in text is called 
    #          "geotagging". This is the preliminary step before spatial
    #          entity normalization, i.e. geocoding. We use Geonames 
    #          for this purpose. Currently, we employ 2 strategies if
    #          a place name is not geocodable in Geonames:
    #          1) We try another geocoder Nominatim, and get the place name from it
    #          2) we discard the place name in question
    #    Remark: We need a lookup strategy for storing the processed results.
    #            Because, even verifying a single place name exists in Geonames may take time.
    #            Hence, this strategy accelerates the processing time when we need to rerun the code.
    # ====================================================================
    geonames_results_in_batch_filepath = os.path.join(self.out_preprocessing_result_folder, "geonames_results_in_batch" + "." + consts.FILE_FORMAT_CSV)

    country_list = self.df_raw_events[consts.COL_LOC_COUNTRY]

    preprocessed_country_list = retrieve_loc_list_searchable_on_geonames(country_list, geonames_results_in_batch_filepath)

    region_list = self.df_raw_events[consts.COL_LOC_REGION]
    preprocessed_region_list = retrieve_loc_list_searchable_on_geonames(region_list, geonames_results_in_batch_filepath)
    
    city_list = self.df_raw_events[consts.COL_LOC_CITY]
    preprocessed_city_list = retrieve_loc_list_searchable_on_geonames(city_list, geonames_results_in_batch_filepath)
    
    df_preprocessed_spatial_entity_info = pd.DataFrame({
       consts.COL_ARTICLE_ID: article_id_list,
       consts.COL_LOC_COUNTRY: preprocessed_country_list,
       consts.COL_LOC_REGION: preprocessed_region_list,
       consts.COL_LOC_CITY: preprocessed_city_list
    })

    # Remove the articles, where the place names are not geocodable
    # Here the filtering is not very strict:
    # - if a country name is recognized, then remove it (without checking the region and city names)
    # - if both the region and city names are not recognized, then remove it
    df_preprocessed_spatial_entity_info = df_preprocessed_spatial_entity_info[
                                                        df_preprocessed_spatial_entity_info[consts.COL_LOC_COUNTRY] != "-1"
                                                        ]
    df_preprocessed_spatial_entity_info = df_preprocessed_spatial_entity_info[
                                                        (df_preprocessed_spatial_entity_info[consts.COL_LOC_REGION] != "-1") &
                                                        (df_preprocessed_spatial_entity_info[consts.COL_LOC_CITY] != "-1")
                                                        ]
    # write the processed location results into a file
    preprocessed_spatial_entity_info_filepath = os.path.join(self.out_preprocessing_result_folder, \
                                                     self.output_preprocessed_spatial_entities_filename + "." + consts.FILE_FORMAT_CSV) 
    df_preprocessed_spatial_entity_info.to_csv(preprocessed_spatial_entity_info_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)



    # =========================================================
    # PART 3: DISEASE ENTITY PROCESSING
    #    Goal: Verify that the raw disease texts provided in the
    #          input event file is really host entities. Sometimes, 
    #          some disease names do not exist in the disease taxonomy databases.
    #          This is the preliminary step before disease entity normalization.
    #    Remark: We don't need a lookup strategy for disease
    #          entity processing, as it is fast. In the future,
    #           we can considering having such a strategy.
    # =========================================================
    self.df_raw_events[consts.COL_DISEASE+"2"] = self.df_raw_events[consts.COL_DISEASE] + " " + self.df_raw_events[consts.COL_DISEASE_SUBTYPE] 
    raw_disease_info_list = self.df_raw_events[consts.COL_DISEASE+"2"].to_list()
    
    preprocessed_disease_list, preprocessed_disease_type_list = retrieve_disease_entity_list(raw_disease_info_list, self.disease_name)
    print(len(article_id_list), len(preprocessed_disease_list), len(preprocessed_disease_type_list))
    df_preprocessed_disease_entity_info = pd.DataFrame({
       consts.COL_ARTICLE_ID: article_id_list,
       consts.COL_DISEASE: preprocessed_disease_list,
       consts.COL_TYPE: preprocessed_disease_type_list
    })
    
    # Remove the articles, where the disease names are not found
    df_preprocessed_disease_entity_info = df_preprocessed_disease_entity_info[df_preprocessed_disease_entity_info[consts.COL_DISEASE] != "-1"]

    # write the processed disease results into a file
    preprocessed_disease_entity_info_filepath = os.path.join(self.out_preprocessing_result_folder, \
                                                self.output_preprocessed_disease_entities_filename + "." + consts.FILE_FORMAT_CSV) 
    df_preprocessed_disease_entity_info.to_csv(preprocessed_disease_entity_info_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)


    
    # ================================================
    # PART 4: HOST ENTITY PROCESSING
    #    Goal: Verify that the raw host texts provided in the
    #          input event file is really host entities. Sometimes, 
    #          some host names do not exist in the host taxonomy databases.
    #          This is the preliminary step before host entity normalization.
    #          We use the host taxonomy database NCBI.
    #    Remark: We need a lookup strategy for storing the processed results.
    #            Because, even verifying a single host name exists in NCBI may take time.
    #            Hence, this strategy accelerates the processing time when we need to rerun the code.
    # ================================================
    raw_host_info_list = self.df_raw_events["host"].values.tolist()
    host_results_DB_filepath = os.path.join(self.out_preprocessing_result_folder, \
                                     "host_results_in_batch" + "." + consts.FILE_FORMAT_CSV)
    preprocessed_host_list, preprocessed_host_type_list = retrieve_host_entity_list(raw_host_info_list, host_results_DB_filepath)
    
    df_preprocessed_host_entity_info = pd.DataFrame({
       consts.COL_ARTICLE_ID: article_id_list,
       consts.COL_HOST: preprocessed_host_list,
       consts.COL_TYPE: preprocessed_host_type_list
    })
   
    # Remove the articles, where the host names do not exist in the taxonomy database
    df_preprocessed_host_entity_info = df_preprocessed_host_entity_info[df_preprocessed_host_entity_info[consts.COL_HOST] != "-1"]

    # write the processed host results into a file
    preprocessed_host_entity_info_filepath = os.path.join(self.out_preprocessing_result_folder, \
                                                          self.output_preprocessed_host_entities_filename + "." + consts.FILE_FORMAT_CSV) 
    df_preprocessed_host_entity_info.to_csv(preprocessed_host_entity_info_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)









class PreprocessingWahis(PreprocessingEmpresi):
  '''
    WAHIS concrete class for event_preprocessing. This class extends the 'PreprocessingEmpresi' class.
    It only renames the columns and does some basic event_preprocessing operations related to WAHIS event data.
    It uses the same main event_preprocessing method implemented in the 'PreprocessingEmpresi' class.
    For information, the main trigger method for event_preprocessing is perform_preprocessing()
  '''
  
  def __init__(self):
    self.input_events_filename = "infur_20230609.csv"
    self.output_raw_events_filename = "raw_events_"+consts.NEWS_DB_WAHIS
    self.output_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_DB_WAHIS
    self.output_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_DB_WAHIS
    self.output_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_DB_WAHIS
  
  
  def set_csv_separator(self):
    return(";")
  
  
  def determine_columns_types(self):
    dtypes = {"Outbreak_id": str}
    return(dtypes)
    
  
  def postprocess_input_event_dataframe(self, df_input_events):
    # DO NOTHING
    return df_input_events
  
    
  def determine_columns_to_rename(self):
    rename_dict = {
      "Outbreak_id": consts.COL_ARTICLE_ID,
      "region": consts.COL_LOC_CONTINENT,
      "country": consts.COL_LOC_COUNTRY,
      "level1_name": consts.COL_LOC_REGION,
      "level2_name": consts.COL_LOC_CITY,
      "Longitude": consts.COL_LNG,
      "Latitude": consts.COL_LAT,
      "disease_eng": consts.COL_DISEASE,
      "sero_sub_genotype_eng": consts.COL_DISEASE_SUBTYPE,
      "Species": consts.COL_HOST,
      "Reporting_date": consts.COL_PUBLISHED_TIME
      }
    return rename_dict
  
  
  def determine_useful_columns(self):
    cols_events = [consts.COL_ARTICLE_ID, consts.COL_LOC_CONTINENT, consts.COL_LOC_COUNTRY, consts.COL_LOC_REGION, \
                   consts.COL_LOC_CITY, consts.COL_LAT, consts.COL_LNG, \
                   consts.COL_PUBLISHED_TIME, consts.COL_DISEASE, consts.COL_DISEASE_SUBTYPE, \
                   consts.COL_HOST]
    return(cols_events)


  def perform_preprocessing(self, disease_name, input_folder, out_preprocessing_result_folder, data_folder):
    super().perform_preprocessing(disease_name, input_folder, out_preprocessing_result_folder, data_folder)





class PreprocessingPromed(PreprocessingEmpresi):
  '''
    ProMED concrete class for event_preprocessing. This class extends the 'PreprocessingEmpresi' class.
    It only renames the columns and does some basic event_preprocessing operations related to ProMED event data.
    It uses the same main event_preprocessing method implemented in the 'PreprocessingEmpresi' class.
    For information, the main trigger method for event_preprocessing is perform_preprocessing()
  '''
  
  def __init__(self):
    self.input_events_filename = "promed Influenza.csv"
    self.output_raw_events_filename = "raw_events_"+consts.NEWS_SURVEILLANCE_PROMED
    self.output_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_SURVEILLANCE_PROMED
    self.output_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_SURVEILLANCE_PROMED
    self.output_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_SURVEILLANCE_PROMED
 
 
  def set_csv_separator(self):
    return(";")
   

  def determine_columns_types(self):
    # ProMED archive number is in this form "number.numer" (e.g. 123213.12320)
    dtypes = {"promed_archive_number": str}
    return(dtypes)
  
  
  def determine_columns_to_rename(self):
    rename_dict = {
      "promed_archive_number": consts.COL_ARTICLE_ID,
      "region": consts.COL_LOC_CONTINENT,
      "country": consts.COL_LOC_COUNTRY,
      "admin1": consts.COL_LOC_REGION,
      "admin2": consts.COL_LOC_CITY,
      "longitude": consts.COL_LNG,
      "latitude": consts.COL_LAT,
      "disease": consts.COL_DISEASE,
      "serotype": consts.COL_DISEASE_SUBTYPE,
      "species": consts.COL_HOST,
      "report_date": consts.COL_PUBLISHED_TIME
      }
    return rename_dict
  
  
  def postprocess_input_event_dataframe(self, df_input_events):
    # DO NOTHING
    return df_input_events
  
  
  def determine_useful_columns(self):
    cols_events = [consts.COL_ARTICLE_ID, consts.COL_LOC_CONTINENT, consts.COL_LOC_COUNTRY, consts.COL_LOC_REGION, \
                   consts.COL_LOC_CITY, consts.COL_LAT, consts.COL_LNG, \
                   consts.COL_PUBLISHED_TIME, consts.COL_DISEASE, consts.COL_DISEASE_SUBTYPE, \
                   consts.COL_HOST]
    return(cols_events)


  def perform_preprocessing(self, disease_name, input_folder, out_preprocessing_result_folder, data_folder):
    super().perform_preprocessing(disease_name, input_folder, out_preprocessing_result_folder, data_folder)

    



class PreprocessingApha(PreprocessingEmpresi):
  '''
    APHA concrete class for event_preprocessing. This class extends the 'PreprocessingEmpresi' class.
    It renames the columns and does some basic event_preprocessing operations related to APHA event data.
    It also add some new columns which are not provided by the input event file.
    For instance, there is no country information, because APHA is specific to United States.
    It uses the same main event_preprocessing method implemented in the 'PreprocessingEmpresi' class.
    For information, the main trigger method for event_preprocessing is perform_preprocessing()
  '''
  
  def __init__(self):
    self.input_events_filename = "outbreaks apha.csv"
    self.output_raw_events_filename = "raw_events_"+consts.NEWS_DB_APHA
    self.output_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_DB_APHA
    self.output_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_DB_APHA
    self.output_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_DB_APHA
  

  def set_csv_separator(self):
    return(";")
  
  
  def determine_columns_types(self):
    dtypes = {} # no need to determine columns types, the id column is already str
    return(dtypes)
  
  
  def determine_columns_to_rename(self):
    rename_dict = {
      "Strain": consts.COL_DISEASE_SUBTYPE,
      "Species involved": consts.COL_HOST,
      "Date Detected": consts.COL_PUBLISHED_TIME
      }
    return rename_dict
  
  
  def postprocess_input_event_dataframe(self, df_input_events):
    df_input_events[consts.COL_ARTICLE_ID] = [consts.NEWS_DB_APHA+"_"+str(i) for i in range(df_input_events.shape[0])]
    df_input_events[consts.COL_LOC_CONTINENT] = "North America"
    df_input_events[consts.COL_LOC_COUNTRY] = "United States"
    df_input_events[consts.COL_LAT] = ""
    df_input_events[consts.COL_LNG] = ""
    df_input_events[consts.COL_DISEASE] = "Avian Influenza"
    
    # process locations which are given as place names, e.g. "Aberdeenshire, Scotland" (admin2, admin1)
    place_name_list = df_input_events["Location"].to_list()
    region_values = [place_name.split(",")[1] for place_name in place_name_list]
    city_values = [place_name.split(",")[0] for place_name in place_name_list]
    df_input_events[consts.COL_LOC_REGION] = region_values
    df_input_events[consts.COL_LOC_CITY] = city_values
    
    # process dates: we know only the year and week values.
    # >> for each entry, we retrieve the date corresponding to the first day of a given week in a year
    year_values = df_input_events["Year"].to_list()
    week_values = df_input_events["Week number carcass collected"].to_list()
    pub_dates = [date.fromisocalendar(year_values[i], week_values[i], 1) for i in range(len(year_values))]
    df_input_events[consts.COL_PUBLISHED_TIME] = pub_dates
    
    # process host values
    # in the host column, the scientific and common names of a host is combined, like here: "Red Fox (Vulpes vulpes)"
    # we retrieve only the scientific name
    host_vals = df_input_events["Species involved"].to_list()
    processed_host_vals = [h.split("(")[1].split(")")[0].strip() for h in host_vals]
    df_input_events["Species involved"] = processed_host_vals

    return df_input_events
  
  
  def determine_useful_columns(self):
    cols_events = [consts.COL_ARTICLE_ID, consts.COL_LOC_CONTINENT, consts.COL_LOC_COUNTRY, consts.COL_LOC_REGION, \
                   consts.COL_LOC_CITY, consts.COL_LAT, consts.COL_LNG, \
                   consts.COL_PUBLISHED_TIME, consts.COL_DISEASE, consts.COL_DISEASE_SUBTYPE, \
                   consts.COL_HOST]
    return(cols_events)


  def perform_preprocessing(self, disease_name, input_folder, out_preprocessing_result_folder, data_folder):
    super().perform_preprocessing(disease_name, input_folder, out_preprocessing_result_folder, data_folder)
    
    
    
    
class PreprocessingAphis(PreprocessingEmpresi):
  '''
    APHIS concrete class for event_preprocessing. This class extends the 'PreprocessingEmpresi' class.
    It renames the columns and does some basic event_preprocessing operations related to APHIS event data.
    It also add some new columns which are not provided by the input event file.
    For instance, there is no country information, because APHIS is specific to United Kingdom.
    It uses the same main event_preprocessing method implemented in the 'PreprocessingEmpresi' class.
    For information, the main trigger method for event_preprocessing is perform_preprocessing()
  '''
  
  def __init__(self):
    self.input_events_filename = "hpai-mammals.csv"
    self.output_raw_events_filename = "raw_events_"+consts.NEWS_DB_APHIS
    self.output_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_DB_APHIS
    self.output_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_DB_APHIS
    self.output_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_DB_APHIS
  

  def set_csv_separator(self):
    return(",")
  
  
  def determine_columns_types(self):
    dtypes = {}
    return(dtypes)
  
  
  def determine_columns_to_rename(self):
    rename_dict = {
      "State": consts.COL_LOC_REGION,
      "County": consts.COL_LOC_CITY,
      "HPAI Strain": consts.COL_DISEASE_SUBTYPE,
      "Species": consts.COL_HOST,
      "Date Detected": consts.COL_PUBLISHED_TIME
      }
    return rename_dict
  
  
  def postprocess_input_event_dataframe(self, df_input_events):
    df_input_events[consts.COL_ARTICLE_ID] = [consts.NEWS_DB_APHIS+"_"+str(i) for i in range(df_input_events.shape[0])]
    df_input_events[consts.COL_LOC_CONTINENT] = "Europe"
    df_input_events[consts.COL_LOC_COUNTRY] = "United Kingdom"
    df_input_events[consts.COL_LAT] = ""
    df_input_events[consts.COL_LNG] = ""
    df_input_events[consts.COL_DISEASE] = "Avian Influenza"
    return df_input_events
  
  
  def determine_useful_columns(self):
    cols_events = [consts.COL_ARTICLE_ID, consts.COL_LOC_CONTINENT, consts.COL_LOC_COUNTRY, consts.COL_LOC_REGION, \
                   consts.COL_LOC_CITY, consts.COL_LAT, consts.COL_LNG, \
                   consts.COL_PUBLISHED_TIME, consts.COL_DISEASE, consts.COL_DISEASE_SUBTYPE, \
                   consts.COL_HOST]
    return(cols_events)


  def perform_preprocessing(self, disease_name, input_folder, out_preprocessing_result_folder, data_folder):
    super().perform_preprocessing(disease_name, input_folder, out_preprocessing_result_folder, data_folder)




class PreprocessingBVBRC(PreprocessingEmpresi):
  '''
    BVBRC concrete class for event_preprocessing. This class extends the 'PreprocessingEmpresi' class.
    It only renames the columns and does some basic event_preprocessing operations related to BVBRC event data.
    It uses the same main event_preprocessing method implemented in the 'PreprocessingEmpresi' class.
    For information, the main trigger method for event_preprocessing is perform_preprocessing()
  '''

  def __init__(self, input_events_filename):
    self.input_events_filename = input_events_filename #"BVBRC_genome_preprocessed_adj.csv"
    self.output_raw_events_filename = "raw_events_" + consts.NEWS_DB_BVBRC
    self.output_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_" + consts.NEWS_DB_BVBRC
    self.output_preprocessed_disease_entities_filename = "preprocessed_disease_entities_" + consts.NEWS_DB_BVBRC
    self.output_preprocessed_host_entities_filename = "preprocessed_host_entities_" + consts.NEWS_DB_BVBRC

  def set_csv_separator(self):
    return (";")

  def determine_columns_types(self):
    dtypes = {"id": str}
    return (dtypes)

  def determine_columns_to_rename(self):
    rename_dict = {
      "id": consts.COL_ARTICLE_ID,
      "country_name": consts.COL_LOC_COUNTRY,
      "region_name": consts.COL_LOC_REGION,
      "city_name": consts.COL_LOC_CITY,
      "lng": consts.COL_LNG,
      "lat": consts.COL_LAT,
      "disease": consts.COL_DISEASE,
      "Serotype": consts.COL_DISEASE_SUBTYPE,
      "host": consts.COL_HOST,
      "ObservationDate": consts.COL_PUBLISHED_TIME,
      #"ReportDate": consts.COL_PUBLISHED_TIME,
      "Segment2GenomeID": consts.COL_SENTENCES
    }
    return rename_dict

  def postprocess_input_event_dataframe(self, df_input_events):
    df_input_events[consts.COL_LOC_CONTINENT] = ""
    return df_input_events

  def determine_useful_columns(self):
    cols_events = [consts.COL_ARTICLE_ID, consts.COL_LOC_CONTINENT, consts.COL_LOC_COUNTRY, consts.COL_LOC_REGION, \
                   consts.COL_LOC_CITY, consts.COL_LAT, consts.COL_LNG, \
                   consts.COL_PUBLISHED_TIME, consts.COL_DISEASE, consts.COL_DISEASE_SUBTYPE, \
                   consts.COL_HOST]
    return (cols_events)

  def perform_preprocessing(self, disease_name, input_folder, out_preprocessing_result_folder, data_folder):
    super().perform_preprocessing(disease_name, input_folder, out_preprocessing_result_folder, data_folder)
