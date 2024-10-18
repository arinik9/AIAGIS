import os
import json
import pandas as pd
import src.event.event as event

from src.util_event import read_events_from_df

import src.consts as consts

from src.event_retrieval.event_retrieval_strategy import EventRetrievalStrategy
from abc import ABC, abstractmethod

from src.event.location import Location
from src.event.temporality import Temporality
from src.event.hosts import Hosts
from src.event.disease import Disease
from src.event.symptom import Symptom

from src.event.event_duplicate_identification_strategy import EventDuplicateIdentificationStrategy
from src.event_clustering.event_clustering_strategy import EventClusteringStrategy



class AbstractEventClustering(ABC):

  def __init__(self, event_retrieval_strategy:EventRetrievalStrategy, \
               event_duplicate_ident_strategy:EventDuplicateIdentificationStrategy, event_clustering_strategy:EventClusteringStrategy):
    self.event_retrieval_strategy = event_retrieval_strategy
    self.event_duplicate_ident_strategy = event_duplicate_ident_strategy
    self.event_clustering_strategy = event_clustering_strategy
  
    

  def get_description(self):
    return("_task1=" +  self.event_duplicate_ident_strategy.get_description() \
           + "_task2=" +  self.event_clustering_strategy.get_description())
    
        
  @abstractmethod
  def perform_event_clustering(self, doc_events_filepath, output_dirpath, result_filename):
    cols_doc_events = [consts.COL_ID, consts.COL_ARTICLE_ID, consts.COL_URL, consts.COL_SOURCE, \
                            consts.COL_GEONAMES_ID, "geoname_json", "loc_name", "loc_country_code", consts.COL_LOC_CONTINENT, \
                            consts.COL_LAT, consts.COL_LNG, "hierarchy_data", consts.COL_PUBLISHED_TIME, consts.COL_DISEASE,  \
                           consts.COL_HOST, consts.COL_SYMPTOM_SUBTYPE, consts.COL_SYMPTOM, \
                            consts.COL_TITLE, consts.COL_SENTENCES
                           ]
    
    if os.path.exists(doc_events_filepath):
      df_doc_events = pd.read_csv(doc_events_filepath, usecols=cols_doc_events, sep=";", keep_default_na=False)
    
      if df_doc_events.shape[0]>0:
        doc_events = read_events_from_df(df_doc_events)
    
        # perform clustering
        self.event_clustering_strategy.perform_clustering(self.event_duplicate_ident_strategy, doc_events, output_dirpath, result_filename)
      else:
        print("!! there are no events for event clustering !!")
    else:
      print("!! there are no event file for event clustering !!")
    

# class EventClusteringDoNothing(AbstractEventClustering):
#
#   def __init__(self, event_duplicate_ident_strategy:EventDuplicateIdentificationStrategy, event_clustering_strategy:EventClusteringStrategy):
#     super().__init__(event_duplicate_ident_strategy, event_clustering_strategy)
#
#
#   def perform_event_clustering(self, in_doc_events_folder, out_clustering_folder):
#     self.out_csv_folder = in_doc_events_folder
#     self.out_eval_clustering_folder = out_clustering_folder
#
#
#     doc_events_filepath = os.path.join(self.out_csv_folder, self.in_doc_events_filename + "." + consts.FILE_FORMAT_CSV)
#     output_dirpath = self.out_eval_clustering_folder
#
#     try:
#       if not os.path.exists(output_dirpath):
#         os.makedirs(output_dirpath)
#     except OSError as err:
#        print(err)
#
#     result_filename = self.out_doc_event_clustering_filename + "." + consts.FILE_FORMAT_TXT
#     super().perform_event_clustering(doc_events_filepath, output_dirpath, result_filename)
  
  
  
    
    
class EventClusteringPadiweb(AbstractEventClustering):

  def __init__(self, event_retrieval_strategy:EventRetrievalStrategy, \
               event_duplicate_ident_strategy:EventDuplicateIdentificationStrategy, event_clustering_strategy:EventClusteringStrategy):
    super().__init__(event_retrieval_strategy, event_duplicate_ident_strategy, event_clustering_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_SURVEILLANCE_PADIWEB + "_task1=" +  event_retrieval_strategy.get_description()
    self.out_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_SURVEILLANCE_PADIWEB + self.get_description()


  def perform_event_clustering(self, in_doc_events_folder, out_clustering_folder):
    in_doc_events_filepath = os.path.join(in_doc_events_folder, self.in_doc_events_filename + "." + consts.FILE_FORMAT_CSV)
    out_doc_event_clustering_filepath = self.out_doc_event_clustering_filename + "." + consts.FILE_FORMAT_TXT
    super().perform_event_clustering(in_doc_events_filepath, out_clustering_folder, out_doc_event_clustering_filepath)
    
    
    

class EventClusteringEmpresi(EventClusteringPadiweb):

  def __init__(self, event_retrieval_strategy:EventRetrievalStrategy, \
               event_duplicate_ident_strategy:EventDuplicateIdentificationStrategy, event_clustering_strategy:EventClusteringStrategy):
    super().__init__(event_retrieval_strategy, event_duplicate_ident_strategy, event_clustering_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_DB_EMPRESS_I + "_task1=" +  event_retrieval_strategy.get_description()
    self.out_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_DB_EMPRESS_I + self.get_description()


  def perform_event_clustering(self, in_doc_events_folder, out_clustering_folder):
    super().perform_event_clustering(in_doc_events_folder, out_clustering_folder)




class EventClusteringWahis(EventClusteringPadiweb):

  def __init__(self, event_retrieval_strategy:EventRetrievalStrategy, \
               event_duplicate_ident_strategy:EventDuplicateIdentificationStrategy, event_clustering_strategy:EventClusteringStrategy):
    super().__init__(event_retrieval_strategy, event_duplicate_ident_strategy, event_clustering_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_DB_WAHIS + "_task1=" +  event_retrieval_strategy.get_description()
    self.out_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_DB_WAHIS + self.get_description()


  def perform_event_clustering(self, in_doc_events_folder, out_clustering_folder):
    super().perform_event_clustering(in_doc_events_folder, out_clustering_folder)
       
    
    

class EventClusteringPromed(EventClusteringPadiweb):

  def __init__(self, event_retrieval_strategy:EventRetrievalStrategy, \
               event_duplicate_ident_strategy:EventDuplicateIdentificationStrategy, event_clustering_strategy:EventClusteringStrategy):
    super().__init__(event_retrieval_strategy, event_duplicate_ident_strategy, event_clustering_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_SURVEILLANCE_PROMED + "_task1=" +  event_retrieval_strategy.get_description()
    self.out_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_SURVEILLANCE_PROMED + self.get_description()


  def perform_event_clustering(self, in_doc_events_folder, out_clustering_folder):
    super().perform_event_clustering(in_doc_events_folder, out_clustering_folder)
 




class EventClusteringApha(EventClusteringPadiweb):

  def __init__(self, event_retrieval_strategy:EventRetrievalStrategy, \
               event_duplicate_ident_strategy:EventDuplicateIdentificationStrategy, event_clustering_strategy:EventClusteringStrategy):
    super().__init__(event_retrieval_strategy, event_duplicate_ident_strategy, event_clustering_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_DB_APHA + "_task1=" +  event_retrieval_strategy.get_description()
    self.out_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_DB_APHA + self.get_description()


  def perform_event_clustering(self, in_doc_events_folder, out_clustering_folder):
    super().perform_event_clustering(in_doc_events_folder, out_clustering_folder)
        
        



class EventClusteringAphis(EventClusteringPadiweb):

  def __init__(self, event_retrieval_strategy:EventRetrievalStrategy, \
               event_duplicate_ident_strategy:EventDuplicateIdentificationStrategy, event_clustering_strategy:EventClusteringStrategy):
    super().__init__(event_retrieval_strategy, event_duplicate_ident_strategy, event_clustering_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_DB_APHIS + "_task1=" +  event_retrieval_strategy.get_description()
    self.out_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_DB_APHIS + self.get_description()


  def perform_event_clustering(self, in_doc_events_folder, out_clustering_folder):
    super().perform_event_clustering(in_doc_events_folder, out_clustering_folder)
    
        



class EventClusteringBVBRC(EventClusteringPadiweb):

  def __init__(self, event_retrieval_strategy:EventRetrievalStrategy, \
               event_duplicate_ident_strategy:EventDuplicateIdentificationStrategy, event_clustering_strategy:EventClusteringStrategy):
    super().__init__(event_retrieval_strategy, event_duplicate_ident_strategy, event_clustering_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_DB_BVBRC + "_task1=" +  event_retrieval_strategy.get_description()
    self.out_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_DB_BVBRC + self.get_description()


  def perform_event_clustering(self, in_doc_events_folder, out_clustering_folder):
    super().perform_event_clustering(in_doc_events_folder, out_clustering_folder)