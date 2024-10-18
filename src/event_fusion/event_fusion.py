import os
import pandas as pd

import src.util as util

import src.consts as consts


from abc import ABC, abstractmethod


from src.event_fusion.event_fusion_strategy import EventFusionStrategy
from src.event_clustering.event_clustering import AbstractEventClustering
from src.event_retrieval.event_retrieval_strategy import EventRetrievalStrategy

from src.util_event import read_events_from_df
from src.util_event import get_df_from_events




class AbstractEventFusion(ABC):

  def __init__(self,  event_retrieval_strategy:EventRetrievalStrategy, event_clustering:AbstractEventClustering, event_fusion_strategy: EventFusionStrategy):
    print("--- using event fusion strategy: ", event_fusion_strategy.get_description())
    self.event_retrieval_strategy = event_retrieval_strategy
    self.event_clustering = event_clustering
    self.event_fusion_strategy = event_fusion_strategy
    
    
    
  def get_strategy(self):
    return(self.strategy)
  
  def set_strategy(self, strategy):
    self.strategy = strategy
    

      
  @abstractmethod
  def perform_event_fusion(self, doc_events_filepath, output_dirpath, clustering_filepath, result_filename):
    res_event_clustering = None
    if os.path.exists(clustering_filepath):
      res_event_clustering = pd.read_csv(clustering_filepath, header=None, dtype=str)[0].to_numpy()
    else:
      print("!! there are no event clustering result file for event fusion !!")
      return

    cols_event_candidates = [consts.COL_ID, consts.COL_ARTICLE_ID, consts.COL_URL, consts.COL_SOURCE, \
                        consts.COL_GEONAMES_ID, "geoname_json", "loc_name", "loc_country_code", consts.COL_LOC_CONTINENT, \
                        consts.COL_LAT, consts.COL_LNG, "hierarchy_data", consts.COL_PUBLISHED_TIME, consts.COL_DISEASE,  \
                       consts.COL_HOST, consts.COL_SYMPTOM_SUBTYPE, consts.COL_SYMPTOM, \
                        consts.COL_TITLE, consts.COL_SENTENCES,
                        # 'year', 'month_no', 'day_no', 'week_no', 'season'
                       ]
      
    if os.path.exists(doc_events_filepath):

      df_doc_events = pd.read_csv(doc_events_filepath, usecols=cols_event_candidates, sep=";", keep_default_na=False)

      if df_doc_events.shape[0]>0:
        # =====================================================================
        # this method does not affect the data frame
        self.event_fusion_strategy.perform_preprocessing(df_doc_events, res_event_clustering)
        # =====================================================================
    
        event_canditates = read_events_from_df(df_doc_events)
          
        id_to_event = {}
        for e in event_canditates:
          id_to_event[e.e_id] = e
        
        
        events = []
        # res_event_clustering does not contain a result of hard clustering ==> check the file structure
        for grouping_info_str in res_event_clustering:
          e = id_to_event[int(grouping_info_str.split(",")[0])] # for single events
          if "," in grouping_info_str: # for more than one event
            e_list = [id_to_event[int(id)] for id in grouping_info_str.split(",")]
            # ===========================================================================
            # EVENT FUSION
            e = self.event_fusion_strategy.merge_event_candidates(e_list)
            # ===========================================================================
    
          events.append(e)
          
        df_events = get_df_from_events(events)
       
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
        
        df_events[consts.COL_ID] = list(range(df_events.shape[0]))
         
        events_filepath = os.path.join(output_dirpath, result_filename)
        df_events.to_csv(events_filepath, sep=";")


      else:
        print("!! there are no events for event clustering !!")
    else:
      print("!! there are no event file for event fusion !!")


class EventFusionPadiweb(AbstractEventFusion):

  def __init__(self,  event_retrieval_strategy:EventRetrievalStrategy, event_clustering:AbstractEventClustering, event_fusion_strategy: EventFusionStrategy):
    super().__init__(event_retrieval_strategy, event_clustering, event_fusion_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_SURVEILLANCE_PADIWEB + "_task1=" +  event_retrieval_strategy.get_description()
    self.in_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_SURVEILLANCE_PADIWEB + event_clustering.get_description()
    self.out_corpus_event_filename = "corpus_events_" + consts.NEWS_SURVEILLANCE_PADIWEB + "_task1=" + event_fusion_strategy.get_description()


       
  def perform_event_fusion(self, in_doc_events_folder, in_clustering_folder, out_fusion_folder):
    doc_events_filepath = os.path.join(in_doc_events_folder, self.in_doc_events_filename + "." + consts.FILE_FORMAT_CSV)
    clustering_filepath = os.path.join(in_clustering_folder, self.in_doc_event_clustering_filename + "." + consts.FILE_FORMAT_TXT)
    result_filename = self.out_corpus_event_filename + "." + consts.FILE_FORMAT_CSV
    super().perform_event_fusion(doc_events_filepath, out_fusion_folder, clustering_filepath, result_filename)
    
    
    
    
class EventFusionPromed(EventFusionPadiweb):

  def __init__(self,  event_retrieval_strategy:EventRetrievalStrategy, event_clustering:AbstractEventClustering, event_fusion_strategy: EventFusionStrategy):
    super().__init__(event_retrieval_strategy, event_clustering, event_fusion_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_SURVEILLANCE_PROMED + "_task1=" +  event_retrieval_strategy.get_description()
    self.in_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_SURVEILLANCE_PROMED + event_clustering.get_description()
    self.out_corpus_event_filename = "corpus_events_" + consts.NEWS_SURVEILLANCE_PROMED + "_task1=" + event_fusion_strategy.get_description()
        
        
  def perform_event_fusion(self, in_doc_events_folder, in_clustering_folder, out_fusion_folder):
    super().perform_event_fusion(in_doc_events_folder, in_clustering_folder, out_fusion_folder)
    
    
    
    
    
    
class EventFusionEmpresi(EventFusionPadiweb):

  def __init__(self,  event_retrieval_strategy:EventRetrievalStrategy, event_clustering:AbstractEventClustering, event_fusion_strategy: EventFusionStrategy):
    super().__init__(event_retrieval_strategy, event_clustering, event_fusion_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_DB_EMPRESS_I + "_task1=" +  event_retrieval_strategy.get_description()
    self.in_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_DB_EMPRESS_I + event_clustering.get_description()
    self.out_corpus_event_filename = "corpus_events_" + consts.NEWS_DB_EMPRESS_I + "_task1=" + event_fusion_strategy.get_description()
        
        
  def perform_event_fusion(self, in_doc_events_folder, in_clustering_folder, out_fusion_folder):
    super().perform_event_fusion(in_doc_events_folder, in_clustering_folder, out_fusion_folder)
    
    
    
    
    
    
class EventFusionWahis(EventFusionPadiweb):

  def __init__(self,  event_retrieval_strategy:EventRetrievalStrategy, event_clustering:AbstractEventClustering, event_fusion_strategy: EventFusionStrategy):
    super().__init__(event_retrieval_strategy, event_clustering, event_fusion_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_DB_WAHIS + "_task1=" +  event_retrieval_strategy.get_description()
    self.in_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_DB_WAHIS + event_clustering.get_description()
    self.out_corpus_event_filename = "corpus_events_" + consts.NEWS_DB_WAHIS + "_task1=" + event_fusion_strategy.get_description()
        
        
  def perform_event_fusion(self, in_doc_events_folder, in_clustering_folder, out_fusion_folder):
    super().perform_event_fusion(in_doc_events_folder, in_clustering_folder, out_fusion_folder)
    
    
    
    
    
class EventFusionApha(EventFusionPadiweb):

  def __init__(self,  event_retrieval_strategy:EventRetrievalStrategy, event_clustering:AbstractEventClustering, event_fusion_strategy: EventFusionStrategy):
    super().__init__(event_retrieval_strategy, event_clustering, event_fusion_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_DB_APHA + "_task1=" +  event_retrieval_strategy.get_description()
    self.in_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_DB_APHA + event_clustering.get_description()
    self.out_corpus_event_filename = "corpus_events_" + consts.NEWS_DB_APHA + "_task1=" + event_fusion_strategy.get_description()
        

  def perform_event_fusion(self, in_doc_events_folder, in_clustering_folder, out_fusion_folder):
    super().perform_event_fusion(in_doc_events_folder, in_clustering_folder, out_fusion_folder)
    
  
  
  
  
    
class EventFusionAphis(EventFusionPadiweb):

  def __init__(self,  event_retrieval_strategy:EventRetrievalStrategy, event_clustering:AbstractEventClustering, event_fusion_strategy: EventFusionStrategy):
    super().__init__(event_retrieval_strategy, event_clustering, event_fusion_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_DB_APHIS + "_task1=" +  event_retrieval_strategy.get_description()
    self.in_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_DB_APHIS + event_clustering.get_description()
    self.out_corpus_event_filename = "corpus_events_" + consts.NEWS_DB_APHIS + "_task1=" + event_fusion_strategy.get_description()
        
        
  def perform_event_fusion(self, in_doc_events_folder, in_clustering_folder, out_fusion_folder):
    super().perform_event_fusion(in_doc_events_folder, in_clustering_folder, out_fusion_folder)





class EventFusionBVBRC(EventFusionPadiweb):

  def __init__(self, event_retrieval_strategy: EventRetrievalStrategy, event_clustering: AbstractEventClustering,
               event_fusion_strategy: EventFusionStrategy):
    super().__init__(event_retrieval_strategy, event_clustering, event_fusion_strategy)
    self.in_doc_events_filename = "doc_events_" + consts.NEWS_DB_BVBRC + "_task1=" + event_retrieval_strategy.get_description()
    self.in_doc_event_clustering_filename = "doc_event_clustering_" + consts.NEWS_DB_BVBRC + event_clustering.get_description()
    self.out_corpus_event_filename = "corpus_events_" + consts.NEWS_DB_BVBRC + "_task1=" + event_fusion_strategy.get_description()

  def perform_event_fusion(self, in_doc_events_folder, in_clustering_folder, out_fusion_folder):
    super().perform_event_fusion(in_doc_events_folder, in_clustering_folder, out_fusion_folder)

