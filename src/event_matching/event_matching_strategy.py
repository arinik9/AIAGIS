'''
Created on Jan 12, 2022

@author: nejat
'''

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from typing import List

from src.event.event_duplicate_identification_strategy import EventDuplicateHierIdentificationStrategy
from src.event.event import Event

from src.myutil import haversine

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.event_matching.calculate_matrix_sim_values_in_parallel import calculate_matrix_sim_values_in_parallel

import csv
import os


class EventMatchingStrategy(ABC):
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
  def perform_event_matching(self, events1:List, events2:List, output_dirpath:str, result_filename1:str, result_filename2:str):
    pass





class EventMatchingStrategyPossiblyDuplicate(EventMatchingStrategy):
  
  def get_description(self) -> str:
    return("possibly_duplicate_in_event_matching")
      
  def findPossiblyDuplicateEvents(self, event:Event, other_event_list:List):
    event_duplication_strategy = EventDuplicateHierIdentificationStrategy()
    possibly_duplicate_event_ids = []
    for event_other in other_event_list:
      # 90+90 = 180 days
      if event_duplication_strategy.are_two_events_possibly_duplicate(event, event_other, 90):
        possibly_duplicate_event_ids.append(event_other.e_id)
    return(possibly_duplicate_event_ids)
    
    
  def perform_event_matching(self, events_padiweb, events_healthmap):
    print(self.get_description())
    
    possibly_duplicate_event_ids_list = []
    for e in events_padiweb:
      possibly_duplicate_event_ids = self.findPossiblyDuplicateEvents(e, events_healthmap)
      possibly_duplicate_event_ids_list.append("||".join([str(x) for x in possibly_duplicate_event_ids]))
      
    padiweb_event_ids = [e.e_id for e in events_padiweb]
    data = {"padiweb_event_id": padiweb_event_ids, "possibly_associated_healthmap_event_ids": possibly_duplicate_event_ids_list}
    df_event_matching = pd.DataFrame(data) 
    return df_event_matching
  
  
  
class EventMatchingStrategyEventSimilarity(EventMatchingStrategy):

  def __init__(self, event_sim_strategy):
    self.event_sim_strategy = event_sim_strategy

  def get_description(self) -> str:
    return("event_similarity_event_matching")
      
  def bind_eventSet1_to_eventSet2(self, events1_list, events2_list, out_sim_matrix_filepath):
    print("we are in bind_eventSet1_to_eventSet2() .....")
    id_to_event1 = {}
    for e in events1_list:
      id_to_event1[e.e_id] = e
    
    id_to_event2 = {}
    for e in events2_list:
      id_to_event2[e.e_id] = e
      #id_to_event2[e.article_id] = e
      
    # source: https://stackoverflow.com/questions/66874211/match-two-sets-of-objects
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html

    event1_id_to_index = dict(zip([s.e_id for s in events1_list], range(len(events1_list))))
    index_to_event1_id = dict(zip(range(len(events1_list)), [s.e_id for s in events1_list]))
    event2_id_to_index = dict(zip([e.e_id for e in events2_list], range(len(events2_list))))
    index_to_event2_id = dict(zip(range(len(events2_list)), [e.e_id for e in events2_list]))
    #index_to_event2_id = dict(zip(range(len(events2_list)), [e.article_id for e in events2_list]))

    # ======================================================
    if not os.path.exists(out_sim_matrix_filepath):
      print(out_sim_matrix_filepath)
      df_sim_values = calculate_matrix_sim_values_in_parallel(events1_list, events2_list, self.event_sim_strategy, nb_processes=8)
      df_sim_values.to_csv(out_sim_matrix_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)
    df_sim_values = pd.read_csv(out_sim_matrix_filepath, sep=";", keep_default_na=False)
    df_sim_values["pair_id"] = df_sim_values["event1_id"].astype(str) + "_" + df_sim_values["event2_id"].astype(str)
    pairId2score = dict(zip(df_sim_values["pair_id"], df_sim_values["sim_score"]))
    # ======================================================
  
    MAX_VALUE = 100000000
    cost_mat = np.zeros((len(events1_list), len(events2_list))) + MAX_VALUE # init with something very big value
    for event1 in events1_list:
      event1_id = event1.e_id
      for event2 in events2_list:
        event2_id = event2.e_id
        #sim_score = self.event_sim_strategy.perform_event_similarity(event1, event2, False)
        if str(event1_id)+"_"+str(event2_id) in pairId2score:
          sim_score = pairId2score[str(event1_id)+"_"+str(event2_id)]
          # if sim_score > -5:
          #   print("-----------")
          #   print(event1)
          #   print(event2)
          #   print("--", sim_score)
          cost_score = -sim_score
          cost_mat[event1_id_to_index[event1_id], event2_id_to_index[event2_id]] = cost_score

    # STRATEGY 1: solve this assignment problem
    #print("strategy 1: solve the assignment problem (1-to-1)")
    #row_ind, col_ind = linear_sum_assignment(cost_mat)
    
    # STRATEGY 2: get the max value for each row
    print("strategy 2: take the max (1-to-many)")
    col_ind = np.argmin(cost_mat, axis=1)
    row_ind = list(range(cost_mat.shape[0]))
    #print(col_ind)
    #print(row_ind)
    #print(cost_mat[row_ind, col_ind])

    assigned_event1_ids = []
    assigned_event2_ids = []
    sim_scores = []
    norm_sim_scores = []
    assigned_event1_loc_info = []
    assigned_event1_loc_hierarchy = []
    assigned_event1_disease_info = []
    assigned_event1_host_info = []
    assigned_event1_date_info = []
    assigned_event2_loc_info = []
    assigned_event2_disease_info = []
    assigned_event2_host_info = []
    assigned_event2_date_info = []
    assigned_event2_loc_hierarchy = []
    time_diff_list = []
    dist_diff_list = []
    for i in range(len(row_ind)):
      #if cost_mat[row_ind[i], col_ind[i]] < 3000:
      # we negate the cost to switch to our similarity score background
      # when the time diff is around 60, the score will be around -2 >> put your threshold based on this
      if (-cost_mat[row_ind[i], col_ind[i]]) > 0: # TODO: make this threshold more explicite in the function
        signal_id = index_to_event1_id[row_ind[i]]
        event_id = index_to_event2_id[col_ind[i]]
        assigned_event1_ids.append(signal_id)
        assigned_event2_ids.append(event_id)
        sim_score = -cost_mat[row_ind[i], col_ind[i]]
        sim_scores.append(sim_score)
        #norm_sim_scores.append(sim_score/(1+1+1+1)) # the values in the denominator are the max values for location, date, disease and host
        norm_sim_scores.append(sim_score/self.event_sim_strategy.norm_factor)
        event1 = id_to_event1[signal_id]
        event2 = id_to_event2[event_id]
        assigned_event1_loc_info.append(event1.loc.__str__())
        assigned_event1_loc_hierarchy.append(str(event1.loc.hierarchy_data))
        assigned_event1_disease_info.append(event1.disease.__str__())
        assigned_event1_host_info.append(event1.host.__str__())
        assigned_event1_date_info.append(event1.date.__str__())
        assigned_event2_loc_info.append(event2.loc.__str__())
        assigned_event2_loc_hierarchy.append(str(event2.loc.hierarchy_data))
        assigned_event2_disease_info.append(event2.disease.__str__())
        assigned_event2_host_info.append(event2.host.__str__())
        assigned_event2_date_info.append(event2.date.__str__())
        #
        dist_diff = haversine(event1.loc.lng, event1.loc.lat, event2.loc.lng, event2.loc.lat)
        if dist_diff < 1.0:
          dist_diff = 5.0 # force to be the minimal dist
        time_diff = (event1.date.get_entry()-event2.date.get_entry()).days
        if time_diff < 0:
          time_diff *= -1
        time_diff_list.append(time_diff)
        dist_diff_list.append(dist_diff)
          
    df = pd.DataFrame({
                      "time_diff": time_diff_list, \
                      "dist_diff": dist_diff_list, \
                      "sim_score": sim_scores, \
                      "norm_sim_score": norm_sim_scores, \
                      "event1_id": assigned_event1_ids,\
                        "event1_loc_info": assigned_event1_loc_info,\
                        "event1_loc_hierarchy": assigned_event1_loc_hierarchy,\
                        "event1_date": assigned_event1_date_info, \
                        "event1_disease": assigned_event1_disease_info, \
                        "event1_host": assigned_event1_host_info, \
                        "event2_id": assigned_event2_ids,\
                        "event2_loc_info": assigned_event2_loc_info, \
                        "event2_loc_hierarchy": assigned_event2_loc_hierarchy, \
                        "event2_date": assigned_event2_date_info,\
                        "event2_disease": assigned_event2_disease_info, \
                        "event2_host": assigned_event2_host_info
                        })
    return df
  
  
  
    
  def perform_event_matching(self, events_set1, events_set2, out_sim_matrix_filepath):
    print(self.get_description())

    df_event_matching = self.bind_eventSet1_to_eventSet2(events_set1, events_set2, out_sim_matrix_filepath)
    return df_event_matching
  
    
