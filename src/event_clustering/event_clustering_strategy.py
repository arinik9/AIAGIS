'''
Created on Dec 11, 2021

@author: nejat
'''

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import os

from src.event.event_duplicate_identification_strategy import EventDuplicateIdentificationStrategy

import numpy as np
import pandas as pd
import csv

# ==================================================================
# IDEA 1 for obtaining a signed graph from cosine similarity graph:
# first, calculate cosine similarity
# use this: https://github.com/furkangursoy/signed_backbones
# then, apply CC
# ==================================================================

# ==================================================================
# IDEA 2 for obtaining a signed graph from cosine similarity graph:
# first, calculate cosine similarity
# then, signed_score = cos_score*2 - 1 >> in the range of [-1,+1]
# finally, FILTERING STEP, as in (Arinik et al, 2017)
    #   an automatic method, consisting in applying k-means
    #   separately to the negative similarity values, and to the
    #   positive ones, with k = 2. This allows distinguishing the values that
    #   are close to zero from the other
# ==================================================================


# TODO: https://stackoverflow.com/questions/55619176/how-to-cluster-similar-sentences-using-bert?rq=1


class EventClusteringStrategy(ABC):
  """
  The Strategy interface declares operations common to all supported versions
  of some algorithm.
  
  The Context uses this interface to call the algorithm defined by Concrete
  Strategies.
  """
  @abstractmethod
  def get_description(self) -> str:
      pass
    
    
  # no output
  @abstractmethod
  def perform_clustering(self, event_canditates:List, output_dirpath:str, graph_filename:str, result_filename:str):
      pass


"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""
class EventClusteringStrategyHierDuplicate(EventClusteringStrategy):
  
  def __init__(self):
    pass
    
    
  def get_description(self) -> str:
    return("hier_duplicate")
      
      
  # no output
  def perform_clustering(self, event_duplicate_strategy:EventDuplicateIdentificationStrategy, event_canditates:List, output_dirpath:str, result_filename:str):
    nb_event_cands = len(event_canditates)
    id_to_event = {}
    possibly_duplicate_events_dict = {}
    for e in event_canditates:
      possibly_duplicate_events_dict[e.e_id] = [] # init
      id_to_event[e.e_id] = e
      
    # --------------------------------------------------------------------
    # PART 1: find possibly duplicate events (with different hierarchical differences)
    # --------------------------------------------------------------------
    for indx in range(nb_event_cands):
      e1 = event_canditates[indx]
      e1_id = e1.e_id
      for indx2 in range(nb_event_cands):
        if indx != indx2:
          #print("indx: " + str(indx) + ", indx2: " + str(indx2))
          e2 = event_canditates[indx2]
          e2_id = e2.e_id
          is_possibly_duplicate = event_duplicate_strategy.are_two_events_possibly_duplicate(e1, e2, verbose=False)
          if is_possibly_duplicate:
            possibly_duplicate_events_dict[e1_id].append(e2_id)
            

    # --------------------------------------------------------------------
    # PART 2: keep only events, where the spatial entities are not contained 
    #        by other events identified from the previous step.
    #        In the end, undeleted events will be the main events (without duplication)
    # --------------------------------------------------------------------
    events_to_remove = []
    for e_id in possibly_duplicate_events_dict.keys():
      if e_id not in events_to_remove:
        e_curr = id_to_event[e_id]
        e_others_identical = []
        for e_other_id in possibly_duplicate_events_dict[e_id]:
          e_other = id_to_event[e_other_id]
          if e_curr.loc.is_identical(e_other.loc):
            e_others_identical.append(e_other.e_id)
        #
        for e_other_id in possibly_duplicate_events_dict[e_id]:
          e_other = id_to_event[e_other_id]
          if e_curr.loc.spatially_contains(e_other.loc):
            for id in e_others_identical+[e_id]:
              events_to_remove.append(id)
            #
            extended_list = list(set(possibly_duplicate_events_dict[e_other_id] + e_others_identical))
            #if e_id in extended_list:
            #  extended_list.remove(e_id)
            possibly_duplicate_events_dict[e_other_id] = extended_list
            
    unique_events_to_remove = np.unique(events_to_remove)
    for e_id in unique_events_to_remove:
      del possibly_duplicate_events_dict[e_id]
      
      
    # Merge the events: multiple lines can correspond to the same event, see the example below
    # ex: 
    #  line 1: Paris (1): Paris (2), France (3), France (4)
    #  line 2: Paris (2): Paris (1), France (4) >> this one is contained in line 1, so delete it
    processed = {}
    events_to_remove2 = []
    for e_id in possibly_duplicate_events_dict.keys():
      if e_id not in processed.keys():
        processed[e_id] = 1
        e_curr = id_to_event[e_id]
        for e_other_id in possibly_duplicate_events_dict[e_id]:
          processed[e_other_id] = 1
          e_other = id_to_event[e_other_id]
          if e_curr.loc.is_identical(e_other.loc) and e_other_id in possibly_duplicate_events_dict:
            extended_list = list(set(possibly_duplicate_events_dict[e_id] + possibly_duplicate_events_dict[e_other_id]))
            if e_id in extended_list:
              extended_list.remove(e_id)
            possibly_duplicate_events_dict[e_id] = extended_list
      else:
        events_to_remove2.append(e_id)
      
    unique_events_to_remove = np.unique(events_to_remove2)
    for e_id in unique_events_to_remove:
      del possibly_duplicate_events_dict[e_id]
      
      


    # --------------------------------------------------------------------
    # PART 3: write into file: instead of clustering info, we write the ids of the possibly duplicate events
    # --------------------------------------------------------------------
    clusters = []
    for e in event_canditates:
      if e.e_id in possibly_duplicate_events_dict:
        grouping_info = str(e.e_id)
        if len(possibly_duplicate_events_dict[e.e_id])>0:
          grouping_info = str(e.e_id) + "," + ",".join([str(i) for i in possibly_duplicate_events_dict[e.e_id]])
        clusters.append(grouping_info)
          
    df = pd.DataFrame({"clustering": clusters})
    result_filepath = os.path.join(output_dirpath, result_filename)
    df.to_csv(result_filepath, index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)
        
   
  
