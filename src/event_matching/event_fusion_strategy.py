'''
Created on Nov 23, 2021

@author: nejat
'''


from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from typing import List
import event.event as event
from event.hosts import Hosts

from event.temporality import Temporality

import numpy as np



# Another strategy: TruthFinder
# https://github.com/IshitaTakeshi/TruthFinder


class EventFusionStrategy(ABC):
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
  def perform_preprocessing(self, dataframe:pd.DataFrame, res_event_clustering:np.ndarray):
    pass
  
    
  @abstractmethod
  def merge_event_candidates(self, event_candidate_list:List) -> event.Event:
    pass


"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class EventFusionStrategyMaxOccurrence(EventFusionStrategy):
  
  def get_description(self) -> str:
    return("max-occurrence")
      
  
  def perform_preprocessing(self, dataframe:pd.DataFrame, res_event_clustering:np.ndarray):
    pass
  
        
  def merge_event_candidates(self, event_candidate_list:List) -> event.Event:
    ids = [e.e_id for e in event_candidate_list]
    article_ids = [e.article_id for e in event_candidate_list]
    urls = [e.url for e in event_candidate_list]
    sources = [e.source for e in event_candidate_list]
    #
    # loc
    loc_entries = [e.loc for e in event_candidate_list]
    final_loc = self.merge_location_entries(loc_entries)
    #
    # date
    dates = [e.date for e in event_candidate_list]
    final_date = self.merge_date_entries(dates)
    #
    # disease
    disease_entries = [e.disease for e in event_candidate_list]
    final_disease = self.merge_disease_entries(disease_entries)
    #
    # host
    host_entries = []
    for e in event_candidate_list:
      for h in e.host.get_entry():
        print(h)
        host_entries.append(h)
    final_host = self.merge_host_entries(host_entries)
    #
    # symptom
    symptom_entries = [e.symptom for e in event_candidate_list]
    final_symptom = self.merge_symptom_entries(symptom_entries)
    #
    # event construction s
    e = event.Event(ids, article_ids, urls, sources, final_loc, final_date, \
                     final_disease, final_host, final_symptom, "", "") # no default id
    
    return e



  def merge_location_entries(self, loc_entries):
    print(loc_entries)
    final_loc = loc_entries[0]
    for loc in loc_entries:
      if loc.is_spatially_included(final_loc):
        final_loc = loc
    return(final_loc)


  
  def merge_date_entries(self, date_entries):
    dates = [t.date for t in date_entries]
    dates.sort()
    final_date = Temporality(dates[0])
    return(final_date)

  
    
  def merge_disease_entries(self, disease_entries):
    final_disease = disease_entries[0]
    for disease in disease_entries:
      if disease.hierarchically_includes(final_disease):
      #if disease.is_hierarchically_included(final_disease):
        final_disease = disease
    return(final_disease)

 
  
  def merge_host_entries(self, host_entries):
    host_entries.sort(key=lambda h: h.get_entry()["level"], reverse=True)
    final_host = Hosts([host_entries[0]])
      
    host_to_remove = []
    host_to_add = []
    new_host = []
    for h_new in host_entries[1:]:
      for h_exis in final_host.get_entry():
        # if host_info.is_hierarchically_included(e.host) or (list(host_info.get_entry().keys())[0] == "human" and list(host_info.get_entry().keys())[0] != list(e.host.get_entry().keys())[0]):
        if h_new.hierarchically_includes(h_exis):
          host_to_remove.append(h_exis)
          host_to_add.append(h_new)
        else:
          new_host.append(h_new)
    for h_exis in host_to_remove:
      final_host.remove_host_info(h_exis)
    for h_new in host_to_add:
      final_host.add_host_info_if_success(h_new)
    for h_new in new_host:
      final_host.add_host_info_if_success(h_new)
          
    return(final_host)
  
  
  
  def merge_symptom_entries(self, symptom_entries):
    final_symptom = symptom_entries[0]
    return(final_symptom)  
  



  



