'''
Created on Dec 11, 2021

@author: nejat
'''

from abc import ABC, abstractmethod

import event
from src.myutil import haversine
from src.event_matching.calculate_genome_similarity import calculate_isolate_similarity

import json


class EventSimilarityStrategy(ABC):
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
  def perform_event_similarity(self, event1:event, event2:event):
      pass


class EventSimilarityStrategyIsolateGenome(EventSimilarityStrategy):

  def __init__(self, id2segmentGenome1, id2segmentGenome2, id2seq):
    self.id2segmentGenome1 = id2segmentGenome1
    self.id2segmentGenome2 = id2segmentGenome2
    self.id2seq = id2seq
    self.norm_factor = 1

  def get_description(self) -> str:
    return ("isolate_event_similarity")

  def perform_event_similarity(self, event1: event, event2: event, verbose=False):
    event1_id = event1.e_id
    event2_id = event2.e_id
    event1_isolate = json.loads(self.id2segmentGenome1[event1_id])
    event2_isolate = json.loads(self.id2segmentGenome2[event2_id])
    score, genome_state = calculate_isolate_similarity(event1_isolate, event2_isolate, self.id2seq)
    return(score)

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""
class EventSimilarityStrategyManual(EventSimilarityStrategy):

  def __init__(self):
    self.norm_factor = 4 # 1+1+1+1


  def get_description(self) -> str:
    return("manual_event_similarity")
      
      
  def perform_event_similarity(self, event1:event, event2:event, verbose=False):
    signed_score = 0
    # loc
    loc_score = self.compute_location_similarity(event1.loc, event2.loc)
    if verbose:
      print("loc", loc_score)
    # date
    date_score = self.compute_date_similarity(event1.date, event2.date)
    if verbose:
      print("date", date_score)
    # host
    host_score = self.compute_host_similarity(event1.host, event2.host)
    if verbose:
      print("host", host_score)
    # disease
    disease_score = self.compute_disease_similarity(event1.disease, event2.disease)
    if verbose:
      print("disease", disease_score)
    # symptom
    symptom_score = self.compute_symptom_similarity(event1.symptom, event2.symptom)
    if verbose:
      print("symptom", symptom_score)
    # final score
    signed_score = signed_score + loc_score + date_score + disease_score \
                  + host_score + symptom_score
                  
    # if signed_score > -5:
    #   print("loc", loc_score)
    #   print("date", date_score)
    #   print("host", host_score)
    #   print("disease", disease_score)
    return signed_score


  # # remark: root node ROOT, which is something useless, must have level 0
  # def compute_location_similarity(self, l1, l2):
  #   score = -10 # default value >> penalization factor
  #   hier_related = False
  #   if l1.is_identical(l2):
  #     score = 1.0
  #   elif l1.spatially_contains(l2) or l1.is_spatially_included(l2): # hierarchically related
  #       # workaround: since we initially designed that country level is level 0, we change this for this calculation
  #       # so, the country level is 2
  #       lvl1 = l1.get_hierarchy_level()
  #       lvl1 += 2
  #       lvl2 = l2.get_hierarchy_level()
  #       lvl2 += 2
  #       common_ancestor_hier_level = min(lvl1, lvl2)-1
  #       score = (2*common_ancestor_hier_level)/(lvl1+lvl2)
  #   return(score)

  # remark: root node ROOT, which is something useless, must have level 0
  def compute_location_similarity(self, l1, l2):
    dist = haversine(l1.lng, l1.lat, l2.lng, l2.lat) # in km
    #score = -10 # default value >> penalization factor
    score = -dist/500  # default value >> penalization factor >> for dist=1000, pen score is -2
    #score = -dist / 1000  # default value >> penalization factor >> for dist=1000, pen score is -1
    hier_related = False
    if l1.is_identical(l2):
      score = 1.0
    elif l1.spatially_contains(l2) or l1.is_spatially_included(l2): # hierarchically related
        # workaround: since we initially designed that country level is level 0, we change this for this calculation
        # so, the country level is 2
        lvl1 = l1.get_hierarchy_level()
        lvl1 += 2
        lvl2 = l2.get_hierarchy_level()
        lvl2 += 2
        common_ancestor_hier_level = min(lvl1, lvl2)-1
        score = (2*common_ancestor_hier_level)/(lvl1+lvl2)
    return(score)
  
    
  
  def compute_date_similarity(self, t1, t2):
    # Timeliness is the proportion of time saved by detection relative to the other event.
    # source: Jafarpour2015
    L = 30 # max window size # 21 days, i.e. 3 weeks
    diff = abs((t2.date - t1.date).days) # difference in days
    #score = 1 - (diff/L) # old formula >> when the time diff is around 90, the score will be around -3 >> put your threshold based on this
    #tolerance_delay = 3 # 3 months >> # why 3 ? 90/30 = 3; so a delay of 3 months is okay to get a positive value in the comparison
    tolerance_delay = 1
    score = tolerance_delay - (diff / L)
    score = score/tolerance_delay # quasi-normalized score, in practice in [-1,1]
    return(score)
  
  
  # remark: root node ROOT, which is something useless, must have level 0
  def compute_host_similarity(self, hl1, hl2):
    best_score = -10 # default value >> penalization factor
    for h1 in hl1.get_entry():
      for h2 in hl2.get_entry():
        if h1.is_identical(h2):
          return(1.0) # best score is 1 when identical
        hier_related = False
        if h1.hierarchically_includes(h2) or h1.is_hierarchically_included(h2):
          hier_related = True
        if hier_related:
          # so, the bird or human level is 1
          lvl1 = h1.get_entry()["level"]
          #lvl1 += 1
          lvl2 = h2.get_entry()["level"]
          #lvl2 += 1
          common_ancestor_hier_level = min(lvl1, lvl2)-1
          score = (2*common_ancestor_hier_level)/(lvl1+lvl2)
          if best_score < score:
            best_score = score
    return(best_score)
  
  
  # remark: root node ROOT, which is something useless, must have level 0
  def compute_disease_similarity(self, d1, d2):
    score = -10 # default value >> penalization factor
    if d1.is_identical(d2):
      score = 1.0
    elif d1.hierarchically_includes(d2) or d1.is_hierarchically_included(d2): # hierarchically related
      # so, the AI or WNV level is 1
      lvl1 = d1.get_max_hierarchy_level()
      #lvl1 += 1
      lvl2 = d2.get_max_hierarchy_level()
      #lvl2 += 1
      common_ancestor_hier_level = min(lvl1, lvl2)-1
      score = (2*common_ancestor_hier_level)/(lvl1+lvl2)
    return(score)
  
  
  def compute_symptom_similarity(self, s1, s2):
    score = 0
    # if len(s1.dict_data.keys())>0 and len(s2.dict_data.keys())>0:
    #   common_keys = [x for x in s1.dict_data.keys() if x in s2.dict_data.keys()]
    #   if len(common_keys)>0:
    #     for key in common_keys:
    #       values = s1.dict_data[key]
    #       other_values = s2.dict_data[key]
    #       common_values = [x for x in values if x in other_values]
    #       if len(common_values)>0:
    #         score = score + len(common_values)*10
    #       else:
    #         score = score - 100
    #   else:
    #     score = score - 300
        
    return(score)  
  
  
  
    
