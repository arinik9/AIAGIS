'''
Created on Sep 1, 2022

@author: nejat
'''

import json
import pandas as pd

import src.consts as consts
from src.event.event import Event
from src.event.temporality import Temporality
from src.event.disease import Disease
from src.event.symptom import Symptom
from src.event.host import Host
from src.event.hosts import Hosts
from src.event.location import Location






def read_df_events(events_filepath):
  cols_events = [consts.COL_ID, consts.COL_ARTICLE_ID, consts.COL_URL, consts.COL_SOURCE, \
                    consts.COL_GEONAMES_ID, "geoname_json", "loc_name", "loc_country_code", consts.COL_LOC_CONTINENT, \
                    consts.COL_LAT, consts.COL_LNG, "hierarchy_data", consts.COL_PUBLISHED_TIME, consts.COL_DISEASE,  \
                   consts.COL_HOST, consts.COL_SYMPTOM_SUBTYPE, consts.COL_SYMPTOM, \
                    consts.COL_TITLE, consts.COL_SENTENCES, \
                    "day_no", "week_no", "month_no", "month_no", "year", "season"
                   ]
  df_events = pd.read_csv(events_filepath, usecols=cols_events, sep=";", keep_default_na=False)
  return df_events




def read_events_from_df(df_events):
  
  # df_events = read_df_events(events_filepath)

  events = []
  for index, row in df_events.iterrows():
    loc = Location(row["loc_name"], row[consts.COL_GEONAMES_ID], json.loads(row["geoname_json"]), \
                   row[consts.COL_LAT], row[consts.COL_LNG], row["loc_country_code"], row[consts.COL_LOC_CONTINENT], \
                   eval(row["hierarchy_data"]))
    t = Temporality(row[consts.COL_PUBLISHED_TIME])
    disease_tuple = eval(row[consts.COL_DISEASE])
    dis_parts = disease_tuple[2].split(" ") # example: "ai (unknown-pathogenicity)"
    dis_type = dis_parts[0]
    dis_pathogenicity = dis_parts[1].replace("(","").replace(")","").strip()
    dis = Disease(disease_tuple[0], disease_tuple[1], dis_type)
    dis.pathogenicity = dis_pathogenicity
    h_list = []
    for d in eval(row[consts.COL_HOST]):
      h = Host(d)
      h_list.append(h)
    h = Hosts(h_list)
    sym = Symptom()
    # sym.load_dict_data_from_str(row[consts.COL_SYMPTOM_SUBTYPE], row[consts.COL_SYMPTOM])
    e = Event(int(row[consts.COL_ID]), row[consts.COL_ARTICLE_ID], row[consts.COL_URL], \
                    row[consts.COL_SOURCE], loc, t, dis, h, sym, row[consts.COL_TITLE], row[consts.COL_SENTENCES])
    events.append(e)
    
  return events



def get_df_from_events(events):
  nb_events  = len(events)
  if nb_events == 0:
    print("!! there are no events !!")
    return(-1)
  
  df_events = pd.DataFrame(columns=( \
                         consts.COL_ID, consts.COL_ARTICLE_ID, consts.COL_URL, consts.COL_SOURCE, \
                         consts.COL_GEONAMES_ID, "geoname_json",
                         "loc_name", "loc_country_code", consts.COL_LOC_CONTINENT, \
                         consts.COL_LAT, consts.COL_LNG, "hierarchy_data", \
                         consts.COL_PUBLISHED_TIME, 
                         # consts.COL_DISEASE_SUBTYPE, 
                         consts.COL_DISEASE,  \
                         # consts.COL_HOST_SUBTYPE, 
                         consts.COL_HOST, consts.COL_SYMPTOM_SUBTYPE, consts.COL_SYMPTOM, \
                         consts.COL_TITLE, consts.COL_SENTENCES
                         ))
  for indx in range(0,nb_events):
    e = events[indx]
    df_events.loc[indx] = e.get_event_entry() + [e.title, e.sentences]
    
  # if signal_info_exists:
  #   signal_ids = [self.article_id_to_signal_ids[a_id] for a_id in df_event_candidates[consts.COL_ARTICLE_ID]]
  #   df_event_candidates[consts.COL_SIGNAL_ID] = signal_ids
  
  return df_events

