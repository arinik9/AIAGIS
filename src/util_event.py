'''
Created on Sep 1, 2022

@author: nejat
'''

import src.consts as consts
import re
import unicodedata
import json

from src.event.event import Event
from src.event.temporality import Temporality
from src.event.disease import Disease
from src.event.symptom import Symptom
from src.event.host import Host
from src.event.hosts import Hosts
from src.event.location import Location

import pandas as pd
from iso3166 import countries
import csv
import dateutil.parser as parser


def build_disease_instance(disease_str):
  disease_tuple = eval(disease_str)
  dis_parts = disease_tuple[2].split(" ")  # example: "ai (unknown-pathogenicity)"
  dis_type = dis_parts[0]
  dis_pathogenicity = dis_parts[1].replace("(", "").replace(")", "").strip()
  dis = Disease(disease_tuple[0], disease_tuple[1], dis_type)
  dis.pathogenicity = dis_pathogenicity
  return dis


def read_df_events(events_filepath, extra_cols=None):
  cols_events = [consts.COL_ID, consts.COL_ARTICLE_ID, consts.COL_URL, consts.COL_SOURCE, \
                 consts.COL_GEONAMES_ID, "geoname_json", "loc_name", "loc_country_code", consts.COL_LOC_CONTINENT, \
                 consts.COL_LAT, consts.COL_LNG, "hierarchy_data", consts.COL_PUBLISHED_TIME, consts.COL_DISEASE, \
                 consts.COL_HOST, \
                 # consts.COL_SYMPTOM_SUBTYPE, consts.COL_SYMPTOM, \
                 # consts.COL_TITLE, consts.COL_SENTENCES, \
                 "day_no", "week_no", "month_no", "month_no", "year", "season"
                 ]
  if extra_cols is not None:
    cols_events.extend(extra_cols)
  df_events = pd.read_csv(events_filepath, usecols=cols_events, sep=";", keep_default_na=False)
  df_events[consts.COL_PUBLISHED_TIME] = df_events[consts.COL_PUBLISHED_TIME].apply(lambda x: parser.parse(x))

  df_events = df_events.rename(columns={"month_no": "month_no_simple"})
  df_events = df_events.rename(columns={"week_no": "week_no_simple"})
  df_events["month_no"] = df_events["month_no_simple"].apply(str) + "_" + df_events["year"].apply(str)
  df_events["week_no"] = df_events["week_no_simple"].apply(str) + "_" + df_events["year"].apply(str)
  df_events["season_no_simple"] = df_events["season"].replace(["winter", "spring", "summer", "autumn"], [1, 2, 3, 4])
  df_events["season_no"] = df_events["season_no_simple"].apply(str) + "_" + df_events["year"].apply(str)
  return df_events


def read_events_from_df(df_events):
  
  # df_events = read_df_events(events_filepath)

  events = []
  for index, row in df_events.iterrows():
    loc = Location(row["loc_name"], row[consts.COL_GEONAMES_ID], json.loads(row["geoname_json"]), \
                   row[consts.COL_LAT], row[consts.COL_LNG], row["loc_country_code"], row[consts.COL_LOC_CONTINENT], \
                   row["hierarchy_data"])
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
    h_vals = Hosts(h_list)

    sym = Symptom()
    # sym.load_dict_data_from_str(row[consts.COL_SYMPTOM_SUBTYPE], row[consts.COL_SYMPTOM])
    e = Event(int(row[consts.COL_ID]), row[consts.COL_ARTICLE_ID], row[consts.COL_URL], \
                    row[consts.COL_SOURCE], loc, t, dis, h_vals, sym, "", "")
    events.append(e)
    
  return events


def get_df_from_events(events):
  nb_events = len(events)
  if nb_events == 0:
    print("!! there are no events !!")
    return (-1)

  df_event_candidates = pd.DataFrame(columns=( \
    consts.COL_ID, consts.COL_ARTICLE_ID, consts.COL_URL, consts.COL_SOURCE, \
    consts.COL_GEONAMES_ID, "geoname_json",
    "loc_name", "loc_country_code", consts.COL_LOC_CONTINENT, \
    consts.COL_LAT, consts.COL_LNG, "hierarchy_data", \
    consts.COL_PUBLISHED_TIME,
    # consts.COL_DISEASE_SUBTYPE,
    consts.COL_DISEASE, \
    # consts.COL_HOST_SUBTYPE,
    consts.COL_HOST, consts.COL_SYMPTOM_SUBTYPE, consts.COL_SYMPTOM, \
    consts.COL_TITLE, consts.COL_SENTENCES
  ))
  for indx in range(0, nb_events):
    e = events[indx]
    df_event_candidates.loc[indx] = e.get_event_entry() + [e.title, e.sentences]

  # if signal_info_exists:
  #   signal_ids = [self.article_id_to_signal_ids[a_id] for a_id in df_event_candidates[consts.COL_ARTICLE_ID]]
  #   df_event_candidates[consts.COL_SIGNAL_ID] = signal_ids

  return df_event_candidates


def retrieve_disease_from_raw_sentence(sentence_text, disease_name):
  sentence_text = sentence_text.lower()
  clean_text = unicodedata.normalize("NFKD",sentence_text)

  # "hpai" and "lpai" are the first two elements in DISEASE_KEYWORDS_DICT >> intentionally
  disease_keywords = consts.DISEASE_KEYWORDS_DICT[disease_name].keys()
  dis_type = ""
  for kw in disease_keywords:
    #print(kw)
    parts = kw.split(" ")
    kw_pattern = ' '.join([p+".{0,2}" for p in parts])
    # I dont add a space in the beginning, maybe a phrase can start with the keyword
    #kw_pattern = " "+ kw_pattern # to ensure that our keyword is not contained in a string >> the found string should start with our keyword
    res = re.findall(kw_pattern, clean_text)
    if len(res)>0:
      dis_type = disease_name #consts.DISEASE_KEYWORDS_DICT[disease_name][kw]
      if kw == "hpai":
        dis_type = "HPAI"
      elif kw == "lpai":
        dis_type = "LPAI"
      break
      
  dis_subtype = ""
  if disease_name == consts.DISEASE_AVIAN_INFLUENZA:
    res = re.findall("h[0-9]{1,2}n[0-9]{1,2}", clean_text)
    res = [r.strip() for r in res]
    res2 = re.findall("h[0-9]{1,2}", clean_text)
    res2 = [r2.strip() for r2 in res2]
    # an event must have a single AI subtype
    # for now, we do not treat complex sentences
    if len(res)>0:
      dis_subtype = ','.join(res).strip()
      if len(res) > 1:
        dis_subtype = ""
      #dis_subtype = res[0] # take the first one, if there are multiple candidates # >> TODO we can improve it
      dis_type = disease_name
    elif len(res2)>0:
      dis_subtype = ','.join(res2).strip()
      if len(res2) > 1:
        dis_subtype = ""
      #dis_subtype = res2[0] # take the first one, if there are multiple candidates # >> TODO we can improve it
      dis_type = disease_name

  dis = None
  if dis_subtype != "" or dis_type != "":
    if disease_name == consts.DISEASE_AVIAN_INFLUENZA and dis_type == disease_name:
      dis_type = dis_type+"-unknown"
      if dis_subtype != "" and dis_subtype not in ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"]:
        if dis_subtype in ["h5n1", "h7n9", "h5n6", "h5n8"]:
          dis_type = "HPAI"
        else:
          dis_type = "LPAI"
    dis = Disease(dis_subtype, dis_type)
  return(dis)
  
  

def retrieve_host_from_raw_sentence(sentence_text, add_space_for_search=True):
  h = Host()
  sentence_text = sentence_text.lower()
  
  for host_type in list(consts.HOST_KEYWORDS_HIERARCHY_DICT.keys()): # hierarchy level 0
    host_subtype_data = consts.HOST_KEYWORDS_HIERARCHY_DICT[host_type] # it is a list of dicts with two keys: "text" and "hierarchy"
    # each entry in 'host_subtype_data["hierarchy"]' is organized from general to specialized tuple_data
    for lvl in [4,3,2,1,0]:
      if h.is_host_info_empty(): #if a human case already found, or another host type, we do not treat a new one
        for dict_entry in host_subtype_data:
          curr_lvl = dict_entry["level"]
          if lvl == curr_lvl:
            kw = dict_entry["text"]
            #print(kw)
            parts = kw.split(" ")
            kw_pattern = ' '.join([p+".{0,2}" for p in parts])
            if add_space_for_search: 
              kw_pattern = " "+ kw_pattern # to ensure that our keyword is not contained in a string >> the found string should start with our keyword
            res = re.findall(kw_pattern, sentence_text)
            if len(res)>0:
              #print("host:", host_type)
              # each 'd' is organized from general to specialized tuple_data
              d = dict_entry["hierarchy"]
              h_temp = Host()
              h_temp.add_host_info(d)
              if h.is_host_info_empty() or (not h.is_host_info_empty() and list(h.get_entry().keys())[0] == list(h_temp.get_entry().keys())[0]):
                h.add_host_info(d)
            
  
  # if len(h.get_entry()) == 0:
  #   return None   
  return(h)



def contains_ban_related_keyword(text):
  ban = False
  for kw in consts.BAN_KEYWORDS_LIST:
    parts = kw.split(" ")
    kw_pattern = ' '.join([" "+p+".{0,2}" for p in parts])
    if len(re.findall(kw_pattern, text))>0:
      ban = True
      break
  return ban


def simplify_df_events_at_hier_level1(events_filepath, new_events_filepath=None):
  df_events = read_df_events(events_filepath)

  country_code_list = []
  country_name_list = []
  for index, country_code in enumerate(df_events["loc_country_code"].to_list()):
    country_code_alpha2 = countries.get(country_code).alpha2
    country_code_list.append(country_code_alpha2)

    country_name = countries.get(country_code).name
    if country_code == "KOR":
      country_name = "South Korea"
    if country_code == "PRK":
      country_name = "North Korea"
    if country_code == "IRN":
      country_name = "Iran"
    if country_code == "GBR":
      country_name = "United Kingdom"
    if country_code == "USA":
      country_name = "United States"
    if country_code == "TWN":
      country_name = "Taiwan"
    if country_code == "RUS":
      country_name = "Russia"
    if country_code == "LAO":
      country_name = "Laon"
    country_name_list.append(country_name)

  region_name_list = []
  for index, geoname_json_str in enumerate(df_events["geoname_json"].to_list()):
    geoname_json = json.loads(geoname_json_str)
    region_name = ""
    if "adminName1" in geoname_json:
      region_name = geoname_json["adminName1"]
    region_name_list.append(region_name)

  city_name_list = []
  for index, geoname_json_str in enumerate(df_events["geoname_json"].to_list()):
    geoname_json = json.loads(geoname_json_str)
    # print(geoname_json)
    fcode = None
    if "fcode" in geoname_json:
      fcode = geoname_json["fcode"]
    if "code" in geoname_json:
      fcode = geoname_json["code"]
    if fcode is not None and fcode != "ADM1" and fcode != "PCLI" and fcode != "PCLS":
      city_name = geoname_json["toponymName"]
      city_name_list.append(city_name)
    else:
      city_name_list.append("")

  disease_name_list = []
  disease_serotype_list = []
  for index, diseae_info_str in enumerate(df_events["disease"].to_list()):
    diseae_info = eval(diseae_info_str)
    disease_name = diseae_info[-1]
    disease_name_list.append(disease_name)
    disease_serotype = diseae_info[0]
    disease_serotype_list.append(disease_serotype)

  host_list = []
  host_subtype_list = []
  for index, host_str in enumerate(df_events["host"].to_list()):
    h = [Host(d).get_entry()["hierarchy"][0] for d in eval(host_str)]
    hsub = [Host(d).get_entry()["common_name"] for d in eval(host_str)]
    host_list.append(h)
    host_subtype_list.append(hsub)

  data = {"country_code": country_code_list, "country": country_name_list, "region": region_name_list,
          "locality": city_name_list, \
          "disease": disease_name_list, "disease subtype": disease_serotype_list, \
          "host": host_list, "host subtype": host_subtype_list}
  data["id"] = df_events["id"].to_list()
  data["article_id"] = df_events["article_id"].to_list()
  data["url"] = df_events["url"].to_list()
  data["source"] = df_events["source"].to_list()
  data["continent"] = df_events["continent"].to_list()
  data["geonames_id"] = df_events["geonames_id"].to_list()
  data["lat"] = df_events["lat"].to_list()
  data["lng"] = df_events["lng"].to_list()
  data["published_at"] = df_events["published_at"].to_list()
  # data["symptom"] = df_events["symptom"].to_list()
  # data["symptom subtype"] = df_events["symptom subtype"].to_list()
  # data["title"] = df_events["title"].to_list()
  # data["sentences"] = df_events["sentences"].to_list()
  data["day_no"] = df_events["day_no"].to_list()
  data["week_no"] = df_events["week_no"].to_list()
  #data["week_no_simple"] = df_events["week_no"].apply(lambda x: int(x.split("_")[0])).to_list()
  data["month_no"] = df_events["month_no"].to_list()
  #data["month_no_simple"] = df_events["month_no"].apply(lambda x: int(x.split("_")[0])).to_list()
  data["year"] = df_events["year"].to_list()
  data["season"] = df_events["season"].to_list()
  df = pd.DataFrame(data)

  if new_events_filepath is not None:
    print(new_events_filepath)
    df.to_csv(new_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  return (df)
