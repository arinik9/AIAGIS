'''
Created on Nov 16, 2021

@author: nejat
'''

import src.user_consts as user_consts
import os
import pandas as pd


# ====================================================
# Event-Based Surveillance (EBS) Platforms
# ====================================================
NEWS_SURVEILLANCE_PADIWEB = "padiweb"
NEWS_SURVEILLANCE_PROMED = "promed"
#NEWS_SURVEILLANCE_HEALTHMAP = "healthmap"
NEWS_DB_EMPRESS_I = "empres-i"
NEWS_DB_WAHIS = "wahis"
NEWS_DB_APHA = "apha"
NEWS_DB_APHIS = "aphis"
NEWS_DB_BVBRC = "bvbrc"



# ====================================================
# Folders
# ====================================================
DATA_FOLDER = os.path.join(user_consts.MAIN_FOLDER, "data")
IN_FOLDER = os.path.join(user_consts.MAIN_FOLDER, "in")
OUT_FOLDER = os.path.join(user_consts.MAIN_FOLDER, "out")

IN_PADIWEB_FOLDER = os.path.join(IN_FOLDER, NEWS_SURVEILLANCE_PADIWEB)
IN_PROMED_FOLDER = os.path.join(IN_FOLDER, NEWS_SURVEILLANCE_PROMED)
IN_EMPRESSI_FOLDER = os.path.join(IN_FOLDER, NEWS_DB_EMPRESS_I)
IN_WAHIS_FOLDER = os.path.join(IN_FOLDER,NEWS_DB_WAHIS)
IN_APHA_FOLDER = os.path.join(IN_FOLDER, NEWS_DB_APHA)
IN_APHIS_FOLDER = os.path.join(IN_FOLDER, NEWS_DB_APHIS)
IN_BVBRC_FOLDER = os.path.join(IN_FOLDER, NEWS_DB_BVBRC)
#IN_BVBRC_PRECISE_FOLDER = os.path.join(IN_FOLDER, "bvbrc_precise_serovar")
#IN_BVBRC_IMPRECISE_FOLDER = os.path.join(IN_FOLDER, "bvbrc_imprecise_serovar")

IN_MAP_SHAPEFILE_FOLDER = os.path.join(DATA_FOLDER, "map_shapefiles")
#IN_EXT_DATA_FOLDER = DATA_FOLDER # os.path.join(DATA_FOLDER, "external_data")

PREPROCESSING_RESULT_FOLDER = os.path.join(OUT_FOLDER, "event_preprocessing-results")
PREPROCESSING_RESULT_PADIWEB_FOLDER = os.path.join(PREPROCESSING_RESULT_FOLDER, NEWS_SURVEILLANCE_PADIWEB)
PREPROCESSING_RESULT_PROMED_FOLDER = os.path.join(PREPROCESSING_RESULT_FOLDER, NEWS_SURVEILLANCE_PROMED)
PREPROCESSING_RESULT_EMPRESI_FOLDER = os.path.join(PREPROCESSING_RESULT_FOLDER, NEWS_DB_EMPRESS_I)
PREPROCESSING_RESULT_WAHIS_FOLDER = os.path.join(PREPROCESSING_RESULT_FOLDER, NEWS_DB_WAHIS)
PREPROCESSING_RESULT_APHA_FOLDER = os.path.join(PREPROCESSING_RESULT_FOLDER, NEWS_DB_APHA)
PREPROCESSING_RESULT_APHIS_FOLDER = os.path.join(PREPROCESSING_RESULT_FOLDER, NEWS_DB_APHIS)
PREPROCESSING_RESULT_BVBRC_FOLDER = os.path.join(PREPROCESSING_RESULT_FOLDER, NEWS_DB_BVBRC)
PREPROCESSING_RESULT_BVBRC_PRECISE_FOLDER = os.path.join(PREPROCESSING_RESULT_FOLDER, "bvbrc_precise_serovar")
PREPROCESSING_RESULT_BVBRC_IMPRECISE_FOLDER = os.path.join(PREPROCESSING_RESULT_FOLDER, "bvbrc_imprecise_serovar")

GENOME_PREPROCESSING_FOLDER = os.path.join(OUT_FOLDER, "genome_preprocessing-results")
GENOME_PREPROCESSING_BVBRC_FOLDER = os.path.join(GENOME_PREPROCESSING_FOLDER, NEWS_DB_BVBRC)

NORM_RESULT_FOLDER = os.path.join(OUT_FOLDER, "normalization-results")
NORM_RESULT_PADIWEB_FOLDER = os.path.join(NORM_RESULT_FOLDER, NEWS_SURVEILLANCE_PADIWEB)
NORM_RESULT_PROMED_FOLDER = os.path.join(NORM_RESULT_FOLDER, NEWS_SURVEILLANCE_PROMED)
NORM_RESULT_EMPRESI_FOLDER = os.path.join(NORM_RESULT_FOLDER, NEWS_DB_EMPRESS_I)
NORM_RESULT_WAHIS_FOLDER = os.path.join(NORM_RESULT_FOLDER, NEWS_DB_WAHIS)
NORM_RESULT_APHA_FOLDER = os.path.join(NORM_RESULT_FOLDER, NEWS_DB_APHA)
NORM_RESULT_APHIS_FOLDER = os.path.join(NORM_RESULT_FOLDER, NEWS_DB_APHIS)
NORM_RESULT_BVBRC_FOLDER = os.path.join(NORM_RESULT_FOLDER, NEWS_DB_BVBRC)
NORM_RESULT_BVBRC_PRECISE_FOLDER = os.path.join(NORM_RESULT_FOLDER, "bvbrc_precise_serovar")
NORM_RESULT_BVBRC_IMPRECISE_FOLDER = os.path.join(NORM_RESULT_FOLDER, "bvbrc_imprecise_serovar")

DOC_EVENTS_FOLDER = os.path.join(OUT_FOLDER, "doc-events")
DOC_EVENTS_PADIWEB_FOLDER = os.path.join(DOC_EVENTS_FOLDER, NEWS_SURVEILLANCE_PADIWEB)
DOC_EVENTS_PROMED_FOLDER = os.path.join(DOC_EVENTS_FOLDER, NEWS_SURVEILLANCE_PROMED)
DOC_EVENTS_EMPRESI_FOLDER = os.path.join(DOC_EVENTS_FOLDER, NEWS_DB_EMPRESS_I)
DOC_EVENTS_WAHIS_FOLDER = os.path.join(DOC_EVENTS_FOLDER, NEWS_DB_WAHIS)
DOC_EVENTS_APHA_FOLDER = os.path.join(DOC_EVENTS_FOLDER, NEWS_DB_APHA)
DOC_EVENTS_APHIS_FOLDER = os.path.join(DOC_EVENTS_FOLDER, NEWS_DB_APHIS)
DOC_EVENTS_BVBRC_FOLDER = os.path.join(DOC_EVENTS_FOLDER, NEWS_DB_BVBRC)
DOC_EVENTS_BVBRC_PRECISE_FOLDER = os.path.join(DOC_EVENTS_FOLDER, "bvbrc_precise_serovar")
DOC_EVENTS_BVBRC_IMPRECISE_FOLDER = os.path.join(DOC_EVENTS_FOLDER, "bvbrc_imprecise_serovar")

EVENT_CLUSTERING_FOLDER = os.path.join(OUT_FOLDER, "event-clustering")
EVENT_CLUSTERING_PADIWEB_FOLDER = os.path.join(EVENT_CLUSTERING_FOLDER, NEWS_SURVEILLANCE_PADIWEB)
EVENT_CLUSTERING_PROMED_FOLDER = os.path.join(EVENT_CLUSTERING_FOLDER, NEWS_SURVEILLANCE_PROMED)
EVENT_CLUSTERING_EMPRESI_FOLDER = os.path.join(EVENT_CLUSTERING_FOLDER, NEWS_DB_EMPRESS_I)
EVENT_CLUSTERING_WAHIS_FOLDER = os.path.join(EVENT_CLUSTERING_FOLDER, NEWS_DB_WAHIS)
EVENT_CLUSTERING_APHA_FOLDER = os.path.join(EVENT_CLUSTERING_FOLDER, NEWS_DB_APHA)
EVENT_CLUSTERING_APHIS_FOLDER = os.path.join(EVENT_CLUSTERING_FOLDER, NEWS_DB_APHIS)
EVENT_CLUSTERING_BVBRC_FOLDER = os.path.join(EVENT_CLUSTERING_FOLDER, NEWS_DB_BVBRC)
EVENT_CLUSTERING_BVBRC_PRECISE_FOLDER = os.path.join(EVENT_CLUSTERING_FOLDER, "bvbrc_precise_serovar")
EVENT_CLUSTERING_BVBRC_IMPRECISE_FOLDER = os.path.join(EVENT_CLUSTERING_FOLDER, "bvbrc_imprecise_serovar")

CORPUS_EVENTS_FOLDER = os.path.join(OUT_FOLDER, "corpus-events")
CORPUS_EVENTS_PADIWEB_FOLDER = os.path.join(CORPUS_EVENTS_FOLDER, NEWS_SURVEILLANCE_PADIWEB)
CORPUS_EVENTS_PROMED_FOLDER = os.path.join(CORPUS_EVENTS_FOLDER, NEWS_SURVEILLANCE_PROMED)
CORPUS_EVENTS_EMPRESI_FOLDER = os.path.join(CORPUS_EVENTS_FOLDER, NEWS_DB_EMPRESS_I)
CORPUS_EVENTS_WAHIS_FOLDER = os.path.join(CORPUS_EVENTS_FOLDER, NEWS_DB_WAHIS)
CORPUS_EVENTS_APHA_FOLDER = os.path.join(CORPUS_EVENTS_FOLDER, NEWS_DB_APHA)
CORPUS_EVENTS_APHIS_FOLDER = os.path.join(CORPUS_EVENTS_FOLDER, NEWS_DB_APHIS)
CORPUS_EVENTS_BVBRC_FOLDER = os.path.join(CORPUS_EVENTS_FOLDER, NEWS_DB_BVBRC)
CORPUS_EVENTS_BVBRC_PRECISE_FOLDER = os.path.join(CORPUS_EVENTS_FOLDER, "bvbrc_precise_serovar")
CORPUS_EVENTS_BVBRC_IMPRECISE_FOLDER = os.path.join(CORPUS_EVENTS_FOLDER, "bvbrc_imprecise_serovar")


# ====================================================
# FILE NAMES
# ====================================================
PADIWEB_ARTICLES_CSV_FILENAME = "articlesweb"
PADIWEB_ARTICLES_INIT_CSV_FILENAME = "articlesweb_init"
PADIWEB_CLASSIF_LABELS_CSV_FILENAME = "classification_label"
PADIWEB_INFO_EXTRAC_CSV_FILENAME = "extracted_information"
PADIWEB_INFO_EXTRAC_INIT_CSV_FILENAME = "extracted_information_init"
PADIWEB_KEYWORD_CSV_FILENAME = "keyword"
PADIWEB_KEYW_ALIGN_CSV_FILENAME = "keyword_alignment"
PADIWEB_SENTENCES_CSV_FILENAME = "sentences_with_labels"
PADIWEB_EXT_SENTENCES_CSV_FILENAME = "extended_sentences_with_labels"
PADIWEB_SENTENCES_INIT_CSV_FILENAME = "sentences_with_labels_init"
PADIWEB_DISEASE_KEYWORDS_FILENAME = "disease_keyword"
PADIWEB_AGG_SENTENCES_CSV_FILENAME = "aggregated_sentences_with_labels"
PADIWEB_RELEVANT_SENTENCES_CSV_FILENAME = "relevant_sentences_with_labels"
PADIWEB_EXT_INFO_EXTR_CSV_FILENAME = "extended_extracted_information"
PADIWEB_INFO_EXTR_SPATIAL_ENTITIES_CSV_FILENAME = "extr_spatial_entities_info"
PADIWEB_SELECTED_SENTENCE_IDS_FILENAME = "selected_sentence-ids"
PADIWEB_SENTENCE_IDENTIFICATION_GOUND_TRUTH_FILENAME = "sentence-identification-ground-truth"
GEONAMES_HIERARCHY_INFO_FILENAME = "geonames_hierarchy"


# ====================================================
# FILE FORMAT
# ====================================================
FILE_FORMAT_CSV = "csv"
FILE_FORMAT_TXT = "txt"
FILE_FORMAT_PNG = "png"
FILE_FORMAT_PDF = "pdf"


# ====================================================
# STRATEGIES (e.g. event retrieval, event clustering etc.)
# ====================================================
EVENT_RETRIEVAL_STRATEGY_RELEVANT_SENTENCE = "relevant_sentence"
EVENT_RETRIEVAL_STRATEGY_STRUCTURED_DATA = "structured_data"
EVENT_RETRIEVAL_STRATEGY_BERT = "bert"


# ====================================================
# COLUMN NAMES
# ====================================================
# CSV file-related variables
COL_ID = "id"
COL_ARTICLE_ID = "article_id"
COL_ARTICLE_ID_RENAMED = "id_articleweb"
COL_SUMMARY = "summary"
COL_LANG = "lang"
COL_CLASSIF_LABEL_ID = "classificationlabel_id"
COL_ARTICLE_CLASSIF_LABEL_ID = "a_classificationlabel_id"
COL_SENTENCE_CLASSIF_LABEL_ID = "s_classificationlabel_id"
COL_START = "start"
COL_END = "end"
COL_TEXT = "text"
COL_SENT_TEXT = "sentence_text"
COl_POS = "position"
COl_TOKEN_INDEX = "token_index"
COl_LENGTH = "length"
COL_TYPE = "type"
COL_LABEL = "label"
COL_VALUE = "value"
COL_TITLE = "title"
COL_SENTENCES = "sentences"
COL_URL = "url"
COL_POST_ID = "post_id"
COL_SOURCE = "source"
COL_DESCR = "description"
COL_NAME = "name"
COL_PUBLISHED_TIME = "published_at"
COL_PROCESSED_TEXT = "processed_text"
COL_KEYW_ID = "keyword_id"
COL_KEYW_TYPE_ID = "keyword_type_id"
COL_ALIGN_KEYW_ID = "aligned_keyword_id"
COL_RSSFEED_ID = "id_rssfeed"
COL_FROM_AUTO_EXTR = "from_automatic_extraction"
COL_SENTENCE_ID = "sentence_id"
COL_LOCAL_SENTENCE_ID = "local_sentence_id"

# Location-related variables
COL_LAT = "lat"
COL_LNG = "lng"
COL_GEONAMES_HIERARCHY_INFO = "geonames_hierarchy"
COL_GEONAMES_ID = "geonames_id"
COL_GEONAMS_JSON = "geonames_json"
COL_FEATURE_CODE = "feature_code"
COL_COUNTRY = "country"
COL_COUNTRY_ID = "id_country"
COL_COUNTRY_FREQ = "country_freq"
COL_NB_COUNTRY_FREQ = "nb_country"
COL_PUB_COUNTRY = "pub_country"
COL_COUNTRY_ALPHA2_CODE = "Alpha-2 code"
COL_COUNTRY_ALPHA3_CODE = "Alpha-3 code"
COL_LOC_DISTRICT = "district"
COL_LOC_CITY = "city"
COL_LOC_REGION = "region"
COL_LOC_COUNTRY = "country"
COL_LOC_CONTINENT = "continent"
COL_LOC_PLACE_NAME = "place_name"
KEY_GEONAMES_COUNTRY = "country"
KEY_GEONAMES_CODE = "code"
KEY_GEONAMES_NAME = "name"
KEY_GEONAMES_STATE = "state"
KEY_GEONAMES_ADDRESS = "address"

# Disease-related variables
COL_DISEASE = "disease"
COL_DISEASE_SUBTYPE = "disease subtype"
COL_DISEASE_ENTRY = "disease_entry"
COL_DISEASE_HIERARCHY = "disease_hierarchy"
COL_DISEASE_ENTRY = "disease_entry"

# Host-related variables
COL_HOST = "host"
COL_HOST_SUBTYPE = "host subtype"
COL_HOST_ENTRY = "host_entry"
COL_SYMPTOM = "symptom"
COL_SYMPTOM_SUBTYPE = "symptom subtype"





# ====================================================
# PADI-web - classification labels
# ====================================================
CLASSIF_LABEL_CURR_EVENT_ID = 11 # 'current event'
CLASSIF_LABEL_DESCR_EPID_ID = 12 # "'Descriptive epidemiology'"
CLASSIF_LABEL_GENERAL_EPID_ID = 16 # "'General epidemiology'"
CLASSIF_LABEL_GOUTBREAK_DECL1_ID = 6 # "'Outbreak declaration'" >> it is defined in the file "classification_classificationlabel.csv" but not used
# CLASSIF_LABEL_OUTBREAK_DECL2_ID = 24 # "'Outbreak declaration'" >> it is defined in the file "classification_classificationlabel.csv" but not used
CLASSIF_LABEL_TRANS_PATHWAY_ID = 22 # "'Transmission pathway'"


# ====================================================
# GEONAMES & API
# ====================================================
MAX_NB_LOCS_PER_GEONAMES_REQUEST = 100 # by default from geonames
MAX_NB_LOCS_PER_GEOCODING_REQUEST = 50

GEONAMES_API_USERNAME1 = user_consts.GEONAMES_API_USERNAME1 # for geonames api
GEONAMES_API_USERNAME2 = user_consts.GEONAMES_API_USERNAME2 # for geonames api
GEONAMES_API_USERNAME3 = user_consts.GEONAMES_API_USERNAME3 # for geonames api
GEONAMES_API_USERNAME4 = user_consts.GEONAMES_API_USERNAME4 # for geonames api
GEONAMES_API_USERNAME5 = user_consts.GEONAMES_API_USERNAME5 # for geonames api
GEONAMES_API_USERNAME6 = user_consts.GEONAMES_API_USERNAME6 # for geonames api
GEONAMES_API_USERNAME7 = user_consts.GEONAMES_API_USERNAME7 # for geonames api
GEONAMES_API_USERNAME8 = user_consts.GEONAMES_API_USERNAME8 # for geonames api


# ====================================================
# Continents
# ====================================================
CONTINENTS_CODE = ["AF", "NA", "OC", "AN", "AS", "EU", "SA"]
CONTINENTS_NAME = ["Africa", "North America", "Oceania", "Antarctica", "Asia", "Europe", "South America"]



# ====================================================
# PADI-web - BAN LIST FOR SPATIAL ENTITY EXTRACTION/DESAMBIGUATION
# ====================================================
BAN_KEYWORDS_LIST_FOR_SPATIAL_ENTITIES = [
  "Zoo",
  "Bird Sanctuary",
  "Waterworks",
  "Country Park",
  "Biological Park",
  "National Park",
  "Park",
  "Wildlife Sanctuary",
  "rgn",
  "Canal",
  "Marina",
  "Reserve",
]

# source: https://www.dictionary.com/e/dr-vs-phd-vs-md/
BAN_LIST_FOR_SPATIAL_ENTITY_ABBREVIATIONS = [
  "M.D.",
  "Ph.D.", 
  "Dr."
]

BAN_LIST_FOR_SPATIAL_ENTITY_EXTRACTION = [
  "earth", 
  "h5", 
  "states", 
  "zone", 
  "prefecture", 
  "forest", 
  "mars", 
  "vet", 
  "covid", 
  "far east", 
  "county", 
  #"will", >> EZN41UUU8H
  "valley", 
  "border", 
  "sürmland",
  "ai", # avian influenza
  "wnv", # west nile virus
  "wnf", # west nile fever
  "nile", 
  "nile virus", 
  "west nile virus", 
  "west nile", 
  "west nile fever", 
  "nile fever", 
  "blue nile", 
  "swan", 
  "bird", 
  "america",
  "ok",
  "virus",
  "down",
  "up",
  "left",
  "right",
  "centre",
  "center",
  "middle",
  "uganda", # West Nile Virus is named after the West-Nile district of Uganda, where it was first isolated.
  "mosquito",
  "mosquitos",
  "park",
  "more",
  "less",
  "city",
  "not",
  "drive",
  "bay",
  "county",
  "trail",
  "trails",
  "some",
  "still",
  "van", # >> it is used as a proper noun, not the city of Van in Turkey
  "zika",
  "his",
  "this",
  "street",
  "district",
  "get",
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "a.m",
  "a.m."
  "p.m",
  "p.m.",
  "central district",
  "the",
  "dengue",
  "ill",
  "p.s",
  "p.s.",
  "low",
  "high",
  "sun",
  "rain",
  "snow",
  "person",
  "san",
  "old",
  "new",
  "lake",
  "rich",
  "early",
  "late",
  "river",
  "click",
  "hill",
  "plan",
  "spring",
  "springs",
  "town",
  "sle", # I think, it is a disease abbreviation >> see article Z4JJD8RFHJ
  "nbc", # >> the name of a media source
  "cbs", # >> the name of a media source
  "las", # >> las vegas
  "ok",
  "true",
  "false",
  "eastern equine encephalitis",
  "EEE",
  "covid-19",
  "covid19",
  "covid",
  "corona",
  "coronavirus",
  "health",
  "january",
  "february",
  "march",
  "april",
  "may",
  "june",
  "july",
  "august",
  "september",
  "october",
  "november",
  "december",
  "2021",
  "2022",
  "2020",
  "2019",
  "state",
  "will" # altough there is a county called "Will county", it is possible to get false examples >> 0VDSMO3KPZ
]
BAN_LIST_FOR_SPATIAL_ENTITY_EXTRACTION = \
  BAN_LIST_FOR_SPATIAL_ENTITY_EXTRACTION + [t.lower() for t in BAN_LIST_FOR_SPATIAL_ENTITY_ABBREVIATIONS]

# ====================================================
# DISEASE RELATED VARIABLES
# ====================================================
#DISEASE_AVIAN_INFLUENZA = "AI"
#DISEASE_WEST_NILE_VIRUS = "WNV"

DISEASE_HOST_RELATION = {
  # DISEASE_AVIAN_INFLUENZA: ["bird", "avian", "bird-unknown", "wild bird", "domestic bird", "human", "equine", "mammal", "mammal-unknown", "canidae", "arctoidea", "camelidae", "equidae"],
  user_consts.DISEASE_AVIAN_INFLUENZA: ["equine", "mammal", "mammal-unknown", "canidae", "arctoidea", "camelidae", "equidae"],
  user_consts.DISEASE_WEST_NILE_VIRUS: ["bird", "avian", "bird-unknown", "wild bird", "domestic bird", "human", "mosquito", "equine", "mammal"]
}



# ====================================================
# PADI-web - BAN-RELATED KEYWORDS
# ====================================================
BAN_KEYWORDS_LIST = []
BAN_KEYWORDS_LIST.append("banned")
BAN_KEYWORDS_LIST.append("ban on")
BAN_KEYWORDS_LIST.append("banning of")
BAN_KEYWORDS_LIST.append("no threat of")
BAN_KEYWORDS_LIST.append("temporary ban on")
BAN_KEYWORDS_LIST.append("suspend the import")
BAN_KEYWORDS_LIST.append("no cases of")
BAN_KEYWORDS_LIST.append(" bans ")
BAN_KEYWORDS_LIST.append("freedom from avian")
BAN_KEYWORDS_LIST.append("freedom from bird flu")
BAN_KEYWORDS_LIST.append("Prevention of avian influenza") # a bit risky
BAN_KEYWORDS_LIST.append("free of h1n")
BAN_KEYWORDS_LIST.append("free of h2n")
BAN_KEYWORDS_LIST.append("free of h3n")
BAN_KEYWORDS_LIST.append("free of h4n")
BAN_KEYWORDS_LIST.append("free of h5n")
BAN_KEYWORDS_LIST.append("free of h6n")
BAN_KEYWORDS_LIST.append("free of h7n")
BAN_KEYWORDS_LIST.append("free of h8n")
BAN_KEYWORDS_LIST.append("free of h9n")
BAN_KEYWORDS_LIST.append("free of h10n")
BAN_KEYWORDS_LIST.append("virus-free")





# ====================================================
# DISEASE KEYWORDS
# ====================================================
# remark >> HPAI: H5N1, H7N9, H5N6, H5N8
DISEASE_KEYWORDS_DICT = {user_consts.DISEASE_AVIAN_INFLUENZA: {}, user_consts.DISEASE_WEST_NILE_VIRUS: {}}
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_AVIAN_INFLUENZA]["hpai"] = "Avian influenza"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_AVIAN_INFLUENZA]["lpai"] = "Avian influenza"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_AVIAN_INFLUENZA]["avian flu"] = "Avian influenza"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_AVIAN_INFLUENZA]["bird flu"] = "Avian influenza"
# DISEASE_KEYWORDS_DICT["bird flu outbreak"] = "Avian influenza"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_AVIAN_INFLUENZA]["avian influenza"] = "Avian influenza"
#DISEASE_KEYWORDS_DICT["HPAI"] = "Avian influenza" # TODO: DELETE IT
#DISEASE_KEYWORDS_DICT["LPAI"] = "Avian influenza" # TODO: DELETE IT
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_AVIAN_INFLUENZA]["avian"] = "Avian influenza"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_AVIAN_INFLUENZA]["aviaire"] = "Avian influenza"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_AVIAN_INFLUENZA]["грипп"] = "Avian influenza"
#
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_WEST_NILE_VIRUS]["mosquito-born"] = "West Nile Virus"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_WEST_NILE_VIRUS]["mosquito born"] = "West Nile Virus"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_WEST_NILE_VIRUS]["wnv"] = "West Nile Virus"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_WEST_NILE_VIRUS]["west nile"] = "West Nile Virus"
#
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_WEST_NILE_VIRUS]["nilo occidental"] = "West Nile Virus"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_WEST_NILE_VIRUS]["nilo ocidental"] = "West Nile Virus"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_WEST_NILE_VIRUS]["vno"] = "West Nile Virus"
DISEASE_KEYWORDS_DICT[user_consts.DISEASE_WEST_NILE_VIRUS]["западного нила"] = "West Nile Virus"


# ====================================================
# DISEASE TAXONOMY
# ====================================================
HOST_KEYWORDS_HIERARCHY_DICT = {}
HOST_KEYWORDS_HIERARCHY_DICT[user_consts.HOST_HUMAN] = [
  #{"text": "man", "level": 1, "hierarchy": ("human", "male")},
  {"text": "hombre", "level": 2, "hierarchy": ("human", "male")}, # (ES)
  {"text": "mujer", "level": 2, "hierarchy": ("human", "female")}, # (ES)
  {"text": "señor", "level": 2, "hierarchy": ("human", "male")}, # (ES)
  {"text": "niña", "level": 2, "hierarchy": ("human", "female")}, # >> child (ES)
  {"text": "niño", "level": 2, "hierarchy": ("human", "male")}, # >> child (ES)
  {"text": "adulto", "level": 2, "hierarchy": ("human", "male")},
  {"text": "adulta", "level": 2, "hierarchy": ("human", "female")},
  {"text": "uomo", "level": 2, "hierarchy": ("human", "male")},
  {"text": "donna", "level": 2, "hierarchy": ("human", "female")},
  {"text": "bambino", "level": 2, "hierarchy": ("human", "male")},
  {"text": "bambina", "level": 2, "hierarchy": ("human", "female")},
  {"text": "homme", "level": 2, "hierarchy": ("human", "male")},
  {"text": "femme", "level": 2, "hierarchy": ("human", "female")},
  {"text": "человек", "level": 2, "hierarchy": ("human", "male")}, # man (RU)
  {"text": "женщина", "level": 2, "hierarchy": ("human", "female")}, # woman (RU)
  {"text": "boy", "level": 2, "hierarchy": ("human", "male")},
  {"text": "old man", "level": 2, "hierarchy": ("human", "male")},
  {"text": "man living", "level": 2, "hierarchy": ("human", "male")},
  {"text": "old male", "level": 2, "hierarchy": ("human", "male")},
  {"text": "adult male", "level": 2, "hierarchy": ("human", "male")},
  {"text": "girl", "level": 2, "hierarchy": ("human", "female")},
  {"text": "woman", "level": 2, "hierarchy": ("human", "female")},
  {"text": "old female", "level": 2, "hierarchy": ("human", "female")},
  {"text": "adult female", "level": 2, "hierarchy": ("human", "female")},
  {"text": "old woman", "level": 2, "hierarchy": ("human", "female")},
  #{"text": "hospital", "level": 1, "hierarchy": ("human", "gender-unknown")},
  #{"text": "people", "level": 1, "hierarchy": ("human", "gender-unknown")},
  #{"text": "death", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "humain", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "humano", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "malato", "level": 1, "hierarchy": ("human", "gender-unknown")}, # (ES) or (IT)
  {"text": "paciente", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "enfant", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "adulte", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "patient", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "ребенок", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "пациент", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "взрослый", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human case", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human h1n", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human h2n", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human h3n", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human h4n", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human h5n", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human h6n", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human h7n", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human h8n", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human h9n", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human h10n", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "case of human", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "case of a human", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "bird flu in human", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "avian influenza in human", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "new human infect", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human infection", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human transmission", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human west nile virus infection", "level": 1, "hierarchy": ("human", "gender-unknown")},
  # {"text": "resident", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "child", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "patient", "level": 1, "hierarchy": ("human", "gender-unknown")},
  # {"text": "victim", "level": 1, "hierarchy": ("human", "gender-unknown")},
  # {"text": "hospital", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "human", "level": 1, "hierarchy": ("human", "gender-unknown")},
  # {"text": "death", "level": 1, "hierarchy": ("human", "gender-unknown")}
  {"text": "people", "level": 1, "hierarchy": ("human", "gender-unknown")},
  # {"text": "year-old", "level": 1, "hierarchy": ("human", "gender-unknown")},
  # {"text": "year old", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "covid-19", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "covid", "level": 1, "hierarchy": ("human", "gender-unknown")},
  {"text": "omicron", "level": 1, "hierarchy": ("human", "gender-unknown")}
]
    


# ==============================================================================
HOST_KEYWORDS_HIERARCHY_DICT[user_consts.HOST_MAMMAL] = []
item = {"id": 40674, "text": "mammal", "level": 1, "hierarchy": ["mammal"]}
HOST_KEYWORDS_HIERARCHY_DICT[user_consts.HOST_MAMMAL].append(item)
    
HOST_KEYWORDS_HIERARCHY_DICT[user_consts.HOST_AVIAN] = []
item = {"id": 8782, "text": "avian", "level": 1, "hierarchy": ["mammal"]}
HOST_KEYWORDS_HIERARCHY_DICT[user_consts.HOST_AVIAN].append(item)
  
HOST_KEYWORDS_HIERARCHY_DICT["other"] = []
#item = {"id": NCBIid, "text": host_name.lower(), "level": len(lineage_as_list), "hierarchy": lineage_as_list}
#HOST_KEYWORDS_HIERARCHY_DICT[host_type].append(item)

NCBI_lineage_filepath = os.path.join(DATA_FOLDER, "taxonomy", "host", "lineage.csv")
NCBI_tax_names_filepath = os.path.join(DATA_FOLDER, "taxonomy", "host", "NCBItaxNames.csv")

df_ncbi_lineage = pd.read_csv(NCBI_lineage_filepath, sep="\t", keep_default_na=False)
NCBIid2lineage = dict(zip(df_ncbi_lineage["tax_id"], df_ncbi_lineage["lineage"]))

df_ncbi_init = pd.read_csv(NCBI_tax_names_filepath, sep=",", keep_default_na=False)
df_ncbi_init["common_name"] = df_ncbi_init["common_name"].str.lower()
df_ncbi_init["common_name"] = df_ncbi_init["common_name"].str.replace('(', '\(')
df_ncbi_init["common_name"] = df_ncbi_init["common_name"].str.replace(')', '\)')
df_ncbi_init["sciname"] = df_ncbi_init["sciname"].str.replace('(', '\(')
df_ncbi_init["sciname"] = df_ncbi_init["sciname"].str.replace(')', '\)')
df_ncbi_init["sciname"] = df_ncbi_init["sciname"].str.lower()
df_ncbi_init = df_ncbi_init[~df_ncbi_init["sciname"].str.contains("unclassified")]

df_ncbi_aux = df_ncbi_init[df_ncbi_init["common_name"].str.contains(",")]
#df_ncbi = df_ncbi_init[~df_ncbi_init["common_name"].str.contains(",")]
df_ncbi = df_ncbi_init
df_ncbi["common_name"] = df_ncbi["common_name"].str.replace("sealion","sea lion")
NCBIid2commonName = dict(zip(df_ncbi["tax_id"], df_ncbi["common_name"]))

hostName2NCBIid = dict(zip(df_ncbi["common_name"], df_ncbi["tax_id"]))
name2id_2 = dict(zip(df_ncbi["synonym_name"], df_ncbi["tax_id"]))
name2id_3 = dict(zip(df_ncbi["sciname"], df_ncbi["tax_id"]))
name2id_4 = dict(zip(df_ncbi_aux["sciname"], df_ncbi_aux["tax_id"]))
for key in name2id_2.keys():
  if key not in hostName2NCBIid:
    hostName2NCBIid[key] = name2id_2[key]
for key in name2id_3.keys():
  if key not in hostName2NCBIid:
    hostName2NCBIid[key] = name2id_3[key]
for key in name2id_4.keys():
  if key not in hostName2NCBIid:
    hostName2NCBIid[key] = name2id_4[key]
    
for host_name in hostName2NCBIid.keys():
  NCBIid = hostName2NCBIid[host_name]
  lineage_init = NCBIid2lineage[NCBIid].lower()
  lineage_as_list = []
  if host_name != "" and "tetrapoda;" in lineage_init:
    lineage = lineage_init.split("tetrapoda;")[1]
    # adjustment of hierarchy for 2 species: mammals and birds
    host_type = "other"
    if "homininae" not in lineage: # we do not want humans
      if "mammalia;" in lineage:
        host_type = user_consts.HOST_MAMMAL
        lineage = lineage.split("mammalia;")[1]
        lineage_as_list = ["mammal"] + lineage.split(";")
      elif "aves;" in lineage:
        host_type = user_consts.HOST_AVIAN
        lineage = lineage.split("aves;")[1]
        lineage_as_list = ["avian"] + lineage.split(";")

    item = {"id": NCBIid, "text": host_name.lower(), "level": len(lineage_as_list), "hierarchy": lineage_as_list}
    HOST_KEYWORDS_HIERARCHY_DICT[host_type].append(item)
# ==============================================================================

  

# ====================================================
# TEXT MINING
# ====================================================
STOPWORDS_FILEPATH = os.path.join(DATA_FOLDER, "stopwords.txt")
# source: https://gist.github.com/sebleier/554280
#import requests
#stopwords_list = requests.get(consts.STOPWORDS_FILEPATH).content

with open(STOPWORDS_FILEPATH, "r") as myfile:
  stopwords_content = myfile.read()
  stopwords_ext = set(stopwords_content.splitlines())
    
    

