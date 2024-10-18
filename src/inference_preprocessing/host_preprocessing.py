
import pandas as pd
import src.consts as consts
import numpy as np
import os



# ======================================
# We add if the infected animal from the disease is wild or domestic
# ======================================
def add_wild_domestic_host_info_into_events(df_events, wild_dom_info_filepath):
    cols = ["Event.ID", "Animal.type"]
    df_wild_dom_info = pd.read_csv(wild_dom_info_filepath, usecols=cols, sep=",", keep_default_na=False)
    #df_wild_dom_info["Event.ID"] = df_wild_dom_info["Event.ID"].apply(int)
    eventId2wildness = dict(zip(df_wild_dom_info["Event.ID"], df_wild_dom_info["Animal.type"]))
    #print(df_wild_dom_info["Event.ID"])
    #print(df_events[consts.COL_ID])

    unique_vals = df_wild_dom_info["Animal.type"]
    #print(np.unique(unique_vals))

    df_events["animal_type"] = df_events[consts.COL_ARTICLE_ID].apply(lambda id: eventId2wildness[id] if id in eventId2wildness else "-1")
    df_events = df_events[df_events["animal_type"] != "-1"]
    return df_events




# ======================================
# MAIN FUNCTION
# ======================================
def perform_host_preprocessing(df_events, external_data_folder):
    wild_dom_info_filepath = os.path.join(external_data_folder, "epidemiology-raw-data_202209170105.csv")
    df_events = add_wild_domestic_host_info_into_events(df_events, wild_dom_info_filepath)
    df_events.reset_index(drop=True, inplace=True)
    return df_events