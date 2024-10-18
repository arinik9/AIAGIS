'''
Created on Jan 12, 2022

@author: nejat
'''


import os
import csv
import numpy as np
import pandas as pd

from src.util_event import read_df_events, read_events_from_df
from src.event_matching.event_matching_strategy import EventMatchingStrategy, EventMatchingStrategyPossiblyDuplicate



class EventMatching():

  def __init__(self, event_matching_strategy:EventMatchingStrategy, split_by_country):
    self.event_matching_strategy = event_matching_strategy
    self.split_by_country = split_by_country
    
        

  def perform_event_matching(self, platform1_desc, platform1_events_filepath,\
                                        platform2_desc, platform2_events_filepath, output_dirpath):
    try:
      if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)
      output_aux_dirpath = os.path.join(output_dirpath, "auxiliary")
      if not os.path.exists(output_aux_dirpath):
        os.makedirs(output_aux_dirpath)
    except OSError as err:
      print(err)


    df_events_platform1 = read_df_events(platform1_events_filepath)
    df_events_platform2 = read_df_events(platform2_events_filepath)

    country_code_list = df_events_platform1["loc_country_code"].unique()
    unique_country_code_list = np.sort(country_code_list)
    year_list = df_events_platform1["year"].unique()
    unique_year_list = np.sort(year_list)
    print(unique_country_code_list)
    print(unique_year_list)

    if self.split_by_country == False:
        unique_country_code_list = [None]

    df_event_matching_list = []

    # # unique_year_list = [x for x in unique_year_list if x > 2011]
    # for year in unique_year_list:
    #     print("year:", year)
    #     df_events_platform1_by_year = df_events_platform1[(df_events_platform1["year"] == year)]
    #     print(df_events_platform1_by_year.shape)
    #     df_events_platform2_by_year_around = df_events_platform2[
    #         (df_events_platform2["year"].isin([year - 1, year, year + 1]))]
    #     print(df_events_platform2_by_year_around.shape)
    #
    #     events_platform1 = read_events_from_df(df_events_platform1_by_year)
    #     events_platform2 = read_events_from_df(df_events_platform2_by_year_around)

    #unique_year_list = [2013]
    unique_year_list = [x for x in unique_year_list if x > 2011]
    for year in unique_year_list:
        for country_code in unique_country_code_list:
            print("year:",year,", country:",country_code)
            df_events_platform1_by_country_and_year = df_events_platform1[df_events_platform1["year"] == year]
            if country_code is not None:
                df_events_platform1_by_country_and_year = df_events_platform1[(df_events_platform1["year"] == year) & (df_events_platform1["loc_country_code"] == country_code)]
            print(df_events_platform1_by_country_and_year.shape)
            df_events_platform2_by_country_and_year_around = df_events_platform2[df_events_platform2["year"].isin([year-1,year,year+1])]
            if country_code is not None:
                df_events_platform2_by_country_and_year_around = df_events_platform2[(df_events_platform2["year"].isin([year-1,year,year+1])) & (df_events_platform2["loc_country_code"] == country_code)]
            print(df_events_platform2_by_country_and_year_around.shape)
            events_platform1 = read_events_from_df(df_events_platform1_by_country_and_year)
            events_platform2 = read_events_from_df(df_events_platform2_by_country_and_year_around)

            if len(events_platform1)>0 and len(events_platform2)>0:
                result_filename = platform1_desc + "_" + platform2_desc + "_event_matching-year=" + str(year) + ".csv"
                if country_code is not None:
                    result_filename = platform1_desc + "_" + platform2_desc + "_event_matching-year=" + str(year) + "_country=" + country_code + ".csv"
                result_filepath = os.path.join(output_aux_dirpath, result_filename)

                if not os.path.exists(result_filepath):
                    # ===================================================================================
                    # PERFORM EVENT MATCHING and store the result in output folder
                    # ===================================================================================
                    out_sim_matrix_filepath = os.path.join(output_aux_dirpath, result_filename.replace("event_matching", "sim_matrix"))
                    df_event_matching = self.event_matching_strategy.perform_event_matching(events_platform1, events_platform2, out_sim_matrix_filepath)
                    df_event_matching.rename(columns={'event1_id': platform1_desc+'_id', \
                                                 'event1_loc_info': platform1_desc+'_loc_info', \
                                                 'event1_loc_hierarchy': platform1_desc+'_loc_hierarchy', \
                                                 'event1_date': platform1_desc+'_date',\
                                                 'event1_disease': platform1_desc+'_disease',\
                                                 'event1_host': platform1_desc+'_host',\
                                                 'event2_id': platform2_desc+'_id', \
                                                 'event2_loc_info': platform2_desc+'_loc_info', \
                                                 'event2_loc_hierarchy': platform2_desc+'_loc_hierarchy', \
                                                 'event2_date': platform2_desc+'_date',\
                                                 'event2_disease': platform2_desc+'_disease',\
                                                 'event2_host': platform2_desc+'_host',\
                                                  }, inplace=True)


                    # write into file
                    df_event_matching.to_csv(result_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
                    df_event_matching_list.append(df_event_matching)
                else:
                    df_event_matching = pd.read_csv(result_filepath, sep=";", keep_default_na=False)
                    print(df_event_matching.shape)
                    df_event_matching_list.append(df_event_matching)

    # write all the results into a single file
    df_all = pd.concat(df_event_matching_list)
    result_filename = platform1_desc + "_" + platform2_desc + "_event_matching.csv"
    result_filepath = os.path.join(output_dirpath, result_filename)
    df_all.to_csv(result_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
