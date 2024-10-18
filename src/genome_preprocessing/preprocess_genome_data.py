'''
Created on Jul 18, 2023

@author: nejat
'''

import pandas as pd
import csv
import re
from iso3166 import countries
import src.consts as consts
import os
import numpy as np
import json
from collections import Counter
from unidecode import unidecode

from geopy.geocoders import Nominatim, GeoNames, ArcGIS
import random

import itertools

from src.geocoding.geocode_aux import geocode_batch_with_arcgis_aux, geocode_batch_with_nominatim_aux, geocode_batch_with_geonames_aux





###################################################################################
# It gets a country name and returns an alpha2 code of the corresponding country.
# Since, the function "countries.get(country_name)" is a simple dictionary, it does not
# handle some different country names. That is why we add some custom if/else blocks
# before this function.
#
###################################################################################
def retrieve_country_alpha2_code_from_country_name(country_name):
  country_code = "-1"
  if country_name.lower() == "south korea" or country_name.lower() == "korea":
    country_code = "KR"
  elif country_name.lower() == "russia":
    country_code = "RU"
  elif country_name.lower() == "united kingdom":
    country_code = "GB"
  elif country_name.lower() == "usa" or country_name.lower() == "united states":
    country_code = "US"
  elif country_name.lower() == "reunion" or country_name.lower() == "réunion":
    country_code = "RE"
  elif country_name.lower() == "turkey":
    country_code = "TR"
  elif country_name.lower() == "iran":
    country_code = "IR"
  elif country_name.lower() == "laos":
    country_code = "LA"
  elif country_name.lower() == "czech republic":
    country_code = "CZ"
  elif country_name.lower() == "democratic republic of the congo":
    country_code = "CD"
  elif country_name.lower() == "cote d'ivoire" or country_name.lower() == "ivory coast":
    country_code = "CI"
  elif country_name.lower() == "north korea":
    country_code = "KP"      
  else:
    country_code = countries.get(country_name).alpha2
  return country_code


###################################################################################
#
#
# STEP 1: Reading the input files and some simple event_preprocessing
# STEP 2: Reducing the input data
# STEP 3: Retrieve country alpha2 codes from country names
# STEP 4: Spatial entity disambiguation/event_preprocessing
# STEP 5: host entity event_preprocessing
#
###################################################################################
def preprocess_genome_data(in_genome_folder, out_genome_preprocessing_folder, genome_filepath, genome_seq_filepath, output_filepath, force=False):
  
  # =====================================================================
  # STEP 1: Reading the input files and some simple event_preprocessing
  #         (e.g. renaming the column names)
  # =====================================================================
  
  df_raw_genome = pd.read_csv(genome_filepath, \
        usecols=["Genome ID", "Strain", "Serovar", "Genome Status", "GenBank Accessions", "Segment", "Collection Date", \
         "Collection Year", "Isolation Country", "Geographic Location", "Host Name", "Host Common Name", "Host Group",
                 "Publication", "BioProject Accession", "Sequencing Platform", "Assembly Method", "Isolation Source"], sep=";", keep_default_na=False, dtype=str)
  #df_raw_genome["Genome ID"] = df_raw_genome["Genome ID"].astype(str)
  df_raw_genome["Serovar"] = df_raw_genome["Serovar"].apply(lambda x: x.lower())
  df_raw_genome["Serovar"] = df_raw_genome["Serovar"].apply(lambda x: x.replace("nx", "").strip() if "nx" in x else x)
  df_raw_genome["Serovar"] = df_raw_genome["Serovar"].apply(lambda x: x.replace("hx", "").strip() if "hx" in x else x)
  df_raw_genome.loc[df_raw_genome["Serovar"] == "mixed", "Serovar"] = ""
  df_raw_genome.loc[df_raw_genome["Serovar"] == "unknown", "Serovar"] = ""
  df_raw_genome.loc[df_raw_genome["Serovar"] == "unidentified", "Serovar"] = ""
  df_raw_genome = df_raw_genome.rename(columns={"Strain": "Genome Name"})


  # ====================================================================================================
  # TODO: there can be multiple entries corresponding to the same strain name.
  #       But, the serovar information can be different. See the example below:
  #       Check if there is only one valid serovar name and the segment values are all different
  # Orthomyxoviridae	Alphainfluenzavirus	Influenza A virus	Complete	A/duck/Hunan/1.17_YYGKK93-OC/2017	H6N6				4	H6N6
  # Orthomyxoviridae	Alphainfluenzavirus	Influenza A virus	Complete	A/duck/Hunan/1.17_YYGKK93-OC/2017	H6N6				6	H6N6
  # Orthomyxoviridae	Alphainfluenzavirus	Influenza A virus	Complete	A/duck/Hunan/1.17_YYGKK93-OC/2017	mixed				1	Mixed
  # Orthomyxoviridae	Alphainfluenzavirus	Influenza A virus	Complete	A/duck/Hunan/1.17_YYGKK93-OC/2017	mixed				2	Mixed
  # Orthomyxoviridae	Alphainfluenzavirus	Influenza A virus	Complete	A/duck/Hunan/1.17_YYGKK93-OC/2017	mixed				5	Mixed
  # ====================================================================================================

  df_genome_seq = pd.read_csv(genome_seq_filepath, \
            usecols=["id", "seq"], sep=";", keep_default_na=False, dtype=str)
  
  # Issue: Some genome ids ends with 0, if we do not consider them as string, the last number disappears when considered as integer/flaat.
  #        That is why by adding a prefix, we ensure that it is considered as string  
  df_raw_genome["Genome ID"] = "bvbrc" + df_raw_genome["Genome ID"]
  #df_genome_seq["id"] = "bvbrc" + df_genome_seq["id"]
  
  #print(df_raw_genome.head())
  #print(df_genome_seq.head())
  
  #id2seq = dict(zip(df_genome_seq["id"], df_genome_seq["seq"]))
  #df_raw_genome["seq"] = df_raw_genome["Genome ID"].apply(lambda x: id2seq[x])
  
  # in some data entries, the country information is missing. We simply discard them.
  #print(df_raw_genome[df_raw_genome["Isolation Country"] == ""].shape)
  df_raw_genome = df_raw_genome[df_raw_genome["Isolation Country"] != ""]
  # if "Geographic Location" is empty and "Isolation Country" is not empty, handle this case
  empty_geo_loc_rows_index = df_raw_genome[df_raw_genome["Geographic Location"] == ""].index
  df_raw_genome.loc[empty_geo_loc_rows_index, "Geographic Location"] = df_raw_genome.loc[empty_geo_loc_rows_index, "Isolation Country"]
  df_raw_genome = df_raw_genome[df_raw_genome["Geographic Location"] != ""]

  #print(df_raw_genome.shape[0])
  df_raw_genome2 = df_raw_genome.drop_duplicates(subset=["Genome Name", "Geographic Location", "Serovar"])
  #print(df_raw_genome2.shape[0])

  empty_date_rows_index = df_raw_genome[df_raw_genome["Collection Date"] == ""].index
  df_raw_genome.loc[empty_date_rows_index, "Collection Date"] = df_raw_genome.loc[empty_date_rows_index, "Collection Year"]
  df_raw_genome = df_raw_genome[df_raw_genome["Collection Date"] != ""]

  df_raw_genome = df_raw_genome[df_raw_genome["Segment"] != ""] # this concerns the data before 2011, around 7000 entry removed
  df_raw_genome.loc[df_raw_genome["Segment"] == "PB1", "Segment"] = "2"
  df_raw_genome.loc[df_raw_genome["Segment"] == "PB2", "Segment"] = "1"
  df_raw_genome.loc[df_raw_genome["Segment"] == "PA", "Segment"] = "3"
  df_raw_genome.loc[df_raw_genome["Segment"] == "HA", "Segment"] = "4"
  df_raw_genome.loc[df_raw_genome["Segment"] == "NP", "Segment"] = "5"
  df_raw_genome.loc[df_raw_genome["Segment"] == "NS", "Segment"] = "8"
  df_raw_genome.loc[df_raw_genome["Segment"] == "NA", "Segment"] = "6"
  df_raw_genome.loc[df_raw_genome["Segment"] == "M", "Segment"] = "7"
  #df_raw_genome = df_raw_genome.astype({'Segment': 'int'})


  # =====================================================================
  # STEP 2: Reduce the input data
  #          Each genome id is associated with multiple GenBank Accessions.
  #          We just take the first one in order to have a 1-to-1 maping
  #          in the rest of the analysis
  # =====================================================================
  output_auxiliary_filepath = output_filepath.split(".")[0] + "_aux.csv"
  if not os.path.exists(output_auxiliary_filepath):
    df_raw_genome["Segment2GenomeID"] = df_raw_genome["Genome ID"]+":"+df_raw_genome["Segment"]

    # TODO: group by icin, ""Geographic Location" yerine "Collection Date" daga iyi olur sanki ama
    #         calismadigi yerler de var. >> Exemple: A/blue-winged teal/Minnesota/Sg-00043/2007 ??
    df_raw_genome_grouped = df_raw_genome.groupby(["Genome Name", "Isolation Country", "Geographic Location", "Serovar"]).agg(
                                    {
                                      'GenBank Accessions': lambda d: ", ".join(set(d)),
                                      "Segment2GenomeID": lambda d: ", ".join(set(d)),
                                      'Genome ID': lambda d: ", ".join(set(d)),
                                      'Collection Date' : 'first',
                                      'Host Name' : 'first',
                                      'Host Common Name' : 'first',
                                      'Host Group' : 'first',
                                      'Collection Year' : 'first',
                                      'Isolation Source' : 'first',
                                      'Publication' : 'first',
                                      'Sequencing Platform' : 'first',
                                      'BioProject Accession' : 'first',
                                      'Assembly Method' : 'first'
    }).reset_index()

    df_raw_genome_grouped["Segment2GenomeID"] = df_raw_genome_grouped["Segment2GenomeID"].apply(lambda x:
            json.dumps(dict(zip([int(i.split(":")[1]) for i in x.split(", ")], [i.split(":")[0] for i in x.split(", ")])), ensure_ascii=False))

    # OPTIONAL
    df_raw_genome_grouped.to_csv(output_auxiliary_filepath, index=True, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    df_raw_genome = df_raw_genome_grouped
  df_raw_genome = pd.read_csv(output_auxiliary_filepath, sep=";", keep_default_na=False, dtype=str)
  # ----------------------------------------------------------------


  # =====================================================================
  # STEP 3: Retrieve country alpha2 codes from country names
  #          This info is needed for geocoding
  # =====================================================================
  
  
  df_raw_genome["disease"] = "Influenza - Avian"
  df_raw_genome["ObservationDate"] = df_raw_genome["Collection Date"]
  df_raw_genome["Subregion"] = ""
  df_raw_genome["Region"] = ""
  df_raw_genome["lat"] = ""
  df_raw_genome["lng"] = ""
  
  #df_raw_genome = df_raw_genome.rename(columns={"Genome ID": "id"})
  df_raw_genome = df_raw_genome.rename(columns={"Serovar": "Serotype"})
  df_raw_genome = df_raw_genome.rename(columns={"Collection Date": "ReportDate"})
  
  country_name_list = df_raw_genome["Isolation Country"].to_list()
  country_code_list = []
  for i in range(len(country_name_list)):
    country_name = country_name_list[i]
    country_code = retrieve_country_alpha2_code_from_country_name(country_name)
    country_code_list.append(country_code)
  df_raw_genome["country code"] = country_code_list

  df_raw_genome["Geographic Location"] = df_raw_genome["Geographic Location"].apply(lambda x: x.replace(";", ","))


  # =====================================================================
  # STEP 4: Spatial entity disambiguation
  #         Some values in the column "Geographic Location" are ambiguous (see some examples below).
  #         We use three geocoding tools (ArcGIS, Nominatim, GeoNames) to solve this issue
  #         The goal is not to normalize spatial entities, rather identifying 
  #         which part of the text corresponds to a region name.
  #         Because, our analysis on disease diffusion is at ADM1 level.
  # ===================================================================== 
  
  # example 1: Australia: Victoria, Dan's Reserve, Breamlea
  # example 2: USA: South Bower's Beach, DE
  # example 3: USA: Mispillion Harbor, Sussex county, DE
  # example 4: Japan:Ibaraki, Kasumigaura Lake
  # Czech Republic: Zdar nad Sazavou, Kundratice u Krizanova, Sadecky pond-H. Libochova
  # Czech Republic: Trutnov, Stare Buky
  # Czech Republic: Hodonin, Dvorsky pond
  # USA: Bear Lake County, Idaho
  # China: Dongguan, Guangdong
  # Australia: Victoria, Hospital Swamp, Connawarre
  # USA: New Jersey, Villas Beach; Delaware Bay
  # USA: Ilo; Dunn County, ND
  # USA: Agassiz NWR; Marshall Co., Minnesota
  # India: Dey Para,Nadia,WB
  # Egypt: Sharkia
  # Russia: farmstead Gulyaj-Borisovka, Zernogradskij district, Rostov region

  df_raw_genome_grouped_distinct = df_raw_genome.drop_duplicates(subset=["Geographic Location"])

  #sel_list = ["USA: Madison County, Arkansas"]
  #df_raw_genome_grouped_distinct = df_raw_genome_grouped_distinct[df_raw_genome_grouped_distinct["Geographic Location"].isin(sel_list)]

  country_name_list = df_raw_genome_grouped_distinct["Isolation Country"].tolist()
  country_code_list = df_raw_genome_grouped_distinct["country code"].tolist()
  #df_raw_genome_grouped_distinct["Geographic Location"] = df_raw_genome_grouped_distinct["Geographic Location"].apply(lambda x: x.replace(";", ","))
  unique_spatial_entity_list = df_raw_genome_grouped_distinct["Geographic Location"].tolist()
  print("There are " + str(len(unique_spatial_entity_list)) + " unique spatial entities.")
  #print(unique_spatial_entity_list[0:10])

  # STEP 4.1: Geocoding with 3 different geocoders

  country_alpha3_code_list = df_raw_genome_grouped_distinct["country code"].apply(lambda x: countries.get(x).alpha3).tolist()
  country_alpha2_code_list = df_raw_genome_grouped_distinct["country code"].apply(lambda x: countries.get(x).alpha2).tolist()


  # TODO: test all geocoders for "USA: New Jersey, Reed's Beach, Cape May Co."

  # TODO: arcgis >> the attribute "name" is sometimes empty  
  print("--------- ARCGIS")
  arcgis_results_in_batch_filepath = os.path.join(out_genome_preprocessing_folder, "arcgis_results_in_batch.csv")
  if os.path.exists(arcgis_results_in_batch_filepath):
    df_arcgis_result = pd.read_csv(arcgis_results_in_batch_filepath, sep=";", keep_default_na=False, dtype=str)
    seen_list = df_arcgis_result["text"].tolist()
    df_raw_genome_grouped_distinct_arcgis = df_raw_genome_grouped_distinct[~df_raw_genome_grouped_distinct["Geographic Location"].isin(seen_list)]
    print("nb unseen elements in arcgis:", df_raw_genome_grouped_distinct_arcgis.shape[0])
    #if df_raw_genome_grouped_distinct_arcgis.shape[0]>0: # if there is any unseen element
    country_alpha3_code_list = df_raw_genome_grouped_distinct_arcgis["country code"].apply(lambda x: countries.get(x).alpha3).tolist()
    unique_spatial_entity_list = df_raw_genome_grouped_distinct_arcgis["Geographic Location"].tolist()
    print(unique_spatial_entity_list)
    print(country_alpha3_code_list)
  alternative_spatial_entity_list_of_list = prepare_alternative_spatial_entity_lists_for_spatial_entity_disambiguation(unique_spatial_entity_list)
  geocode_batch_with_arcgis_aux(unique_spatial_entity_list, alternative_spatial_entity_list_of_list, country_alpha3_code_list, arcgis_results_in_batch_filepath)
  df_arcgis_result = pd.read_csv(arcgis_results_in_batch_filepath, sep=";", keep_default_na=False, dtype=str, index_col=0) # usecols=["id", "seq"]

  print("--------- NOMINATIM")
  nominatim_results_in_batch_filepath = os.path.join(out_genome_preprocessing_folder, "nominatim_results_in_batch.csv")
  if os.path.exists(nominatim_results_in_batch_filepath):
    df_nominatim_result = pd.read_csv(nominatim_results_in_batch_filepath, sep=";", keep_default_na=False, dtype=str)
    seen_list = df_nominatim_result["text"].tolist()
    df_raw_genome_grouped_distinct_nominatim = df_raw_genome_grouped_distinct[~df_raw_genome_grouped_distinct["Geographic Location"].isin(seen_list)]
    print("nb unseen elements in nominatim:", df_raw_genome_grouped_distinct_nominatim.shape[0])
    # if df_raw_genome_grouped_distinct_nominatim.shape[0]>0: # if there is any unseen element
    country_alpha2_code_list = df_raw_genome_grouped_distinct_nominatim["country code"].apply(lambda x: countries.get(x).alpha2).tolist()
    unique_spatial_entity_list = df_raw_genome_grouped_distinct_nominatim["Geographic Location"].tolist()
  alternative_spatial_entity_list_of_list = prepare_alternative_spatial_entity_lists_for_spatial_entity_disambiguation(unique_spatial_entity_list)
  geocode_batch_with_nominatim_aux(unique_spatial_entity_list, alternative_spatial_entity_list_of_list, country_alpha2_code_list, nominatim_results_in_batch_filepath)
  df_nominatim_result = pd.read_csv(nominatim_results_in_batch_filepath, sep=";", keep_default_na=False, dtype=str, index_col=0)

  geonames_results_in_batch_filepath = os.path.join(out_genome_preprocessing_folder, "geonames_results_in_batch.csv")
  if os.path.exists(geonames_results_in_batch_filepath):
    df_geonames_result = pd.read_csv(geonames_results_in_batch_filepath, sep=";", keep_default_na=False, dtype=str)
    seen_list = df_geonames_result["text"].tolist()
    df_raw_genome_grouped_distinct_geonames = df_raw_genome_grouped_distinct[~df_raw_genome_grouped_distinct["Geographic Location"].isin(seen_list)]
    print("nb unseen elements in geonames:", df_raw_genome_grouped_distinct_geonames.shape[0])
    # if df_raw_genome_grouped_distinct_nominatim.shape[0]>0: # if there is any unseen element
    country_alpha2_code_list = df_raw_genome_grouped_distinct_geonames["country code"].apply(lambda x: countries.get(x).alpha2).tolist()
    unique_spatial_entity_list = df_raw_genome_grouped_distinct_geonames["Geographic Location"].tolist()
  alternative_spatial_entity_list_of_list = prepare_alternative_spatial_entity_lists_for_spatial_entity_disambiguation(unique_spatial_entity_list)
  geocode_batch_with_geonames_aux(unique_spatial_entity_list, alternative_spatial_entity_list_of_list, country_alpha2_code_list, geonames_results_in_batch_filepath)
  df_geonames_result = pd.read_csv(geonames_results_in_batch_filepath, sep=";", keep_default_na=False, dtype=str, index_col=0)

  print("arcgis: " + str(df_arcgis_result.shape[0]) + " unique spatial entities.")
  print("nominatim: " + str(df_nominatim_result.shape[0]) + " unique spatial entities.")
  print("geonames: " + str(df_geonames_result.shape[0]) + " unique spatial entities.")

  # STEP 4.2: Disambiguation from the results of the geocoders
  # >> we want to get the region name, ADM1 level

  final_geocoding_result_filepath = os.path.join(out_genome_preprocessing_folder, "final_geocoding_result.csv")
  if not os.path.exists(final_geocoding_result_filepath):
    arcgis_first_result_list = []
    nominatim_first_result_list = []
    geonames_first_result_list = []
    country_name_list = []
    country_code_list = []
    region_name_list = []
    city_name_list = []

    unique_spatial_entity_list = df_raw_genome_grouped_distinct["Geographic Location"].tolist()
    for i, spatial_entity in enumerate(unique_spatial_entity_list):
      print(i,"/",len(unique_spatial_entity_list))
      print("spatial entity:", spatial_entity)

      available_results_for_region = []
      available_results_for_city = []
      for i in range(50):
        arcgis_result = json.loads(df_arcgis_result.loc[spatial_entity, str(i)])
        if type(arcgis_result) is dict:
          available_results_for_city.append(arcgis_result)
        nominatim_result = json.loads(df_nominatim_result.loc[spatial_entity, str(i)])
        if type(nominatim_result) is dict:
          available_results_for_city.append(nominatim_result)
        geonames_result = json.loads(df_geonames_result.loc[spatial_entity, str(i)])
        if type(geonames_result) is dict:
          available_results_for_city.append(geonames_result)

      arcgis_result = json.loads(df_arcgis_result.loc[spatial_entity, "0"]) # first result
      if type(arcgis_result) is dict:
        available_results_for_region.append(arcgis_result)
      nominatim_result = json.loads(df_nominatim_result.loc[spatial_entity, "0"]) # first result
      if type(nominatim_result) is dict:
        available_results_for_region.append(nominatim_result)
      geonames_result = json.loads(df_geonames_result.loc[spatial_entity, "0"]) # first result
      if type(geonames_result) is dict:
        available_results_for_region.append(geonames_result)

      country_code = ""
      country_name = ""
      region_name = ""
      city_name = ""
      if len(available_results_for_region) > 0:
        # the country of all thee geocoding results is supposed to be the same >> so, take the first result
        country_code = available_results_for_region[0]["country_code"]
        #country_name = available_results[2]["country_name"] # from GeoNames (I noticed some errors from ArcGis)
        country_name = spatial_entity.split(":")[0].strip()
        nb_entities = len(spatial_entity.split(":"))
        # print(i, "/", len(unique_spatial_entity_list))
        # print("spatial entity:", spatial_entity)
        # print("!!! INCONSISTENCY >> spatial entity:", spatial_entity)
        # for res in available_results:
        #  print("--", res["region_name"])
        region_name = ""
        if nb_entities > 1:
          region_name = choose_best_geocoding_result_for_spatial_entity_region_name(available_results_for_region)
        city_name = ""
        if nb_entities == 2 and region_name != "":
          other_entity = process_spatial_entity_name(spatial_entity.split(":")[1].strip())
          if other_entity not in process_spatial_entity_name(region_name):
            city_name = choose_best_geocoding_result_for_spatial_entity_city_name(available_results_for_city, region_name)
        if nb_entities >2 and region_name != "":
          city_name = choose_best_geocoding_result_for_spatial_entity_city_name(available_results_for_city, region_name)
        print(region_name, city_name)

      country_code_list.append(country_code)
      country_name_list.append(country_name)
      region_name_list.append(region_name)
      city_name_list.append(city_name)

    df_final_geocoding_result = pd.DataFrame(list(zip(unique_spatial_entity_list, country_name_list, country_code_list, region_name_list, city_name_list)),
                 columns=['spatial_entity_text', 'country_name', 'country_code', 'region_name', 'city_name'])

    df_final_geocoding_result.to_csv(final_geocoding_result_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)


  df_final_geocoding_result = pd.read_csv(final_geocoding_result_filepath, sep=";", keep_default_na=False, dtype=str, index_col=0)

  #df_raw_genome_grouped["country"] = country_list
  df_raw_genome["country_name"] = df_raw_genome["Geographic Location"].apply(lambda x:
                                                                 df_final_geocoding_result.loc[x,"country_name"])
  df_raw_genome["country_code"] = df_raw_genome["Geographic Location"].apply(lambda x:
                                                                 df_final_geocoding_result.loc[x,"country_code"])
  df_raw_genome["region_name"] = df_raw_genome["Geographic Location"].apply(lambda x:
                                                                 df_final_geocoding_result.loc[x,"region_name"])
  df_raw_genome["city_name"] = df_raw_genome["Geographic Location"].apply(lambda x:
                                                                            df_final_geocoding_result.loc[
                                                                              x, "city_name"])

  df_raw_genome_grouped = df_raw_genome[df_raw_genome["country_code"] != ""]

  # TODO: NOT SURE
  #df_raw_genome_grouped.drop("Geographic Location", inplace=True, axis=1)

  #sfsdfsd()



  # =====================================================================
  # STEP 5: host entity event_preprocessing
  # =====================================================================

  # process host names
    # mallard; gender M; age hatch year
    # mallard; gender F; age hatch year
    # duck; Anas platyrhynchos
    # duck; Anseriformes sp.
    # Bean goose(anser fabalis)
    # Black-winged_curlew
    # Duck(anas sp.)
    # Duck: mallard
    # chicken; commercial broiler-breeder
    # Broiler chicken
    # layer hen
  host_name_list = df_raw_genome_grouped["Host Name"].to_list()
  host_common_name_list = df_raw_genome_grouped["Host Common Name"].to_list()
  host_group_name_list = df_raw_genome_grouped["Host Group"].to_list()
  host_processed_list = []
  for i in range(len(host_name_list)):
    host_candidate_list = []
    host_name = host_name_list[i].lower()
    host_common_name = host_common_name_list[i].lower()
    host_group_name = host_group_name_list[i].lower()
    if ";" in host_name:
      parts = host_name.split(";")
      parts = [p.strip() for p in parts]
      for h in parts:
        if ":" in h:
          inner_parts = h.split(":")
          inner_parts = [p.strip() for p in inner_parts]
          host_candidate_list = host_candidate_list + inner_parts
        else:
          host_candidate_list.append(h)
      
    host_candidate_list.append(host_common_name)
    host_candidate_list.append(host_group_name)
    host_processed = ",".join(host_candidate_list)
    host_processed_list.append(host_processed)
    
  
  df_raw_genome_grouped["host"] = host_processed_list
  df_raw_genome_grouped.drop("Host Name", inplace=True, axis=1)
  
  print(output_filepath)
  df_raw_genome_grouped.to_csv(output_filepath, index=True, sep=";", quoting=csv.QUOTE_NONNUMERIC)












##################################################################################################
#
#
##################################################################################################
def process_spatial_entity_name(spatial_entity_text):

  # processing 1: remove unnecessary words
  spatial_entity_text = spatial_entity_text.lower().replace("governorate", "")
  spatial_entity_text = spatial_entity_text.lower().replace("oblast", "")
  spatial_entity_text = spatial_entity_text.lower().replace("province", "")
  spatial_entity_text = spatial_entity_text.lower().replace("region", "")
  spatial_entity_text = spatial_entity_text.lower().replace("prefecture", "")
  spatial_entity_text = spatial_entity_text.lower().replace("kraj", "")
  spatial_entity_text = spatial_entity_text.lower().replace("krai", "")
  spatial_entity_text = spatial_entity_text.lower().replace("district of", "")
  spatial_entity_text = spatial_entity_text.lower().replace("district", "")
  spatial_entity_text = spatial_entity_text.lower().replace("special region of", "")
  spatial_entity_text = spatial_entity_text.lower().replace("state", "")
  spatial_entity_text = spatial_entity_text.lower().replace("republic of", "")
  spatial_entity_text = spatial_entity_text.lower().replace("republic", "")
  spatial_entity_text = spatial_entity_text.lower().replace("respublika", "")
  spatial_entity_text = spatial_entity_text.lower().replace("county", "")
  spatial_entity_text = spatial_entity_text.lower().replace("city", "")
  spatial_entity_text = spatial_entity_text.lower().replace("city", "län")
  spatial_entity_text = spatial_entity_text.strip()
  if "federal capital territory" not in spatial_entity_text.lower():
    spatial_entity_text = spatial_entity_text.lower().replace("capital territory", "")

  # processing 2: if there is a comma, take the first part
  # example: Washington, D.C. >> Washington
  spatial_entity_text = spatial_entity_text.split(",")[0].strip()

  # processing 3: remove some unnecessary whitespaces
  # example: "Bà Rịa-Vũng Tàu" and "Bà Rịa - Vũng Tàu"
  spatial_entity_text_parts = spatial_entity_text.split("-")
  spatial_entity_text_parts = [p.strip() for p in spatial_entity_text_parts]
  spatial_entity_text = "-".join(spatial_entity_text_parts)

  # processing 4: if the last character is a quote, remove it
  # example: mangistauskaya oblast'
  if spatial_entity_text[-1] == "'":
    spatial_entity_text = spatial_entity_text.split("'")[0]

  # processing 5: if string has 1 character
  # example: "province 1"
  if len(spatial_entity_text) == 1:
    spatial_entity_text = ""

  # processing 6: unidecoding
  # example: Baden-Württemberg >> Baden-Wurttemberg
  spatial_entity_text = unidecode(spatial_entity_text)

  return spatial_entity_text



##################################################################################################
# It chooses the item with the max frequency. if there are many items having the same max value,
#   we take the first item.
#
##################################################################################################
# Difficult example1:
# spatial entity: USA: Monroe county, AK >> arkansas
# -- Alabama (arcgis)
# -- Florida (nominatim)
# -- Indiana (geonames)
# Difficult example2:
#  spatial entity: USA: Agassiz NWR, Parker Pool, Marshall county, MN
# -- Minnesota (arcgis)
# -- California (nominatim)
# -- Illinois (geonames)
def choose_best_geocoding_result_for_spatial_entity_region_name(spatial_entity_list):
  region_names = [loc["region_name"] for loc in spatial_entity_list if loc["region_name"] != "-1" and loc["region_name"] != ""]
  region_names = [process_spatial_entity_name(name) for name in region_names if process_spatial_entity_name(name) != ""]

  if len(region_names)>0:
    c = Counter(region_names)
    #print(c)
    return max(c, key=c.get)
  return ""


def choose_best_geocoding_result_for_spatial_entity_city_name(spatial_entity_list, region_name):
  #print("girdi", region_name)
  city_names = [loc["city_name"] for loc in spatial_entity_list if loc["region_name"] != "" and process_spatial_entity_name(loc["region_name"]) == region_name]
  #print(city_names)
  city_names = [process_spatial_entity_name(name) for name in city_names if process_spatial_entity_name(name) != ""]
  city_names = [city_name for city_name in city_names if city_name != "-1"]
  #print(city_names)

  if len(city_names)>0:
    c = Counter(city_names)
    #print(c)
    return max(c, key=c.get)
  return ""


##################################################################################################
#
#
# Some spatial entity texts are very detailed information, and this makes the geocoding task difficult
#  That is why, if we do not succeed in geocoding the original text, we split into a numerous smaller texts, see an example below
# example: "India: Dey Para,Nadia,WB"
# alternative_spatial_entity_list = ["India: Dey Para", "India: Nadia", "India: WB"]
##################################################################################################
def prepare_alternative_spatial_entity_lists_for_spatial_entity_disambiguation(spatial_entity_list):
  alternative_spatial_entity_list_of_list = []
  for text in spatial_entity_list:
    #print("--", text)
    alt_list = []
    if ":" in text:
      nb_items = text.count(",")+1
      country = text.split(":")[0].strip()
      parts = text.split(":")[1].split(",")
      parts = [p.strip() for p in parts]
      #print(parts)
      alt_list2 = []
      if nb_items>2:
        alt_list2 = [country + ": "+ ", ".join(c) for c in list(itertools.combinations(parts, nb_items-1))]
      alt_list1 = [country+": "+p for p in parts]
      alt_list = alt_list2 + alt_list1 # we add first the list with two terms
      #print(alt_list)
    alternative_spatial_entity_list_of_list.append(alt_list)
  return alternative_spatial_entity_list_of_list




###################################################################################
#
#
#
###################################################################################
def handle_incomplete_dates_in_genome_data_after_preprocessing(preprocess_filepath, output_filepath):
  df_preprocessed = pd.read_csv(preprocess_filepath, \
  usecols=["Genome ID", "Genome Name", "Isolation Country", "Serotype", "Segment2GenomeID", "GenBank Accessions", "Publication", "ReportDate",
           "Host Common Name", "Host Group", "host",  "Collection Year", "disease", "ObservationDate",
           "Subregion", "Region", "lat", "lng", "country_code", "country_name", "region_name", "city_name",
           "BioProject Accession", "Sequencing Platform", "Assembly Method", "Isolation Source"],\
                                sep=";", keep_default_na=False, dtype=str)

  
  df_preprocessed["date_ind"] = df_preprocessed["ReportDate"].apply(lambda x: len(x.split("-")))
  df_preprocessed_full_date = df_preprocessed[df_preprocessed["date_ind"] == 3]
  df_preprocessed_date_with_month = df_preprocessed[df_preprocessed["date_ind"] == 2]
  df_preprocessed_date_with_year = df_preprocessed[df_preprocessed["date_ind"] == 1]
  
  print(df_preprocessed_full_date.shape[0], df_preprocessed_date_with_month.shape[0], df_preprocessed_date_with_year.shape[0])
  
  
  date_list = df_preprocessed_date_with_month["ReportDate"].to_list()
  new_date_list = []
  for date_str in date_list:
    # example of "date_str": 2009-11
    # May-2006
    date_parts = date_str.split("-")
    new_date_str = date_str
    if len(date_parts[0]) == 4:
      new_date_str = new_date_str + "-01"
    else: # if len(date_parts[1]) == 4:
      new_date_str = "01-" + new_date_str
    new_date_list.append(new_date_str)
  df_preprocessed_date_with_month["ReportDate"] = new_date_list

    
  id_list = df_preprocessed_date_with_year["Genome ID"].to_list()
  id2year = dict(zip(df_preprocessed_date_with_year["Genome ID"], df_preprocessed_date_with_year["ReportDate"]))
  df_preprocessed_date_with_year.index = df_preprocessed_date_with_year["Genome ID"]
  df_preprocessed_date_with_year_new = df_preprocessed_date_with_year.loc[df_preprocessed_date_with_year.index.repeat(12)] # # 12 months
  for id in id_list:
    print("--", id)
    year_str = id2year[id]
    new_date_values = ["01-"+str(m)+"-"+year_str for m in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]]
    new_id_values = [str(id)+"."+m for m in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]]
    #print(df_preprocessed_date_with_year_new.loc[df_preprocessed_date_with_year_new.index == id, "ReportDate"])
    df_preprocessed_date_with_year_new.loc[df_preprocessed_date_with_year_new.index == id, "ReportDate"] = new_date_values
    df_preprocessed_date_with_year_new.loc[df_preprocessed_date_with_year_new.index == id, "ObservationDate"] = new_date_values
    df_preprocessed_date_with_year_new.loc[df_preprocessed_date_with_year_new.index == id, "Genome ID"] = new_id_values
    #print(df_preprocessed_date_with_year_new.loc[df_preprocessed_date_with_year_new.index == id, "ReportDate"])

  df_final = pd.concat([df_preprocessed_full_date, df_preprocessed_date_with_month, df_preprocessed_date_with_year_new])
  df_final.drop("date_ind", inplace=True, axis=1)
  print(df_final.shape[0])
  df_final["id"] = list(range(df_final.shape[0]))
  df_final.to_csv(output_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
   
      



def split_final_results_into_precise_and_imprecise_serovar(preprocess_adj_output_filepath, out_genome_preprocessing_folder):
  df = pd.read_csv(preprocess_adj_output_filepath, sep=";", keep_default_na=False, dtype=str)

  imprecise_serovar_filepath = os.path.join(out_genome_preprocessing_folder, "imprecise_serovar.csv")
  serovar1_list = ["h"+str(i) for i in range(20)]
  serovar2_list = ["n" + str(i) for i in range(20)]
  serovar_list = serovar1_list + serovar2_list + [""]
  df_raw_genome_with_imprecise_serovar = df[df["Serotype"].isin(serovar_list)]
  df_raw_genome_with_imprecise_serovar["id"] = list(range(df_raw_genome_with_imprecise_serovar.shape[0]))
  df_raw_genome_with_imprecise_serovar.to_csv(imprecise_serovar_filepath, index=True, sep=";", quoting=csv.QUOTE_NONNUMERIC)

  precise_serovar_filepath = os.path.join(out_genome_preprocessing_folder, "precise_serovar.csv")
  df_raw_genome_with_precise_serovar = df[~df["Serotype"].isin(serovar_list)]
  df_raw_genome_with_precise_serovar["id"] = list(range(df_raw_genome_with_precise_serovar.shape[0]))
  df_raw_genome_with_precise_serovar.to_csv(precise_serovar_filepath, index=True, sep=";", quoting=csv.QUOTE_NONNUMERIC)