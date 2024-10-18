'''
Created on Nov 16, 2021

@author: nejat
'''

import os
import json

import pandas as pd
import numpy as np

import src.consts as consts

from abc import ABC, abstractmethod



import csv
import re


from src.util_gis import resolve_manually_loc_name_ambiguity

from src.event_normalization.disease_entity_normalization import retrieve_disease_data_from_text
                                
from src.event_preprocessing.disease_entity_processing import retrieve_disease_from_raw_sentence

                                
from src.event_normalization.host_entity_normalization import retrieve_normalized_host_list, \
                                retrieve_NCBI_id_from_keyword, retrieve_NCBI_hierarchy_and_host_type_from_id
                                
from src.event_normalization.disease_entity_normalization import retrieve_normalized_disease_list
                                
from src.event_normalization.spatial_entity_normalization import search_geonames_for_admin1_loc_list, retrieve_normalized_loc_list

from src.util_gis import get_country_apolitical_name_from_alpha3_code, process_spatial_entity_abbreviation

from src.geocoding.relative_spatial_entity import idenitfy_relative_spatial_cardinal
from src.geocoding.geocode import geocode_with_geonames, geocode_raw_with_geonames
from src.geocoding.enhance_geonames_data_with_hierarchy import enhance_geonames_data_with_hierarchy
    
from iso3166 import countries


from src.event_preprocessing.spatial_entity_processing_in_title import apply_different_gecoders_on_spatial_entities_in_title_for_normalization,\
                                                                  normalize_spatial_entities_in_title_from_geocoding_results

from src.event_normalization.spatial_entity_normalization import update_geocoding_results_in_DB




class AbstractNormalization(ABC):
  
  def __init__(self):
    pass
  
  @abstractmethod
  def perform_normalization(self):
    pass




class NormalizationPadiweb(AbstractNormalization):
  
  def __init__(self):
    # input
    self.input_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
    self.input_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
    self.input_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
    # output
    self.output_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
    self.output_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
    self.output_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_SURVEILLANCE_PADIWEB
    
    

  def perform_normalization(self, disease_name, in_preprocessing_result_folder, out_norm_result_folder, data_folder):
    self.disease_name = disease_name
    self.in_preprocessing_result_folder = in_preprocessing_result_folder
    self.data_folder = data_folder
    self.out_norm_result_folder = out_norm_result_folder
    
    self.normalize_already_found_spatial_entities_in_title_for_all_articles()
    
    # ===========================================================================================
    # PART 6: Geocoding spatial entities in the main text of the articles 
    #            (with the help of the geocoded spatial entities found in the titles)
    #          In the end, it produces the file "extr_spatial_entities_info.csv", which has these information:
    #          - id_articleweb
    #          - local_sentence_id
    #          - value (i.e. spatial entity final name)
    #          - geonames_id
    #          - geonames_json
    #          - updated: if my geocoding method changed the GeoNames result provided by PADI-web. 1 if updated
    #          - prev_value (i.e. spatial entity previous name, as provided by PADI-web)
    #          - prev_geonames_json
    #          - ref_countries_for_geocoding
    #          - country_code
    # ===========================================================================================
    
    self.normalize_already_found_spatial_entities_in_text_with_disambiguation_for_all_articles()
    
    # ======
    extr_host_info_filepath = os.path.join(self.in_preprocessing_result_folder, \
                           self.input_preprocessed_host_entities_filename + "." + consts.FILE_FORMAT_CSV)
    norm_host_info_filepath = os.path.join(self.out_norm_result_folder, \
                           self.output_norm_host_entities_filename + "." + consts.FILE_FORMAT_CSV)    
    self.normalize_already_found_host_entities_in_text_for_all_articles(extr_host_info_filepath, norm_host_info_filepath)
    
    # ======
    
    in_extr_disease_info_filepath = os.path.join(self.in_preprocessing_result_folder, \
                           self.input_preprocessed_disease_entities_filename + "." + consts.FILE_FORMAT_CSV) 
    norm_disease_info_filepath = os.path.join(self.out_norm_result_folder, \
                           self.output_norm_disease_entities_filename + "." + consts.FILE_FORMAT_CSV)      
    self.normalize_already_found_disease_entities_in_text_for_all_articles(in_extr_disease_info_filepath, norm_disease_info_filepath)



  def normalize_already_found_host_entities_in_text_for_all_articles(self, in_extr_host_info_filepath, norm_host_info_filepath):
    df_extr_host_info = pd.read_csv(in_extr_host_info_filepath, sep=";", keep_default_na=False, encoding='utf8')
    
    article_id_list = []
    sentence_id_list = []
    host_type_list = []
    host_txt_list = []
    NCBI_id_list = []
    NCBI_hierarchy_list = []
    for index, row in df_extr_host_info.iterrows():
      article_id = row["article_id"]
      host_txt = row["host"]
      sentence_id = row["local_sentence_id"]
      print(index, "article id", article_id, "sentence", sentence_id)
      NCBI_id = retrieve_NCBI_id_from_keyword(host_txt)
      if NCBI_id is not None:
        NCBI_id_list.append(NCBI_id)
        article_id_list.append(article_id)
        sentence_id_list.append(sentence_id)
        host_txt_list.append(host_txt)
        host_type, NCBI_hierarchy = retrieve_NCBI_hierarchy_and_host_type_from_id(NCBI_id)
        host_type_list.append(host_type)
        NCBI_hierarchy_list.append(NCBI_hierarchy)
    
    df_extr_host_info_norm = pd.DataFrame({
        consts.COL_ARTICLE_ID: article_id_list,
       "local_sentence_id": sentence_id_list,
       "host_text": host_txt_list,
       "host_type": host_type_list,
       "host_id": NCBI_id_list,
       'host_hierarchy': NCBI_hierarchy_list
    })
    df_extr_host_info_norm = df_extr_host_info_norm.drop_duplicates(subset=[consts.COL_ARTICLE_ID, "local_sentence_id", "host_id"])
    df_extr_host_info_norm.to_csv(norm_host_info_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    

  def normalize_already_found_disease_entities_in_text_for_all_articles(self, in_extr_disease_info_filepath, norm_disease_info_filepath):
    df_extr_disease_info = pd.read_csv(in_extr_disease_info_filepath, sep=";", keep_default_na=False, encoding='utf8')
    
    article_id_list = []
    sentence_id_list = []
    disease_data_dict_list = []
    for index, row in df_extr_disease_info.iterrows():
      article_id = row["article_id"]
      disease_txt = row["disease"]
      sentence_id = row["local_sentence_id"]
      print(index, "article id", article_id, "sentence", sentence_id, "text", disease_txt)
      disease_data = retrieve_disease_data_from_text(self.disease_name, disease_txt)
      if disease_data is not None:
        disease_data_dict_list.append(str(disease_data))
        article_id_list.append(article_id)
        sentence_id_list.append(sentence_id)
      else:
        print(index, "=> disease_data", disease_data)
      
    
    df_extr_disease_info_norm = pd.DataFrame({
        consts.COL_ARTICLE_ID: article_id_list,
       "local_sentence_id": sentence_id_list,
       "disease": disease_data_dict_list,
    })
    df_extr_disease_info_norm = df_extr_disease_info_norm.drop_duplicates(subset=[consts.COL_ARTICLE_ID, "local_sentence_id", "disease"])
    df_extr_disease_info_norm.to_csv(norm_disease_info_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
    

  def normalize_already_found_spatial_entities_in_title_for_all_articles(self):
    title_locations_final_filepath = os.path.join(self.in_preprocessing_result_folder, "titles_final_locations" + "." + consts.FILE_FORMAT_CSV)
    title_locations_final_with_geocoding_results_filepath = os.path.join(self.out_norm_result_folder, "titles_final_locations_with_geocoding_results" + "." + consts.FILE_FORMAT_CSV)
    apply_different_gecoders_on_spatial_entities_in_title_for_normalization(title_locations_final_filepath, title_locations_final_with_geocoding_results_filepath)
    
    geonames_results_in_batch_filepath = os.path.join(self.out_norm_result_folder, "geonames_results_in_batch" + "." + consts.FILE_FORMAT_CSV)
    title_locations_final_with_geonames_filepath = os.path.join(self.out_norm_result_folder, "title_locations_final_with_geonames" + "." + consts.FILE_FORMAT_CSV)
    normalize_spatial_entities_in_title_from_geocoding_results(title_locations_final_with_geocoding_results_filepath, \
                                                               geonames_results_in_batch_filepath, \
                                                               title_locations_final_with_geonames_filepath)
    
    

  def normalize_already_found_spatial_entities_in_text_with_disambiguation_for_all_articles(self):
    articles_filepath = os.path.join(self.in_preprocessing_result_folder, \
                           consts.PADIWEB_ARTICLES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    media_source_resolution_filepath = os.path.join(self.in_preprocessing_result_folder, "media_source_resolution" + "." + consts.FILE_FORMAT_CSV)
    title_locations_final_with_geonames_filepath = os.path.join(self.out_norm_result_folder, "title_locations_final_with_geonames" \
                                           + "." + consts.FILE_FORMAT_CSV)
    ext_info_extr_padiweb_filepath = os.path.join(self.in_preprocessing_result_folder, self.input_preprocessed_spatial_entities_filename + "." + consts.FILE_FORMAT_CSV)
    rel_ext_extr_info_filepath = os.path.join(self.in_preprocessing_result_folder, "relevant_extended_extracted_information" + "." + consts.FILE_FORMAT_CSV)
    extr_spatial_entities_info_filepath = os.path.join(self.out_norm_result_folder, self.output_norm_spatial_entities_filename + "." + consts.FILE_FORMAT_CSV)
    ext_sentences_filepath = os.path.join(self.in_preprocessing_result_folder, \
                          consts.PADIWEB_EXT_SENTENCES_CSV_FILENAME + "." + consts.FILE_FORMAT_CSV)
    self.geonames_results_in_batch_filepath = os.path.join(self.out_norm_result_folder, "geonames_results_in_batch" + "." + consts.FILE_FORMAT_CSV)
    self.process_spatial_entities_in_sentences_for_all_articles(articles_filepath, ext_sentences_filepath, \
                                                                media_source_resolution_filepath,
                                                                title_locations_final_with_geonames_filepath,
                                                                ext_info_extr_padiweb_filepath,\
                                                                rel_ext_extr_info_filepath,
                                                                extr_spatial_entities_info_filepath,
                                                                self.geonames_results_in_batch_filepath)
    
    geonames_data_with_hierarchy_filepath = os.path.join(self.out_norm_result_folder, consts.GEONAMES_HIERARCHY_INFO_FILENAME + "." + consts.FILE_FORMAT_CSV)
    # find the hierarchical information of the geocoded spatial entities found in the previous method
    self.extract_geonames_hierarchy_info_for_all_spatial_entities(extr_spatial_entities_info_filepath, geonames_data_with_hierarchy_filepath)
    
    
    
    
  def process_spatial_entities_in_sentences_for_all_articles(self, articles_filepath, sentences_filepath, \
                                                             media_source_resolution_filepath,\
                                                             title_locations_final_with_geonames_filepath, \
                                                             preprocessed_spatial_entities_padiweb_filepath,\
                                                             rel_ext_extr_info_filepath, \
                                                             extr_spatial_entities_info_filepath,
                                                             geonames_results_in_batch_filepath):
    self.df_articles = pd.read_csv(articles_filepath, sep=";", keep_default_na=False, encoding='utf8')

    df_title_geocoding_results = pd.read_csv(title_locations_final_with_geonames_filepath, sep=";", keep_default_na=False)
    
    df_media_source = pd.read_csv(media_source_resolution_filepath, sep=",", keep_default_na=False)
    #source_to_country_assoc = dict(zip(df_media_source["url"].str.replace("www.",""), df_media_source["pub_country"]))
    self.article_id_to_source_country_code_assoc = dict(zip(df_media_source[consts.COL_ARTICLE_ID], df_media_source["pub_country_code"]))    
    
    self.article_id_to_text_assoc = dict(zip(self.df_articles[consts.COL_ID], self.df_articles[consts.COL_TEXT]))    

    self.df_sentences = pd.read_csv(sentences_filepath, sep=";", keep_default_na=False, encoding='utf8')
    #self.sentence_id_to_article_id = dict(zip(self.df_sentences[consts.COL_ID], self.df_sentences[consts.COL_ARTICLE_ID]))
    self.sentence_id_to_text = dict(zip(self.df_sentences[consts.COL_ID], self.df_sentences[consts.COL_TEXT]))
    #self.article_id_to_title = dict(zip(self.df_articles[consts.COL_ID], self.df_articles[consts.COL_TITLE]))
    #self.sentence_id_to_local_sentence_id = dict(zip(self.df_sentences[consts.COL_ID], self.df_sentences["local_sentence_id"]))

    self.df_batch_geonames_results = pd.read_csv(geonames_results_in_batch_filepath, sep=";", keep_default_na=False)
    self.inverted_indices_for_df_batch_geonames_results = dict(zip(self.df_batch_geonames_results["text"], self.df_batch_geonames_results.index))

    
    df_spatial_entities_extr = pd.read_csv(preprocessed_spatial_entities_padiweb_filepath, sep=";", keep_default_na=False)

    # # df_ext_info_extr[consts.COL_ARTICLE_CLASSIF_LABEL_ID] = df_ext_info_extr[consts.COL_ARTICLE_CLASSIF_LABEL_ID].replace("", "[]")
    # # df_ext_info_extr = df_ext_info_extr[df_ext_info_extr[consts.COL_CLASSIF_LABEL_ID] != '[]']
    # # df_ext_info_extr[consts.COL_CLASSIF_LABEL_ID] = [eval(x) for x in df_ext_info_extr[consts.COL_CLASSIF_LABEL_ID]]
    # # filter aggregated sentences by classif label
    # drop_vals = [k for k in df_ext_info_extr.index \
    #              if str(consts.CLASSIF_LABEL_CURR_EVENT_ID)+" "+str(consts.CLASSIF_LABEL_DESCR_EPID_ID)\
    #               != df_ext_info_extr.loc[k, consts.COL_SENTENCE_CLASSIF_LABEL_ID]\
    #              and str(consts.CLASSIF_LABEL_CURR_EVENT_ID)+" "+str(consts.CLASSIF_LABEL_GENERAL_EPID_ID)\
    #               != df_ext_info_extr.loc[k, consts.COL_SENTENCE_CLASSIF_LABEL_ID]\
    #              and str(consts.CLASSIF_LABEL_CURR_EVENT_ID)+" "+str(consts.CLASSIF_LABEL_TRANS_PATHWAY_ID)\
    #               != df_ext_info_extr.loc[k, consts.COL_SENTENCE_CLASSIF_LABEL_ID]\
    #             ]
    # df_ext_info_extr_only_outbreak = df_ext_info_extr.drop(drop_vals)
    #
    # df_ext_info_extr_only_outbreak.to_csv(rel_ext_extr_info_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    #
    # #df_sentence_spatial_entities = df_ext_info_extr_only_outbreak[~((df_ext_info_extr_only_outbreak["geonames_id"] == 'NULL') | (df_ext_info_extr_only_outbreak["geonames_id"] == '-1'))]
    # #print(df_sentence_spatial_entities["geonames_id"])
    #
    # # filepath = os.path.join(self.out_csv_folder, "geonames_results_in_batch" + "." + consts.FILE_FORMAT_CSV)
    # # self.df_batch_geonames_results = pd.read_csv(filepath, sep=";", keep_default_na=False)
    # # self.inverted_indices_for_df_batch_geonames_results = dict(zip(self.df_batch_geonames_results["text"], self.df_batch_geonames_results.index))
    
    
    # ----------------------------------------------------------------------------
    
    ##### df_sentence_spatial_entities = df_ext_info_extr_only_outbreak[~((df_ext_info_extr_only_outbreak["geonames_id"] == 'NULL') | (df_ext_info_extr_only_outbreak["geonames_id"] == '-1'))]


    df_list = []
    article_id_list = np.unique(self.df_articles[consts.COL_ID].to_numpy())
    tot = len(article_id_list)
    #article_id_list = np.unique(df_ext_info_extr_only_outbreak[consts.COL_ARTICLE_ID_RENAMED].to_numpy())
    # avian influenza specific article ids
    #article_id_list = ["3UFCKR6BOV", "RA419Z60HU", "7QZJNVLHTS","L7CG25TT3N", "RA419Z60HU", "JMO1JRJDJW", "5PF9FM6U2Q", "MDQA110VY0", "4SJJ0Y1X0D", "WAYOPZHA5J", "03WD3TAW5C", "02TL7DYIWC", "3UFCKR6BOV", "3TB9FO1ENQ", "3Z3PDHGCJX", "7V4FWGZU0D", "3SJZPBCWT8"]
    #article_id_list = ['0704MFBINF', '0AIB1VZESR', '0GJ5LF54J6', '0O87F8LCK8', '0PDPT5PYS9', '0Q67XKFHXV', '0QI0DU2GNU', '0R0N5UMR7V', '0T726GLBEA', '0U2GRR2W75', '11KVX7JAZQ', '177F18VZQ6', '1971AMZLI4', '19YNIA6HWJ', '1HBU5W57C2', '1IZ1EC2W83', '1MYCH5YFK6', '1NBGIHOCIT', '1PXB83Z9OZ', '1QNUZOLTKY', '1YUA5R509F', '20Y4V9MNXS', '23VOJ0K28O', '26H3V8RJRW', '29LLWFTZAW', '2B5DD6N26Q', '2K9UZD3XIX', '2MM34Z6XQD', '2WHP1T965T', '2XFXH9KTA9', '32T0PDJHRJ', '35MWSOF9KQ', '3EB9NEG0L4', '3NUS4U8AUQ', '3OYHQ9JHHZ', '42Q43OB4EJ', '44C34NZXZ5', '4EEWL8L9VY', '4HFF32S9A3', '4KD0R8BL82', '4KWZ759TMU', '4QLIKXFT8G', '4ZV0B9TBRR', '50232HP69P', '5BRDOVUG9A', '5H7O5D7GH3', '5IXIO97RL3', '5RVEUNRXWW', '66B7VEPYJC', '6K6Y4Z43XM', '6M4OFDQ1V7', '6VQAIA8CQK', '7BMWWVFIDM', '7FJCAPMG0O', '7PSY5T8N6Z','7V3U0C7JKO', '7VZLHZ0BQ4', '83GIYBIRLO', '8CUB576YQR', '8ESFZMJTRR', '8MDE8OPKL5', '8XBAJZZ6H7', '93BLG5CSB3', '9BQPNKC65Y', '9CM8NPLXJ4', '9F70AFXWT4', '9OJXF2ARB1', 'A5OXA2W0C9', 'AB8I05RVEU', 'AQ3UV5S7JC', 'AT5RYQE902', 'B1X567DS92', 'B32FDD4DR7', 'B3FG3KTU3M', 'BB0E3P2BFK', 'BBW8X0T69S', 'BJ9MJ7OSYB', 'BMML3PQARX', 'BOSH3B0JCM', 'C0TD42T355', 'C3I8LR8CS0', 'C4DRNC9P3A', 'C6J8RBT7A4', 'C6XB5Z8WX3', 'C8ONBYSSCX', 'CFN0LX1WGP', 'CNN1P1NC6N', 'D5ZRZPHVAE',  'DFVM16LW3W', 'DFW52S0S52', 'DPUUNKIXYB', 'DX000VYLHT', 'EDHT13DCCY', 'EFIU61MZVK', 'EHTK8L7QQI', 'EQIM3FZKKZ', 'F1PABG9LE6', 'FKMIKFZLI4', 'FLYUELZXTM', 'FM1QCAPW6U', 'FTO5Z62ML8', 'FV54V3738U', 'G24WXYPZ6D', 'G796NHJWT4', 'GAM5PNTXTD', 'GCVIA9OYFU', 'GGABSZD8XD', 'GYV9KNJF87', 'H0Y3PKMV9R', 'H824A9K3O8', 'HHPVWDEQA5', 'HN8ZP9GNR1', 'HSV0GU7M5H', 'I4JPRK5JRY', 'I71R1AJSDN', 'IILH9U8UOM', 'IWA4TWARSG', 'IYAF3MFHXV', 'J1VH02P435', 'J2IJ76N5RG', 'J9OIEXMNKH', 'JXO0TCBEV5', 'K47CT7VJS1', 'KAI45NB57R', 'KSP7XXKXW3', 'KXO0DLMXQD', 'L0NSKTP9XU', 'L8XLYXM18U', 'LNAHSV81G7', 'LOHLOMJR48', 'LTAXW0A9GQ', 'LU7MTQC6WR', 'LZBC5ZICR5', 'M26ISS2HP7', 'M9UG0MEBZH', 'MAV7WSFHSX', 'MD7TJJJ9PE', 'ME0G9QSODT', 'N8QC8A45VB', 'NAEEBAOSIR', 'NBXBIIS12Y', 'NCFS2ZSRQX', 'NPMZCQ4YVA', 'NYX45S6NOB', 'OHQKACB3SF', 'OTN4EGM0Z2', 'OX47LQ6Y9M', 'P72VS2X11U', 'PBNMIE6AIA', 'PK9A02W4A5', 'PW3JQZFD0G', 'Q4FWP59MW2', 'Q5ZBXQK2TH', 'Q9JYOYQGV7', 'QJA3SD3OR0', 'QN55CGR17Y', 'QVMX2DFD8W', 'R1VO5EL0DR', 'RBTFJQNXLK', 'RGKHBV1U53', 'SC65ZXMCIU', 'SDDG5H1HEP', 'SE2C6SZ2JX', 'SNDFVZ2ABY', 'SQROER2L0O', 'T3X02OR2DS', 'TFMQWS8ZU0', 'TK1DHQTHN0', 'TVTQ5L42QG', 'TZ8ONQHMGU', 'U9FRVEE3NW', 'UCCR96C00Z', 'UDICB3UWQK', 'UFBPKE384C', 'UGAM2IVWYL', 'V0UM2KAAA0', 'V497W2UN9G', 'V929V2OROT', 'VPKODNNLQJ', 'VSFMLGU6Y7', 'VXL09XVZ47', 'W27AZA3ZKE', 'W467SMC44K', 'WCAK88I2P1', 'WFZNTYIV8Q', 'WH33PY9YQU', 'WV5TX9HML2', 'WYN1TQWVXJ', 'X4PMJ5GDXB', 'X916RVD5KH', 'XFWQDHPZ26', 'XJVHT538J5', 'XK8F5M8DU8', 'XNQC4XG3P4', 'XOT2H27KSV', 'XR0W2XVJEQ', 'Y16Y5XRX33', 'Y305BLMZBL', 'YBB5PC24QN', 'YN72HT8R2X', 'Z4U1SYU47C', 'ZOX46CHRAX', 'ZVAZZVTV2V']    #article_id_list = ['0704MFBINF', '0AIB1VZESR', '0O87F8LCK8']

    # west nile specific article ids
    #article_id_list = ["KYW5N3E75G", "1SPERC4WN6", "1SXLMRP77B", "2C6ITS4U0O", "2KXMGAV0UH", "3WL8NSPWZC", "4VR9F44JTM", "6BJ46LALGH", "6BLBA7NUGW", "8DIUIT7I4X", "97S27F2LGU", "CCJZ7ZR4GG", "BKO2QPN8XN", "D4WCIUFQ5P", "DW9NS0ZRPL", "FDUPJDMXBM", "G1FNCLNMLF", "GN026JLHE3", "GOC7W8PWAY", "JFRJW5MVYN", "KIJ81LWVBO", "MDMW68AOF5", "NE32DE5J4X", "P0WZ0UZZPZ", "RMC98UXBMP", "V4E3TKDJKM", "VU805K23Y6", "X8DZJUPSX0", "Z4JJD8RFHJ", "ZAP5DFBPEG", "ZSR2QTMK2O"]
    #article_id_list = ["11KVX7JAZQ"]
    for indx, article_id in enumerate(article_id_list):
      print("||----------- ", indx, article_id,"\\",tot)
      #time.sleep(0.5)
      
      df_spatial_entities_by_article = df_spatial_entities_extr[df_spatial_entities_extr[consts.COL_ARTICLE_ID_RENAMED] == article_id]
      print(df_spatial_entities_by_article)
      
      df_title_geocoding_results_by_article = df_title_geocoding_results[df_title_geocoding_results["article_id"] == article_id]
      df_title_geocoding_results_by_article = df_title_geocoding_results_by_article.drop_duplicates(subset=['final_value'])
      print("!!!")
      print(df_title_geocoding_results_by_article)  

      df_title_geocoding_results_by_article = df_title_geocoding_results_by_article[df_title_geocoding_results_by_article["final_geonames_json"] != json.dumps("-1", ensure_ascii=False)]
      
#      # if there is any valid geocoding/JSON info
#      if df_title_geocoding_results_by_article.shape[0]>0:
#        # if df_title_country_by_article.shape[0]<=1 and df_title_country_by_article.shape[0]>0:
#        df_spatial_entities_by_article = self.process_spatial_entities_in_sentences_by_article(article_id, \
#                                                   df_ext_info_by_article, df_title_geocoding_results_by_article)
#        df_list.append(df_spatial_entities_by_article)
#        #print(df_spatial_entities_by_article)

      # TODO: recently changed >> before, if there is no any spatial entity in the title, we were not processing spatial entities in the text
      df_spatial_entities_by_article = self.process_spatial_entities_in_sentences_by_article(article_id, \
                                                   df_spatial_entities_by_article, df_title_geocoding_results_by_article)
      print(df_spatial_entities_by_article)
      df_list.append(df_spatial_entities_by_article)
      
    print("---")
    print(df_list)
    df_all_spatial_entities = pd.concat(df_list)
    print("!!!---!!!---!!! >> ", "extr_spatial_entities_info")
    df_all_spatial_entities.to_csv(extr_spatial_entities_info_filepath, index=True, sep=";", quoting=csv.QUOTE_NONNUMERIC)    
    
    
  # it returns a list of geonames_ids and a list of geonames data
  # test these padiweb articles:
  #  W0ABEZRTYZ
  #  W18BGBUEDV
  #  W58ED3YOXR
  #  WB9JINONGI
  #  WLU21U85VO
  #  WU18S45M79 >> good
  #  XOR6MDO3WW >> good
  #  Y7LP0XDJEV >> good
  #  DCS19Q8BAR >> good
  #
  # note that the sentences in 'df_ext_info_by_article' are already ordered from the beginning to the end 
  def process_spatial_entities_in_sentences_by_article(self, article_id, df_ext_info_by_article,\
                                                       df_title_geocoding_results_by_article):
    sentence_ids = np.unique(df_ext_info_by_article[consts.COL_SENTENCE_ID].to_numpy())
    local_sentence_ids = np.unique(df_ext_info_by_article["local_sentence_id"].to_numpy())
    title_geonames_json_list = [json.loads(geo_json) for geo_json in df_title_geocoding_results_by_article["final_geonames_json"].to_list() if geo_json != "-1"]
    title_country_codes = [geo_json["countryCode"] for geo_json in title_geonames_json_list]
    title_names = [value for value in df_title_geocoding_results_by_article["final_value"]]
    #title_geonames_id_list = [geo_json["geonameId"] for geo_json in title_geonames_json_list]
  
    media_source_spatial_entity_entry = []
    media_source_country_code_alpha3 = self.article_id_to_source_country_code_assoc[article_id]
    if media_source_country_code_alpha3 != '-1':
      media_source_country_code_alpha2 = countries.get(media_source_country_code_alpha3).alpha2
      media_source_country_name = get_country_apolitical_name_from_alpha3_code(media_source_country_code_alpha3)
      media_source_spatial_entity_entry = [0, media_source_country_code_alpha2, media_source_country_name]
      
  
    #------------------------------------------------------------------
    # columns: id  token_index  type  label  value  id_articleweb   geonames_id  geonames_json  sentence_id  classificationlabel_id  local_sentence_id
    # Note that the sentences in 'df_ext_info_by_article' are already ordered from the beginning to the end
    SLIDING_WINDOW_LENGTH = 4 # 4 previous sentences 
    sliding_window_for_history = [] # e.g. [(local_sentence_id1, country_code1, entity_name1), etc.]
    sliding_window_for_all_history = [] # (for booster) the only difference with 'sliding_window_for_history': we do not delete any items
    for i in range(df_title_geocoding_results_by_article.shape[0]):
      sliding_window_for_history.append((0, title_country_codes[i], title_names[i]))
      sliding_window_for_all_history.append((0, title_country_codes[i], title_names[i]))
    if len(sliding_window_for_history) == 0 and len(media_source_spatial_entity_entry)>0:
      sliding_window_for_history.append(tuple(media_source_spatial_entity_entry))
      sliding_window_for_all_history.append(tuple(media_source_spatial_entity_entry))
      
    df_spatial_entities_by_article = []
    df_list = []
    print(local_sentence_ids)
    if len(local_sentence_ids)>0:
      for indx in range(len(local_sentence_ids)):
        local_sentence_id = local_sentence_ids[indx]
        print("-- sentence", local_sentence_id)
        sentence_id = sentence_ids[indx]
        
        # remove the old entries based on time limit
        sliding_window_for_history = [entry for entry in sliding_window_for_history if (local_sentence_id-entry[0])<=SLIDING_WINDOW_LENGTH]
  
        # =================================================
        # booster step: when avian influenza or west nile related key words appear in some sentences
        if len(sliding_window_for_history) == 0 and len(sliding_window_for_all_history)>0:
          sentence_text = self.sentence_id_to_text[sentence_id]
          disease_info = retrieve_disease_from_raw_sentence(sentence_text, self.disease_name)
          if disease_info is not None: # TODO: we might extent this with: "tested positive", "confirmed"
            sliding_window_for_history = sliding_window_for_all_history
            
        #print(local_sentence_id, sliding_window_for_history)
        df_ext_info_by_sentence = df_ext_info_by_article[df_ext_info_by_article["local_sentence_id"] == local_sentence_id]
        # in the list of spatial entities to check in the next iteration,
        # if we have already seen them (even though they are in the current window),
        # then add the country code into sliding window
        for entity_name in df_ext_info_by_sentence["value"].to_list():
          print(entity_name)
          if entity_name not in [entry[2] for entry in sliding_window_for_history]:
            for entry in sliding_window_for_all_history:
              if entity_name == entry[2]: 
                sliding_window_for_history.append(entry)
        
        df_new_sentence_spatial_entities = self.process_spatial_entities_in_sentence(article_id, local_sentence_id,\
                                       df_ext_info_by_sentence, sliding_window_for_history)
        df_list.append(df_new_sentence_spatial_entities)
        # -------------------------------------------------
        # update sliding_window_for_history
        for index, row in df_new_sentence_spatial_entities.iterrows():
          sliding_window_values = [entry[2] for entry in sliding_window_for_history]
          if row["value"] not in sliding_window_values:
            sliding_window_for_history.append((local_sentence_id, row["country_code"], row["value"]))
            sliding_window_for_all_history.append((local_sentence_id, row["country_code"], row["value"]))
      
      
      df_spatial_entities_by_article = pd.concat(df_list)
      print(df_spatial_entities_by_article)
      print("--!--")
      
      # finally, add the title entities, if not included from the news content
      #print(df_spatial_entities_by_article["geonames_id"].to_list())
      for index, row in df_title_geocoding_results_by_article.iterrows():
        name = row["final_value"]
        geonames_json = json.loads(row["final_geonames_json"])
        geonameId = geonames_json["geonameId"]
        country_code_alpha2 = geonames_json["countryCode"]
        country_code_alpha3 = countries.get(country_code_alpha2).alpha3
        feature_code =  "-1"
        if "fcode" in geonames_json:
          feature_code = geonames_json["fcode"]
        elif "code" in geonames_json:
          feature_code = geonames_json["code"]
        country_bias_alpha3 = row["media_source_country_code"]
        country_bias_alpha2 = '-1'
        if country_bias_alpha3 != '-1':
          country_bias_alpha2 = countries.get(country_bias_alpha3).alpha2
        if str(geonameId) not in df_spatial_entities_by_article["geonames_id"].to_list():
          new_row = pd.DataFrame({"id_articleweb":[article_id], "value": [name], "geonames_id": [geonameId],\
                                  "geonames_json": [json.dumps(geonames_json, ensure_ascii=False)], "sentence_id": ['-1'], "token_index": ['-1'],\
                                  "local_sentence_id": ['-1'], "country_code": [country_code_alpha3], "type": ['GPE'], \
                                  "feature_code": [feature_code], "ref_countries_for_geocoding": [country_bias_alpha2]})
          df_spatial_entities_by_article = df_spatial_entities_by_article.append(new_row)
        
      return(df_spatial_entities_by_article)



  # here, we perform a geocoding task with disambiguation. This disambiguation is ensured with a sliding window
  def process_spatial_entities_in_sentence(self, article_id, local_sentence_id, df_ext_info_by_sentence, \
                                           sliding_window_for_history):
    df_sentence_spatial_entities = df_ext_info_by_sentence[~((df_ext_info_by_sentence["geonames_id"] == 'NULL') | (df_ext_info_by_sentence["geonames_id"] == '-1'))]
    
    # ----------------------------------------
    # sometimes, geoname contains the key "country_code", sometimes the key "countryCode"
    # if the key "country_code" does not exist, create this key from the key "countryCode"
    geoname_json_list= []
    for index, row in df_sentence_spatial_entities.iterrows():
      geoname_json = json.loads(row["geonames_json"])
      if "country_code" not in geoname_json and "countryCode" in geoname_json:
        geoname_json["country_code"] = geoname_json["countryCode"]
      geoname_json_list.append(json.dumps(geoname_json))
    df_sentence_spatial_entities["geonames_json"] = geoname_json_list
    # ----------------------------------------
    
    drop_vals = [k for k in df_sentence_spatial_entities.index \
             if "country_code" not in json.loads(df_sentence_spatial_entities.loc[k, "geonames_json"])]
  
  
    drop_vals = [k for k in df_sentence_spatial_entities.index \
                 if "country_code" not in json.loads(df_sentence_spatial_entities.loc[k, "geonames_json"])]
    # it can be ocean, continent, etc.
    df_sentence_spatial_entities = df_sentence_spatial_entities.drop(drop_vals)
    print(df_sentence_spatial_entities["value"])
    # remove GPEs named "turkey", if often conflicted
    df_sentence_spatial_entities = df_sentence_spatial_entities[df_sentence_spatial_entities["value"] != "turkey"] # TODO
    
    fcode_list = []
    for index, row in df_sentence_spatial_entities.iterrows():
      x = row["geonames_json"]
      if "code" in json.loads(x):
        fcode_list.append(json.loads(x)["code"])
      elif "fcode" in json.loads(x):
        fcode_list.append(json.loads(x)["fcode"])
      else:
        fcode_list.append("-1")
    df_sentence_spatial_entities["feature_code"] = fcode_list
    df_sentence_spatial_entities = df_sentence_spatial_entities[df_sentence_spatial_entities["type"].isin(["GPE", "outbreak-related-location"])]
    json_country_codes_alpah2 = df_sentence_spatial_entities["geonames_json"].apply(lambda x: json.loads(x)["country_code"])
    df_sentence_spatial_entities["country_code"] = [countries.get(country_code_alpha2).alpha3 for country_code_alpha2 in json_country_codes_alpah2]
    
    df_sentence_spatial_entities = df_sentence_spatial_entities.drop_duplicates(subset=['position', 'geonames_id'])
  
    df_sentence_spatial_entities["updated"] = 0
    df_sentence_spatial_entities["prev_geonames_json"] = -1
    df_sentence_spatial_entities["ref_countries_for_geocoding"] = -1
    df_sentence_spatial_entities["prev_value"] = df_sentence_spatial_entities["value"]
    
    df_sentence_spatial_entities_copy = df_sentence_spatial_entities.copy()
    print("!!!___!!!", article_id)
    print(df_sentence_spatial_entities_copy)
    
    # ----------------------------------------------
    for index, row in df_sentence_spatial_entities_copy.iterrows():
      text = row["value"]
      text_init = row["value"]
      media_source_country_code_alpha3 = self.article_id_to_source_country_code_assoc[article_id]
      media_source_country_code_alpha2 = "-1"
      if media_source_country_code_alpha3 != "-1":
        media_source_country_code_alpha2 = countries.get(media_source_country_code_alpha3).alpha2
      text = process_spatial_entity_abbreviation(text, media_source_country_code_alpha2) # TODO: I am not sure if requiring country code for obtaining the name of a gievn entity is too restrictive.
      #if row["value"] != text:
      #  print(row["value"], " ==> ", text)
      if ".." in text:
        rgx = r"(?i)[.]{2,}$" # remove multiple dots in the end
        text = re.sub(rgx, "", text)
      rgx = "(?i)(^the )"
      text = re.sub(rgx, "", text)
      rgx = "(?i)( region)"
      text = re.sub(rgx, "", text)
      rgx = "(?i)(^upper)|(^Upper)|(^lower)|(^Lower)" # >> for Upper and Lower Rhine => it is a complicated situation 
      text = re.sub(rgx, "", text)
      rgx = "(^-)|(-$)|(:$)"
      text = re.sub(rgx, "", text)
      rgx = "(^and )|( and$)"
      text = re.sub(rgx, "", text)
      # if text != text.upper(): # remove a dot point from the end, e.g. Pa. >> we do not treat U.K.
      #   rgx = "(\.$)"
      #   text = re.sub(rgx, "", text)
      text = text.strip()
      #print(text_init, text)
      if text_init != text:
        # print(text, text_init)
        df_sentence_spatial_entities.loc[index, "value"] = text
                # ==================================================
        # retrieve the results from our database
        locs = None
        print(text)
        if text in self.inverted_indices_for_df_batch_geonames_results:
          print("DB ACCESS!!!!")
          gindx = self.inverted_indices_for_df_batch_geonames_results[text]
          row_locs = self.df_batch_geonames_results.iloc[gindx]
          locs = [json.loads(row_locs[str(i)]) for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST)] # we access the columns
        # ==================================================
        else:
          locs = geocode_raw_with_geonames(text)
        if text not in self.inverted_indices_for_df_batch_geonames_results:  # update DB
          df_batch, inverted_indices = update_geocoding_results_in_DB(self.geonames_results_in_batch_filepath, text)
          self.df_batch_geonames_results = df_batch
          self.inverted_indices_for_df_batch_geonames_results = inverted_indices
        #
        if len(locs) == 0 or "countryCode" not in locs[0]:
          df_sentence_spatial_entities.loc[index, "value"] = "-1" # to remove
        else:
          df_sentence_spatial_entities.loc[index, "feature_code"] = locs[0]["fcode"]
          df_sentence_spatial_entities.loc[index, "geonames_id"] = str(locs[0]["geonameId"])
          df_sentence_spatial_entities.loc[index, "updated"] = 0
          df_sentence_spatial_entities.loc[index, "country_code"] = countries.get(locs[0]["countryCode"]).alpha3
          df_sentence_spatial_entities.loc[index, "value"] = text
          df_sentence_spatial_entities.loc[index, "prev_value"] = text
          df_sentence_spatial_entities.loc[index, "geonames_json"] = json.dumps(locs[0], ensure_ascii=False)
    
    df_sentence_spatial_entities = df_sentence_spatial_entities[df_sentence_spatial_entities["value"] != "-1"]
    print(df_sentence_spatial_entities[["geonames_id", "value", "country_code", "feature_code"]])
    
    # ---------------------------------------
    print("--!!-- first, resolve the general disambiguity")
    # first, resolve the general disambiguity
    df_sentence_spatial_entities_copy = df_sentence_spatial_entities.copy()
    media_source_country_code = self.article_id_to_source_country_code_assoc[article_id]
    article_text = self.article_id_to_text_assoc[article_id]
    for index, row in df_sentence_spatial_entities_copy.iterrows():
      spatial_entity = row["value"]
        
      processed, loc_type, country_code_alpha3, fcode = resolve_manually_loc_name_ambiguity(spatial_entity, media_source_country_code, article_text)  
      print(spatial_entity, processed, loc_type, country_code_alpha3)
      if processed and loc_type != "continent":
        only_country = False
        country_code_alpha2 = countries.get(country_code_alpha3).alpha2
        country_bias_list = []
        country_limit = None
        text = spatial_entity
        if loc_type != "country":
          country_limit = get_country_apolitical_name_from_alpha3_code(country_code_alpha3)
          text = spatial_entity+", "+country_limit
        else: # = country
          ####country_bias_list = [country_code_alpha2]
          only_country = True
          
          
        # ==================================================
        # retrieve the results from our database
        locs = None
        if text in self.inverted_indices_for_df_batch_geonames_results:
          #print("DB ACCESS!!!!")
          gindx = self.inverted_indices_for_df_batch_geonames_results[text]
          row_locs = self.df_batch_geonames_results.iloc[gindx]
          locs = [json.loads(row_locs[str(i)]) for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST)] # we access the columns
        # ==================================================
        res = geocode_with_geonames(spatial_entity, country_bias_list, locs, country_limit, only_country)
        #print(res)
        if text not in self.inverted_indices_for_df_batch_geonames_results: # and res["geonamesId"] != '-1':  # update DB
          text = spatial_entity
          if country_limit is not None:
            text = spatial_entity+", "+country_limit
          df_batch, inverted_indices = update_geocoding_results_in_DB(self.geonames_results_in_batch_filepath, text)
          self.df_batch_geonames_results = df_batch
          self.inverted_indices_for_df_batch_geonames_results = inverted_indices
        df_sentence_spatial_entities.loc[index, "feature_code"] = res["raw_data"]["fcode"]
        df_sentence_spatial_entities.loc[index, "geonames_id"] = str(res["geonamesId"])
        df_sentence_spatial_entities.loc[index, "updated"] = 1
        df_sentence_spatial_entities.loc[index, "prev_geonames_json"] = row["geonames_json"]
        df_sentence_spatial_entities.loc[index, "country_code"] = res["country_code"]
        df_sentence_spatial_entities.loc[index, "value"] = res["name"]
        df_sentence_spatial_entities.loc[index, "geonames_json"] = json.dumps(res["raw_data"], ensure_ascii=False)
  
    # print("df_sentence_spatial_entities")
    # print(df_sentence_spatial_entities)
    # print(df_sentence_spatial_entities["updated"])
    # PCLS: semi-independent political entity >> Hong Kong and Macao
    df_sentence_country = df_sentence_spatial_entities[(df_sentence_spatial_entities["feature_code"] == "PCLI")\
                                                        | (df_sentence_spatial_entities["feature_code"] == "PCLS")\
                                                        | (df_sentence_spatial_entities["updated"] == 1)]
    df_sentence_other = df_sentence_spatial_entities[(df_sentence_spatial_entities["feature_code"] != "PCLI")\
                                                      & (df_sentence_spatial_entities["feature_code"] != "PCLS")\
                                                       & (df_sentence_spatial_entities["feature_code"] != "CONT")\
                                                       & (df_sentence_spatial_entities["updated"] == 0)]
    df_sentence_country_copy = df_sentence_country.copy()
    # print(df_sentence_spatial_entities["feature_code"])
    # print("df_sentence_country")
    # print(df_sentence_country)
    # print("df_sentence_other")
    # print(df_sentence_other)
    
   # ---------------------------------------
    # second, resolve the other (state, city) disambiguity     
    #print("!!!!!!!!!!!!!! process_spatial_entities_in_sentence => local sentence id: ", local_sentence_id)
    #print("nb countries: ", df_sentence_country.shape[0])
    print("nb others: ", df_sentence_other.shape[0])
    print(df_sentence_other[["geonames_id", "feature_code", "value"]])
    
    #print(sliding_window_for_history)
    prev_sentence_country_codes = list(np.unique([entry[1] for entry in sliding_window_for_history]))
    #print("prev_sentence_country_codes: ", prev_sentence_country_codes)
    ref_country_codes = prev_sentence_country_codes
    # print("ref_country_codes_from_title: ", ref_country_codes_from_title)
    ref_country_codes = list(np.unique([countries.get(val).alpha2 for val in ref_country_codes if val != '-1'])) # from alpha2 to alpha3
    # sentence_country_values = np.unique(df_sentence_country["value"].to_numpy())
    if df_sentence_country_copy.shape[0]>0:
      sentence_country_codes = list(np.unique(df_sentence_country_copy["geonames_json"].apply(lambda x: \
            json.loads(x)["countryCode"] if "countryCode" in json.loads(x) else json.loads(x)["country_code"]).to_numpy()))
      if len(sentence_country_codes)>0:
        ref_country_codes = list(np.unique(ref_country_codes + sentence_country_codes))
    #  
    print("ref_country_codes", ref_country_codes)
    #print(df_sentence_country_copy["value"])
    #print(df_sentence_other["country_code"])
    if len(ref_country_codes) == 0:
      if len(df_sentence_other["country_code"].to_list())>1 and len(np.unique(df_sentence_other["country_code"].to_list()))==1: # if there ar least 2 spatial entiites with the same country, 
        ref_country_code = np.unique(df_sentence_other["country_code"].to_list())[0]
        ref_country_codes.append(countries.get(ref_country_code).alpha2)
      else:
        df_sentence_other = pd.DataFrame()
  
    #print(df_sentence_other)
    df_sentence_other_copy = df_sentence_other.copy()
    for index, row in df_sentence_other.iterrows():
      spatial_entity = row["value"].strip()
      
      process = False
      print(">> '"+spatial_entity.lower()+"'")
      if spatial_entity.lower().replace(",","").strip() not in consts.BAN_LIST_FOR_SPATIAL_ENTITY_EXTRACTION:
        spatial_entity = spatial_entity.replace(",","").strip()
        if "west nile" not in spatial_entity.lower():
          if not ((spatial_entity.lower() == "dc" and "IN" in ref_country_codes) or (spatial_entity.lower() == "cm" and "IN" in ref_country_codes)): # >> 'DC' stands for "Deputy commissioner", CM: Chief minister
            #if not (is_relative_spatial_entity(ent.text) and idenitfy_relative_spatial_cardinal(ent.text) == "-1"):
            # print(spatial_entity, idenitfy_relative_spatial_cardinal(spatial_entity))
            if not (idenitfy_relative_spatial_cardinal(spatial_entity) == "-1"):
              if spatial_entity != "" and len(spatial_entity)>1 and not spatial_entity.isnumeric():
                process = True
      
      #print("process:", process, spatial_entity)
      res = {"geonamesId": "-1", "name": "-1", "country_code": "-1", "raw_data": "-1"}
      if process:
        # ==================================================
        # retrieve the results from our database
        locs = None
        if spatial_entity in self.inverted_indices_for_df_batch_geonames_results:
          #print("DB ACCESS!!!!")
          gindx = self.inverted_indices_for_df_batch_geonames_results[spatial_entity]
          row_locs = self.df_batch_geonames_results.iloc[gindx]
          locs = [json.loads(row_locs[str(i)]) for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST)] # we access the columns
        # ==================================================
        init_geonamesId = str(row["geonames_id"])
        res = geocode_with_geonames(spatial_entity, ref_country_codes, locs)
        #print("after geocoding")
        #print(res)
        if spatial_entity not in self.inverted_indices_for_df_batch_geonames_results: # and res["geonamesId"] != '-1':  # update DB
          df_batch, inverted_indices = update_geocoding_results_in_DB(self.geonames_results_in_batch_filepath, spatial_entity)
          self.df_batch_geonames_results = df_batch
          self.inverted_indices_for_df_batch_geonames_results = inverted_indices
        # # ---------------------------------------------------
        # # If we do not get any results, and if there is only one ref country, then get this country as spatial entity
        # # e.g.spatial entity: "Sylvie" in id='0D2LGTCUQ0' ==> France
        # # if res["geonamesId"] == '-1' and len(ref_country_codes) == 1:
        # #   country_spatial_entity = countries.get(ref_country_codes[0]).apolitical_name
        # #   locs = None
        # #   if country_spatial_entity in self.inverted_indices_for_df_batch_geonames_results:
        # #     #print("DB ACCESS!!!!")
        # #     gindx = self.inverted_indices_for_df_batch_geonames_results[country_spatial_entity]
        # #     row_locs = self.df_batch_geonames_results.iloc[gindx]
        # #     locs = [json.loads(row_locs[str(i)]) for i in range(consts.MAX_NB_LOCS_PER_GEONAMES_REQUEST)] # we access the columns
        # #   res = geocode_with_geonames(country_spatial_entity, [], locs, None, True) # only country
        # #   #print(res)
        # #   if country_spatial_entity not in self.inverted_indices_for_df_batch_geonames_results: # and res["geonamesId"] != '-1': # update DB
        # #     df_batch, inverted_indices = self.update_geocoding_results_in_DB(self.out_csv_folder, country_spatial_entity)
        # #     self.df_batch_geonames_results = df_batch
        # #     self.inverted_indices_for_df_batch_geonames_results = inverted_indices
        # ---------------------------------------------------
        if "fcode" not in res["raw_data"]: # if feature code is not known from geonames, just omit it
          res = {"geonamesId": "-1", "name": "-1", "country_code": "-1", "raw_data": "-1"}
        res_geonamesId = str(res["geonamesId"])
        df_sentence_other_copy.loc[index, "ref_countries_for_geocoding"] = ", ".join(ref_country_codes)
        if res_geonamesId != '-1' and init_geonamesId != res_geonamesId:
          #print("---------------------- article_id: ", article_id, ", local_sentence_id: ", local_sentence_id)
          #print(spatial_entity, " => ", init_geonamesId, " / ", res_geonamesId)
          #print(spatial_entity, res["raw_data"])
          df_sentence_other_copy.loc[index, "feature_code"] = res["raw_data"]["fcode"]
          df_sentence_other_copy.loc[index, "geonames_id"] = str(res_geonamesId)
          df_sentence_other_copy.loc[index, "updated"] = 1
          df_sentence_other_copy.loc[index, "prev_geonames_json"] = row["geonames_json"]
        
      df_sentence_other_copy.loc[index, "country_code"] = res["country_code"]
      df_sentence_other_copy.loc[index, "value"] = res["name"]
      df_sentence_other_copy.loc[index, "geonames_json"] = json.dumps(res["raw_data"], ensure_ascii=False)
        
    if df_sentence_other_copy.shape[0]>0:
      df_sentence_other_copy = df_sentence_other_copy[df_sentence_other_copy["value"] != "-1"]
    if df_sentence_country_copy.shape[0]>0:
      df_sentence_country_copy = df_sentence_country_copy[df_sentence_country_copy["value"] != "-1"]
    df_new_sentence_spatial_entities = pd.concat([df_sentence_country_copy.reset_index(), df_sentence_other_copy.reset_index()])
    
    df_new_sentence_spatial_entities = df_new_sentence_spatial_entities.drop_duplicates(subset=['position', 'geonames_id'])
    #print("final")
    #print(df_new_sentence_spatial_entities)
    return(df_new_sentence_spatial_entities)
    
    
    
  def extract_geonames_hierarchy_info_for_all_spatial_entities(self, extr_spatial_entities_info_filepath, geonames_data_with_hierarchy_filepath):
    df_info_extr = pd.read_csv(extr_spatial_entities_info_filepath, sep=";", keep_default_na=False)
    geonames_id_list = df_info_extr[consts.COL_GEONAMES_ID].to_list()
    geonames_id_list = [int(a) for a in geonames_id_list]
    unique_geonames_id_list = np.unique(geonames_id_list)
    enhance_geonames_data_with_hierarchy(unique_geonames_id_list, geonames_data_with_hierarchy_filepath)    
    
    
    
    
    

class NormalizationEmpresi(AbstractNormalization):

  def __init__(self):
    # input
    self.input_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_DB_EMPRESS_I
    self.input_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_DB_EMPRESS_I
    self.input_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_DB_EMPRESS_I
    # output
    self.output_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_DB_EMPRESS_I
    self.output_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_DB_EMPRESS_I
    self.output_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_DB_EMPRESS_I
    
    
  # Normalize the event entities
  def perform_normalization(self, disease_name, in_preprocessing_folder, out_normalization_folder, data_folder):
    self.disease_name = disease_name
    self.in_preprocessing_folder = in_preprocessing_folder
    self.out_norm_result_folder = out_normalization_folder
    self.data_folder = data_folder
    
    #self.df_raw_events = self.df_raw_events[self.df_raw_events["id"].isin([8708233, 8708303])]
    #self.df_raw_events = self.df_raw_events[self.df_raw_events["id"].isin(["ob_97434", "ob_119023", "ob_104158", "ob_103751", "ob_107239", "ob_104520", "ob_118647", "ob_117835", "ob_117834", "ob_117584", "ob_111890", "ob_111889"])]
    #self.df_raw_events = self.df_raw_events[self.df_raw_events["id"].isin([8699389, 8700432, 8703889])]

    # ===========================================================================================
    # PART 1: SPATIAL ENTITIES
    # ===========================================================================================
    preprocessed_spatial_entities_filepath = os.path.join(self.in_preprocessing_folder, self.input_preprocessed_spatial_entities_filename + "." + consts.FILE_FORMAT_CSV)
    df_preprocessed_spatial_entities = pd.read_csv(preprocessed_spatial_entities_filepath, sep=";", keep_default_na=False)

    search_geonames_for_admin1_loc_list(self.out_norm_result_folder, df_preprocessed_spatial_entities)
    loc_list = retrieve_normalized_loc_list(self.out_norm_result_folder, df_preprocessed_spatial_entities)

    name_list = [loc.name for loc in loc_list]
    geoname_id_list = [loc.geoname_id for loc in loc_list]
    geoname_json_list = [json.dumps(loc.geoname_json) for loc in loc_list]
    lat_list = [loc.lat for loc in loc_list]
    lng_list = [loc.lng for loc in loc_list]
    country_code_list = [loc.country_code for loc in loc_list]
    continent_list = [loc.continent for loc in loc_list]
    hierarchy_data_list = [loc.hierarchy_data for loc in loc_list]

    article_id_list = df_preprocessed_spatial_entities[consts.COL_ARTICLE_ID].to_list()


    df_norm_spatial_entity_info = pd.DataFrame({
      consts.COL_ARTICLE_ID: article_id_list,
       consts.KEY_GEONAMES_NAME: name_list,
       consts.COL_GEONAMES_ID: geoname_id_list,
       consts.COL_GEONAMS_JSON: geoname_json_list,
       consts.COL_LAT: lat_list,
       consts.COL_LNG: lng_list,
       consts.COL_COUNTRY_ALPHA3_CODE: country_code_list,
       consts.COL_LOC_CONTINENT: continent_list,
       consts.COL_GEONAMES_HIERARCHY_INFO: hierarchy_data_list
    })

    # remove the spatial entities which are not geocoded
    df_norm_spatial_entity_info = df_norm_spatial_entity_info[df_norm_spatial_entity_info[consts.COL_GEONAMES_ID] != -1]

    # write the results into file
    norm_spatial_entity_info_filepath = os.path.join(self.out_norm_result_folder, self.output_norm_spatial_entities_filename + "." + consts.FILE_FORMAT_CSV)
    df_norm_spatial_entity_info.to_csv(norm_spatial_entity_info_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    

    # ===========================================================================================
    # PART 2: HOST ENTITIES 
    # ===========================================================================================
    preprocessed_host_entities_filepath = os.path.join(self.in_preprocessing_folder, self.input_preprocessed_host_entities_filename + "." + consts.FILE_FORMAT_CSV)
    df_preprocessed_host_entities = pd.read_csv(preprocessed_host_entities_filepath, sep=";", keep_default_na=False)
    article_id_list = df_preprocessed_host_entities[consts.COL_ARTICLE_ID].to_list()
    host_info_list = df_preprocessed_host_entities[consts.COL_HOST].to_list()
    host_info_list = [eval(h) for h in host_info_list]
    
    NCBI_host_results_filepath = os.path.join(self.out_norm_result_folder, "NCBI_results_in_batch" + "." + consts.FILE_FORMAT_CSV)
    norm_hosts_list = retrieve_normalized_host_list(host_info_list, NCBI_host_results_filepath)
    
    df_norm_host_entity_info = pd.DataFrame({
       consts.COL_ARTICLE_ID: article_id_list,
       consts.COL_HOST_ENTRY: norm_hosts_list
    })
    
    # TODO remove the host entities which are not normalized
    #df_norm_host_entity_info = df_norm_host_entity_info[df_norm_host_entity_info[consts.COL_HOST_ENTRY] != {}] # empty dict
    
    norm_host_entity_info_filepath = os.path.join(self.out_norm_result_folder, self.output_norm_host_entities_filename + "." + consts.FILE_FORMAT_CSV) 
    df_norm_host_entity_info.to_csv(norm_host_entity_info_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
    
    
    # ===========================================================================================
    # PART 3: DISEASE ENTITIES 
    # ===========================================================================================
    preprocessed_disease_entities_filepath = os.path.join(self.in_preprocessing_folder, self.input_preprocessed_disease_entities_filename + "." + consts.FILE_FORMAT_CSV)
    df_preprocessed_disease_entities = pd.read_csv(preprocessed_disease_entities_filepath, sep=";", keep_default_na=False)
    article_id_list = df_preprocessed_disease_entities[consts.COL_ARTICLE_ID].to_list()
    disease_info_list = df_preprocessed_disease_entities[consts.COL_DISEASE].to_list()
    disease_info_list = [eval(h) for h in disease_info_list]
    
    norm_disease_list = retrieve_normalized_disease_list(disease_info_list, self.disease_name)
    disease_hierachy_list = [d.get_hierarchy_as_list() for d in norm_disease_list]
    disease_data_list =[d.get_disease_data() for d in norm_disease_list]
    
    df_norm_disease_entity_info = pd.DataFrame({
       consts.COL_ARTICLE_ID: article_id_list,
       consts.COL_DISEASE: self.disease_name,
       consts.COL_DISEASE_ENTRY: disease_data_list,
       consts.COL_DISEASE_HIERARCHY: disease_hierachy_list
    })
    
    # TODO remove the disease entities which are not normalized
    #df_norm_disease_entity_info = df_norm_disease_entity_info[df_norm_disease_entity_info[consts.COL_DISEASE_HIERARCHY] != []] # empty list
    
    norm_disease_entity_info_filepath = os.path.join(self.out_norm_result_folder, self.output_norm_disease_entities_filename + "." + consts.FILE_FORMAT_CSV) 
    df_norm_disease_entity_info.to_csv(norm_disease_entity_info_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
       
        

    



class NormalizationWahis(NormalizationEmpresi):

  def __init__(self):
    # input
    self.input_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_DB_WAHIS
    self.input_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_DB_WAHIS
    self.input_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_DB_WAHIS
    # output
    self.output_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_DB_WAHIS
    self.output_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_DB_WAHIS
    self.output_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_DB_WAHIS
    
    
  def perform_normalization(self, disease_name, in_preprocessing_folder, out_normalization_folder, data_folder):
    super().perform_normalization(disease_name, in_preprocessing_folder, out_normalization_folder, data_folder)



class NormalizationPromed(NormalizationEmpresi):

  def __init__(self):
    # input
    self.input_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_SURVEILLANCE_PROMED
    self.input_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_SURVEILLANCE_PROMED
    self.input_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_SURVEILLANCE_PROMED
    # output
    self.output_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_SURVEILLANCE_PROMED
    self.output_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_SURVEILLANCE_PROMED
    self.output_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_SURVEILLANCE_PROMED
    
    
  def perform_normalization(self, disease_name, in_preprocessing_folder, out_normalization_folder, data_folder):
    super().perform_normalization(disease_name, in_preprocessing_folder, out_normalization_folder, data_folder)
  


class NormalizationAphis(NormalizationEmpresi):

  def __init__(self):
    # input
    self.input_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_DB_APHIS
    self.input_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_DB_APHIS
    self.input_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_DB_APHIS
    # output
    self.output_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_DB_APHIS
    self.output_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_DB_APHIS
    self.output_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_DB_APHIS
    
    
  def perform_normalization(self, disease_name, in_preprocessing_folder, out_normalization_folder, data_folder):
    super().perform_normalization(disease_name, in_preprocessing_folder, out_normalization_folder, data_folder)
  
  
  
class NormalizationApha(NormalizationEmpresi):

  def __init__(self):
    # input
    self.input_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_"+consts.NEWS_DB_APHA
    self.input_preprocessed_disease_entities_filename = "preprocessed_disease_entities_"+consts.NEWS_DB_APHA
    self.input_preprocessed_host_entities_filename = "preprocessed_host_entities_"+consts.NEWS_DB_APHA
    # output
    self.output_norm_spatial_entities_filename = "norm_spatial_entities_"+consts.NEWS_DB_APHA
    self.output_norm_disease_entities_filename = "norm_disease_entities_"+consts.NEWS_DB_APHA
    self.output_norm_host_entities_filename = "norm_host_entities_"+consts.NEWS_DB_APHA
    
    
  def perform_normalization(self, disease_name, in_preprocessing_folder, out_normalization_folder, data_folder):
    super().perform_normalization(disease_name, in_preprocessing_folder, out_normalization_folder, data_folder)



class NormalizationBVBRC(NormalizationEmpresi):

  def __init__(self):
    # input
    self.input_preprocessed_spatial_entities_filename = "preprocessed_spatial_entities_" + consts.NEWS_DB_BVBRC
    self.input_preprocessed_disease_entities_filename = "preprocessed_disease_entities_" + consts.NEWS_DB_BVBRC
    self.input_preprocessed_host_entities_filename = "preprocessed_host_entities_" + consts.NEWS_DB_BVBRC
    # output
    self.output_norm_spatial_entities_filename = "norm_spatial_entities_" + consts.NEWS_DB_BVBRC
    self.output_norm_disease_entities_filename = "norm_disease_entities_" + consts.NEWS_DB_BVBRC
    self.output_norm_host_entities_filename = "norm_host_entities_" + consts.NEWS_DB_BVBRC

  def perform_normalization(self, disease_name, in_preprocessing_folder, out_normalization_folder, data_folder):
    super().perform_normalization(disease_name, in_preprocessing_folder, out_normalization_folder, data_folder)

