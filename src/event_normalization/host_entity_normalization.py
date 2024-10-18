'''
Created on Nov 16, 2021

@author: nejat
'''

import src.consts as consts
import src.user_consts as user_consts
import json
import os

from src.event.host import Host
from src.event.hosts import Hosts

import pandas as pd
import csv

from src.event_preprocessing.pattern_en import pluralize, singularize # https://www.geeksforgeeks.org/python-program-to-convert-singular-to-plural/





def retrieve_normalized_host_list(host_info_list, NCBI_host_results_filepath):
  print(NCBI_host_results_filepath)
  if not os.path.exists(NCBI_host_results_filepath):
    # create an empty file 
    columns = ["result", "text"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(NCBI_host_results_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
  df_batch_NCBI_results = pd.read_csv(NCBI_host_results_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_NCBI_results = dict(zip(df_batch_NCBI_results["text"], df_batch_NCBI_results.index))
  
  hosts_list = []
  for host_info in host_info_list:
    print(host_info)
    host_list = []
    for host_txt in host_info:
      if host_txt in inverted_indices_for_df_batch_NCBI_results:
        host_indx = inverted_indices_for_df_batch_NCBI_results[host_txt]
        row_host_result = df_batch_NCBI_results.iloc[host_indx]
        host_entry = json.loads(row_host_result["result"])
        host = Host(host_entry)
        host_list.append(host)
      else:
        NCBI_id = retrieve_NCBI_id_from_keyword(host_txt)
        common_name = consts.NCBIid2commonName[NCBI_id]
        host_type, hierarchy = retrieve_NCBI_hierarchy_and_host_type_from_id(NCBI_id)
        host = Host()
        host.add_host_info(NCBI_id, host_type, common_name, hierarchy)
        host_list.append(host)
        #
        df_batch, inverted_indices = update_host_results_in_DB(NCBI_host_results_filepath, host_txt, host.get_entry())
        df_batch_NCBI_results = df_batch
        inverted_indices_for_df_batch_NCBI_results = inverted_indices
    hosts = Hosts(host_list)
    hosts_list.append(hosts)
  return hosts_list



def retrieve_NCBI_id_from_keyword(txt):
  txt_orig = txt 
  # ----------------------------------------------------------------------------------------------
  # workaround for human-related texts, since NCBI does not contain any
  human_db = [entry["text"] for entry in consts.HOST_KEYWORDS_HIERARCHY_DICT[user_consts.HOST_HUMAN]]
  if txt in human_db:
    txt = user_consts.HOST_HUMAN
  # ----------------------------------------------------------------------------------------------
  txt_pl = pluralize(txt)
  txt_sl = singularize(txt)
  taxId = None
  if txt_pl in consts.hostName2NCBIid:
    taxId = consts.hostName2NCBIid[txt_pl]
  elif txt in consts.hostName2NCBIid:
    taxId = consts.hostName2NCBIid[txt]
  elif txt_sl in consts.hostName2NCBIid:
    taxId = consts.hostName2NCBIid[txt_sl]
    
  if taxId is None:
    print("error for", txt_orig, "in retrieve_NCBI_id_from_keyword()")
  return taxId


# source: https://stackoverflow.com/questions/163542/how-do-i-pass-a-string-into-subprocess-popen-using-the-stdin-argument
# source: https://bioinf.shenwei.me/taxonkit/usage
def retrieve_NCBI_hierarchy_and_host_type_from_id(id):
  
  # common_name = kw
  # if search_common_name:
  #   common_name = consts.NCBIid2commonName[NCBI_id]
                      
  from subprocess import Popen, PIPE, STDOUT
  p = Popen(['/opt/homebrew/bin/taxonkit', 'lineage'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
  stdout = p.communicate(input=str.encode(str(id)))[0].decode()
  if "not found" in stdout or "None" in stdout:
    return []
  txt = stdout.split("cellular organisms;")[1].lower()
  
  host_type = ""
  if "homo sapiens" in txt:
    host_type = "human"
  elif "mammalia" in txt:
    host_type = "mammal"
  elif "aves" in txt:
    host_type = "avian"
    
  parts = txt.replace("\n","").split(";")
  
  return(host_type, parts)


def update_host_results_in_DB(host_results_filepath, sentence_text, result):
  result_str = json.dumps(result, ensure_ascii=False)
  #print("in update_NCBI_host_results_in_DB()")
  df_batch_results = pd.read_csv(host_results_filepath, sep=";", keep_default_na=False)
  results = {"result": [], "text": []} # each key corresponds to a column in the future file
  results["text"].append(sentence_text)
  #print(sentence_text, " ==> ", result_str)
  results["result"].append(result_str)
  df_curr = pd.DataFrame(results)
  df_batch_results = df_batch_results.append(df_curr, ignore_index=True)
  df_batch_results.to_csv(host_results_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    
  df_batch_NCBI_results = pd.read_csv(host_results_filepath, sep=";", keep_default_na=False)
  inverted_indices_for_df_batch_NCBI_results = dict(zip(df_batch_NCBI_results["text"], df_batch_NCBI_results.index))
  return(df_batch_NCBI_results, inverted_indices_for_df_batch_NCBI_results)

