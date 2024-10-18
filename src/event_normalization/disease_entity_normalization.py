'''
Created on Nov 16, 2021

@author: nejat
'''

import src.consts as consts
import src.user_consts as user_consts
import re
from src.event.disease import Disease



  
def retrieve_normalized_disease_list(disease_info_list, disease_name):
  normalized_disease_list = []
  for disease_list in disease_info_list:
    final_dis = retrieve_disease_info_from_text(disease_name, disease_list[0])
    for text in disease_list[1:]:
      curr_dis = retrieve_disease_info_from_text(disease_name, text)
      if final_dis is not None and curr_dis is not None:
        if final_dis.is_hierarchically_included(curr_dis): # if final_dis is more detailed
          final_dis = curr_dis
    if final_dis is not None:
      normalized_disease_list.append(final_dis)
  return(normalized_disease_list)



def retrieve_disease_data_from_text(disease_name, text):
  disease = retrieve_disease_info_from_text(disease_name, text)
  # {"serotype": self.d_serotype, "subtype": self.d_subtype, "type": self.d_type, "pathogenicity": self.pathogenicity}
  if disease is None:
    return None
  return disease.get_disease_data()


def retrieve_disease_info_from_text(disease_name, text):
  dis = None
  
  # "hpai" and "lpai" are the first two elements in DISEASE_KEYWORDS_DICT >> intentionally
  disease_keywords = consts.DISEASE_KEYWORDS_DICT[disease_name].keys()
  #print(disease_keywords)
  dis_type = ""
  dis_pathogenicity = None
  dis_serotype = ""
  
  if disease_name == user_consts.DISEASE_AVIAN_INFLUENZA:
    if text in disease_keywords:
      dis_type = disease_name #consts.DISEASE_KEYWORDS_DICT[disease_name][kw]
  
      if text == "hpai":
        dis_pathogenicity = "hpai"
      elif text == "lpai":
        dis_pathogenicity = "lpai"
        
      dis = Disease(dis_serotype, "", dis_type, dis_pathogenicity)
    else:
      res = re.findall("h[0-9]{1,2}n[0-9]{1,2}", text)
      res2 = re.findall("h[0-9]{1,2}", text)
      res3 = re.findall("n[0-9]{1,2}", text)
      # an event must have a single AI subtype
      # for now, we do not treat complex sentences
      if len(res)>0:
        dis_serotype = text
        dis_type = disease_name
        if dis_serotype != "" or dis_type != "":
          d_subtype = dis_serotype.split("n")[0] # "h5" if "h5n5"
          dis = Disease(dis_serotype, d_subtype, dis_type, dis_pathogenicity)
      elif len(res2)>0 or len(res3)>0:
        d_subtype = text
        dis_type = disease_name
        dis_serotype = ""
        dis = Disease(dis_serotype, d_subtype, dis_type, dis_pathogenicity)
        #dis = str(dis)
  return(dis)

