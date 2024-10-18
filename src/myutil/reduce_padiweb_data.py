import os
import pandas as pd
import csv
import shutil

import src.consts as consts


#######################################################################
# This script aims to reduce the raw PADI-web input data based on a list 
#    of article ids.
# To my knowledge, it is not possible get the raw PADI-web input data 
#    for a specific host type only (e.g. mammals). What we do is to download
#    its data for a given period, then we process it with "epidnews2event"
#     to extract document events. Then, we can come back to its input data 
#     for reducing its size. It is more convenient to work with this small size.
#######################################################################


# ====================================
# STEP 1
# ====================================
events_filepath = os.path.join(consts.DOC_EVENTS_PADIWEB_FOLDER, "doc_events_padiweb_task1=relevant_sentence.csv")
df_events = pd.read_csv(events_filepath, sep=";", keep_default_na=False, index_col=0)
target_a_ids = df_events["article_id"].tolist()

articlesweb_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "articlesweb.csv")
df_raw_articles = pd.read_csv(articlesweb_filepath, sep=";", keep_default_na=False)

df = df_raw_articles[df_raw_articles["id"].isin(target_a_ids)]

articlesweb_red_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "articlesweb_reduced.csv")
df.to_csv(articlesweb_red_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)



# ====================================
# STEP 2
# ====================================
articlesweb_red_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "articlesweb_reduced.csv")
df_articles = pd.read_csv(articlesweb_red_filepath, sep=";", keep_default_na=False)
a_ids = df_articles["id"].tolist()

extr_info_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "extracted_information.csv")
df_extr_info = pd.read_csv(extr_info_filepath, sep=";", keep_default_na=False)

df = df_extr_info[df_extr_info["id_articleweb"].isin(a_ids)]
extr_info_red_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "extracted_information_reduced.csv")
df.to_csv(extr_info_red_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)




# ====================================
# STEP 3
# ====================================
articlesweb_red_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "articlesweb_reduced.csv")
df_articles = pd.read_csv(articlesweb_red_filepath, sep=";", keep_default_na=False)
a_ids = df_articles["id"].tolist()

sents_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "sentences_with_labels.csv")
df_sents = pd.read_csv(sents_filepath, sep=";", keep_default_na=False)

df = df_sents[df_sents["article_id"].isin(a_ids)]
sents_red_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "sentences_with_labels_reduced.csv")
df.to_csv(sents_red_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)



# ====================================
# STEP 4
# ====================================
extr_info_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "extracted_information.csv")
df_extr_info = pd.read_csv(extr_info_filepath, sep=";", keep_default_na=False)
kw_ids = df_extr_info["keyword_id"].tolist()
kw_ids = [int(kw_id) for kw_id in kw_ids if kw_id != "NULL"]

keyword_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "keyword.csv")
df_kw = pd.read_csv(keyword_filepath, sep=";", keep_default_na=False)
df = df_kw[df_kw["id"].isin(kw_ids)]

keyword_red_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "keyword_reduced.csv")
df.to_csv(keyword_red_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)


# ====================================
# STEP 5
# ====================================
articles_backup_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "articlesweb_backup.csv")
shutil.copyfile(articlesweb_filepath, articles_backup_filepath)
shutil.copyfile(articlesweb_red_filepath, articlesweb_filepath)
os.remove(articlesweb_red_filepath)

extr_info_backup_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "extracted_information_backup.csv")
shutil.copyfile(extr_info_filepath, extr_info_backup_filepath)
shutil.copyfile(extr_info_red_filepath, extr_info_filepath)
os.remove(extr_info_red_filepath)

sents_backup_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "sentences_with_labels_backup.csv")
shutil.copyfile(sents_filepath, sents_backup_filepath)
shutil.copyfile(sents_red_filepath, sents_filepath)
os.remove(sents_red_filepath)

keyword_backup_filepath = os.path.join(consts.IN_PADIWEB_FOLDER, "keyword_backup.csv")
shutil.copyfile(keyword_filepath, keyword_backup_filepath)
shutil.copyfile(keyword_red_filepath, keyword_filepath)
os.remove(keyword_red_filepath)

