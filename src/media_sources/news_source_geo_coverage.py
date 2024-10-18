'''
Created on Dec 15, 2021

@author: nejat
'''
import os
import pandas as pd
import src.consts as consts

from tldextract import extract
from iso3166 import countries


# http://la1ere.francetvinfo.fr/wallisfutuna/
# http://la1ere.francetvinfo.fr/guadeloupe/
# http://la1ere.francetvinfo.fr/martinique/
# http://www.lepetitjournal.com/rio-de-janeiro
# http://www.lepetitjournal.com/berlin
# http://www.lepetitjournal.com/athenes
# http://www.lepetitjournal.com/budapest
def trim_url_custom2(url):
  http_count = url.count("http")
  if http_count > 0:
    url_part_by_http = url.split("http")
    part_of_interest_url = url_part_by_http[http_count]
    part_of_interest = part_of_interest_url.split("//")[1]
    twitter_count = part_of_interest.count("twitter")
    if twitter_count>0:
      # https://twitter.com/FluTrackers/statuses/1077629051630702593
      if "/status" in part_of_interest:
        part_of_interest = part_of_interest.split("/status")[0]
      else:
        part_of_interest = part_of_interest.split("/statuses")[0]
    # ---------------------------
    tsd, td, tsu = extract(part_of_interest)
    trim_url = td + "." + tsu
    if td not in ["maville"]:
      if tsd != "":
        trim_url = tsd + "." +  td + "." + tsu
      if td in ["francetvinfo", "lepetitjournal", "theguardian"]:
        part2 = url.split(trim_url+"/")[1]
        trim_url = trim_url + "/" +part2.split("/")[0]
    
    #print(url, trim_url)
    return trim_url
  # print(url, "-1")
  return "-1"   


def retrieve_country_code_from_url_public_suffix(url):
  # for more info: https://en.wikipedia.org/wiki/Country_code_top-level_domain
  tsd, td, tsu = extract(url)
  tsu = tsu.replace("co.uk", "gb")
  tsu = tsu.replace("co.", "")
  tsu = tsu.replace("com.", "")
  tsu = tsu.replace("org.", "")
  tsu = tsu.replace("or.", "")
  tsu = tsu.replace("gov.", "")
  tsu = tsu.replace("net.", "")
  if tsu == "com":
    return "-1"
  country_code = "-1"
  if tsu in countries:
    country_code = countries.get(tsu).alpha3
  return country_code
  

# we apply the same trimming mechanism for both url lists.
def perform_media_source_resolution(news_source_url_list, data_folder):
  source_url_to_pub_country_code = create_dict_url_to_pub_country_code(data_folder)
  
  news_source_url_list = ["http://"+url if "http" not in url else url for url in news_source_url_list]
  news_source_trim_url_list = [trim_url_custom2(url) for url in news_source_url_list]
  
  pub_country_code_list = []
  for news_url in news_source_trim_url_list:
    tsd, td, tsu = extract(news_url)
    news_url1 = td + "." + tsu
    news_url2 = tsd + "." + td + "." + tsu
      
    if news_url in ["www.cidrap.umn.edu"]: # ban list, or instance, we know that cidrap reports all outbreaks around the world
      pub_country_code_list.append("-1")
    else:
      pub_country_code = retrieve_country_code_from_url_public_suffix(news_url)
      if pub_country_code != "-1" and news_url1 not in ["tempo.co"]:
        pub_country_code_list.append(pub_country_code)
      else:
        if tsd == "" or tsd == "m": # m for mobile
          news_url2 = "www" + "." + td + "." + tsu
        if news_url1 in source_url_to_pub_country_code:
          pub_country_code_list.append(source_url_to_pub_country_code[news_url1])
        elif news_url2 in source_url_to_pub_country_code:
          pub_country_code_list.append(source_url_to_pub_country_code[news_url2])
        else:
          pub_country_code_list.append("-1")


  pub_country_name_list = [countries.get(c_code).name if c_code != "-1" else "-1" for c_code in pub_country_code_list] 

  df = pd.DataFrame(data={'url':news_source_url_list, 'new_url':news_source_trim_url_list, \
                          'pub_country_name': pub_country_name_list, 'pub_country_code': pub_country_code_list})
  return df



def create_dict_url_to_pub_country_code(data_folder):
  #country_code_filepath = os.path.join(consts.DATA_SOURCE_GEO_COVERAGE_FOLDER, "countries_codes_and_coordinates" + "." + consts.FILE_FORMAT_CSV)
  #cols_country_code = [consts.COL_COUNTRY, consts.COL_COUNTRY_ALPHA3_CODE]
  #df_country_code = pd.read_csv(country_code_filepath, usecols=cols_country_code, sep=",")
  #pub_country_iso_code_to_pub_country = dict(zip(df_country_code[consts.COL_COUNTRY_ALPHA3_CODE], df_country_code[consts.COL_COUNTRY]))
  
  geo_media_sources_filepath = os.path.join(data_folder, "news-websites-geography", "media_sources" + "." + consts.FILE_FORMAT_CSV)
  cols_geo_media_sources = [consts.COL_URL, consts.COL_PUB_COUNTRY]
  df_geo_media_sources = pd.read_csv(geo_media_sources_filepath, usecols=cols_geo_media_sources, sep=",")
  
  new_values = []
  for pub_country_iso_code in df_geo_media_sources[consts.COL_PUB_COUNTRY]:
    name = countries.get(pub_country_iso_code).name
    new_values.append(name)
  df_geo_media_sources["country_name"] = new_values
  
  df_geo_media_sources[consts.COL_URL].str.replace("urn://", "http://")
  df_geo_media_sources[consts.COL_URL] = df_geo_media_sources[consts.COL_URL].apply(lambda x: "http://"+x if "http" not in x else x)
  # trim
  df_geo_media_sources[consts.COL_URL] = df_geo_media_sources[consts.COL_URL].apply(trim_url_custom2)
  
  df_media_source2 = df_geo_media_sources.groupby([consts.COL_URL])[consts.COL_PUB_COUNTRY, "country_name"].aggregate(lambda x: list(x)[0])
  geo_media_sources_filepath = os.path.join(data_folder, "news-websites-geography", "media_sources_adj" + "." + consts.FILE_FORMAT_CSV)
  df_media_source2.to_csv(geo_media_sources_filepath, sep=",")
  
  source_url_to_pub_country_code = dict(zip(df_media_source2.reset_index()[consts.COL_URL], df_media_source2.reset_index()[consts.COL_PUB_COUNTRY]))
  #print(source_url_to_pub_country)
  return(source_url_to_pub_country_code)



  