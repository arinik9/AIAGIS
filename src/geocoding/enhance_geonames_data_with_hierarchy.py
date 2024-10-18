'''
Created on Aug 15, 2022

@author: nejat
'''


from random import randrange
import numpy as np
import csv
import pandas as pd

import src.consts as consts


import geocoder # >>> https://geocoder.readthedocs.io/providers/GeoNames.html

## place_name = 'Hôpitaux-Facultés, Montpellier'
## main = geocoder.geonames(place_name, key='arinik9')
## g = geocoder.geonames(main.geonames_id, method='hierarchy', key='arinik9', language='en')
## for result in g:
  ## print(result.address, result.geonames_id, result.latlng)

# g = geocoder.geonames(main.geonames_id, method='details', key='arinik9')
# for result in g:
#      print(result.raw)




def get_loc_hierarchy(geonames_id):
  print("geonames_id: ", geonames_id)
  # in some way, we could use this one, as well
  # g = geocoder.geonames(main.geonames_id, method='details', key='arinik9', language='en')
  res = {"name": [], "geoname_id": []}

  geonames_api_username_list = [consts.GEONAMES_API_USERNAME1, consts.GEONAMES_API_USERNAME2, \
                                consts.GEONAMES_API_USERNAME3, consts.GEONAMES_API_USERNAME4, \
                                consts.GEONAMES_API_USERNAME5, consts.GEONAMES_API_USERNAME6, \
                                consts.GEONAMES_API_USERNAME7]
  try:
    # there are some hourly and daily query limit for geonames servers.
    # as a workaround, we randomly pick one account and send our request. 
    # >> in theory, those accounts will be picked approx. uniformly
    rand_index = randrange(len(geonames_api_username_list))
    geonames_api_username = geonames_api_username_list[rand_index]
    print("---- account ", geonames_api_username)
    g = geocoder.geonames(geonames_id, method='hierarchy', key=geonames_api_username)
    if g.status == "OK":
      print("status: ", g.status)
      for result in g: # iterate the hierarchy from upper level to lower level
        if "countryName" in result.raw:
          res["name"].append(result.address)
          res["geoname_id"].append(result.geonames_id)
  except:
    print("error in get_loc_hierarchy() with geonames_id=", geonames_id)
    pass
  #print(res)
  return res


def retrieve_unique_geonames_id(filepath):
  df_info_extr = pd.read_csv(filepath, sep=";", keep_default_na=False)
  geonames_id_list = df_info_extr[consts.COL_GEONAMES_ID].to_numpy()
  unique_geonames_id_list = np.unique(geonames_id_list)
  print(2978317 in unique_geonames_id_list)
  return unique_geonames_id_list





def enhance_geonames_data_with_hierarchy(geonames_id_list, filepath):
  print("there are ", len(geonames_id_list), " geonames ids.")
  hierarchy_name_list = []
  hierarchy_geoname_id_list = []
  for geonames_id in geonames_id_list:
    hierarchy_results = get_loc_hierarchy(geonames_id)
    hierarchy_name_list.append(str(hierarchy_results["name"]))
    hierarchy_geoname_id_list.append(str(hierarchy_results["geoname_id"]))
  df = pd.DataFrame(data={consts.COL_GEONAMES_ID: geonames_id_list, "hierarchy_name": hierarchy_name_list, "hierarchy_geoname_id": hierarchy_geoname_id_list})

  df.to_csv(filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
#
#
#
# # if __name__ == '__main__':
# #   filepath = os.path.join(consts.CSV_PADIWEB_FOLDER, "extr_spatial_entities_info" + "." + consts.FILE_FORMAT_CSV)
# #   geonames_id_list = retrieve_unique_geonames_id(filepath)
# #   print(len(geonames_id_list))
# #   filepath = os.path.join(consts.CSV_PADIWEB_FOLDER, consts.GEONAMES_HIERARCHY_INFO_FILENAME + "." + consts.FILE_FORMAT_CSV)
# #   enhance_geonames_data_with_hierarchy(geonames_id_list)
#
#
