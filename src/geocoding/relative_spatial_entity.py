'''
Created on Aug 19, 2022

@author: nejat
'''


# REMARK:
#   The codes in this file are taken from the following paper:
#     Syed, M. A., Arsevska, E., Roche, M., and Teisseire, M.: GeoXTag: Relative Spatial 
#     Information Extraction and Tagging of Unstructured Text, AGILE GIScience Ser., 3, 16,
#     https://doi.org/10.5194/agile-giss-3-16-2022, 2022


import re 

def is_relative_spatial_entity(loc_text):
  regexCardinal = r"(?i)(north|east|south|center|centeral|west)"
  res = re.search(regexCardinal, loc_text)
  if res is None:
    return False
  return True


# relative spatial cardinals are: north, south, etc.
def idenitfy_relative_spatial_cardinal(spatial_entity_text):
  spatial_entity_text2 = spatial_entity_text.lower()
  regexCardinal = r"(?i)(north|nord|east|est|south|sud|center|centre|central|centeral|west|ouest|mid|ern|far|upper|lower)"
  res = re.sub(regexCardinal, "", spatial_entity_text2).strip()
  if res == "" or res == "-" or res == "ern":
      return "-1"
  return spatial_entity_text
