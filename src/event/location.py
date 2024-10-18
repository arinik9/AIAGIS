'''
Created on Nov 14, 2021

@author: nejat
'''

import json


class Location:

  def __init__(self, name, geoname_id, geoname_json, lat, lng, country_code, continent, hierarchy_data):
    self.name = name
    self.geoname_id = geoname_id
    self.geoname_json = geoname_json
    self.lat = lat
    self.lng = lng
    self.country_code = country_code
    self.continent = continent
    self.hierarchy_data = hierarchy_data # it is a list of geoname ids from country level to current level
    self.hierarchy_level = len(hierarchy_data)-1 # country level is level 0, ...
    

  def get_name(self):
    return self.name
  
  def get_geoname_id(self):
    return self.geoname_id
  
  def get_lat_lng(self):
    return self.lat, self.lng

  def get_country_code(self):
    return self.country_code

  def get_continent(self):
    return self.continent

  def get_hierarchy_level(self):
    return self.hierarchy_level
    
  def get_hierarchy_data(self):
    return self.hierarchy_data
  
      
  def get_entry(self):
    return([self.geoname_id, json.dumps(self.geoname_json), self.name, self.country_code, self.continent, self.lat, self.lng, json.dumps(self.hierarchy_data)])


  def is_spatially_included(self, another_loc):
    if self.country_code == another_loc.country_code:
      if another_loc.hierarchy_level < self.hierarchy_level: # e.g. level 1 < level 3
        if self.hierarchy_data[0:(another_loc.hierarchy_level+1)] == another_loc.hierarchy_data:
          return True
    return False
  

  def spatially_contains(self, another_loc):
    if self.country_code == another_loc.country_code:
      if self.hierarchy_level < another_loc.hierarchy_level: # e.g. level 1 < level 3
        if another_loc.hierarchy_data[0:(self.hierarchy_level+1)] == self.hierarchy_data:
          return True
    return False
  
  def is_identical(self, another_loc):
    if self.get_geoname_id() == another_loc.get_geoname_id():
      return True
    return False
  
  
  
  def __repr__(self):
    return "[" + str(self.geoname_id) + "," +  self.name + "," +  self.country_code + "," + str(self.continent) \
       + "," + str(self.lat) + "," + str(self.lng) +  "]"
       
  def __str__(self):
    return "[" + str(self.geoname_id) + "," +  str(self.name) + "," +  str(self.country_code) + "," + str(self.continent) \
       + "," + str(self.lat) + "," + str(self.lng) +  "]"
  
  
     