'''
Created on Nov 14, 2021

@author: nejat
'''


class Host:
  # this class is only for 1 host info

  # host_data is a dictionary, which represents a host info
  def __init__(self, host_data = None):
    self.host_data = {}
    self.hierarchy_level = 0
    if host_data is not None:
      print(host_data)
      # create a new dict object
      self.hierarchy_level = host_data["level"]
      self.host_data = {"id": host_data["id"], "type": host_data["type"], "common_name": host_data["common_name"], "level": host_data["level"], "hierarchy": host_data["hierarchy"]}
    
    
  def is_host_info_empty(self):
    return not bool(self.host_data)
     
        
  # each entry in 'hierarchy_data_as_list' is organized from general to specialized category
  def add_host_info(self, id, type, common_name, hierarchy_data_as_list):
    # ex: ("bird", "domestic bird", "black swan")
    # ex: ("bird", "domestic bird", "level2-unknown")
    
    if self.is_host_info_empty():
      nb_items = len(hierarchy_data_as_list)
      # # d_data = self.dict_data # we do not want to use the recursive approach, so we take advantage of shallow copy of the original dict structure
      # d_data = {}
      # for i in range(nb_items-1): # except the last item
      #   if hierarchy_data_as_list[i] not in d_data:
      #     if i < (nb_items-2):
      #       d_data[hierarchy_data_as_list[i]] = {}
      #     else:
      #       d_data[hierarchy_data_as_list[i]] = []
      #   d_data = d_data[hierarchy_data_as_list[i]]
      # if hierarchy_data_as_list[nb_items-1] not in d_data:
      #   d_data.append(hierarchy_data_as_list[nb_items-1]) # last item
      # #raise Exception('check the type of input parameters in add_host_info()') 
      # # update hier level
      #
      self.host_data = {"id": id, "type": type, "common_name": common_name, "level": nb_items, "hierarchy": hierarchy_data_as_list}
      self.hierarchy_level = nb_items
    
    
  def get_entry(self):
    return self.host_data # a dict
  
  
  def get_value_at_hierarchy_level(self, hierarchy_level):
    entry = self.get_entry()["hierarchy"]
    if hierarchy_level>0 and hierarchy_level <= len(entry):
      value = entry[hierarchy_level-1]
      return value
    return None
  
  
  # this function is not like "is_spatially_included" (i.e. the opposite logic)
  # here, we check if self.host is more general than another_host.host
  def is_hierarchically_included(self, another_host): # in other words, the self object is more generalized than 'another_host'
  # >>> h1
  # {'id': 9692, 'common_name': 'amur leopard', 'level': 12, 'hierarchy': ['mammal', 'theria', 'eutheria', 'boreoeutheria', 'laurasiatheria', 'carnivora', 'feliformia', 'felidae', 'pantherinae', 'panthera', 'panthera pardus', 'panthera pardus orientalis']}
  # >>> 
  # >>> h2
  # {'id': 9691, 'common_name': 'leopard', 'level': 11, 'hierarchy': ['mammal', 'theria', 'eutheria', 'boreoeutheria', 'laurasiatheria', 'carnivora', 'feliformia', 'felidae', 'pantherinae', 'panthera', 'panthera pardus']}
  # h2.is_hierarchically_included(h1) >> True
  
    if self.hierarchy_level < another_host.hierarchy_level: # e.g. self.disease is at level 1 and another_dis.disease is at level 3
      more_general_hierarchy_level = self.hierarchy_level
      #print("more_general_hierarchy_level", more_general_hierarchy_level)
      # we check if there is a common ancestor at "more_general_hierarchy_level"
      #print("another_dis", another_dis.get_value_at_hierarchy_level(more_general_hierarchy_level))
      #print("self", self.get_value_at_hierarchy_level(more_general_hierarchy_level))
      if another_host.get_value_at_hierarchy_level(more_general_hierarchy_level) == self.get_value_at_hierarchy_level(more_general_hierarchy_level):
        return True
    return False


  def hierarchically_includes(self, another_host):
    if another_host.hierarchy_level < self.hierarchy_level: # e.g. self.disease is at level 3 and another_dis.disease is at level 1
      more_general_hierarchy_level = another_host.hierarchy_level
      # we check if there is a common ancestor at "more_general_hierarchy_level"
      if another_host.get_value_at_hierarchy_level(more_general_hierarchy_level) == self.get_value_at_hierarchy_level(more_general_hierarchy_level):
        return True
    return False
  
  
  def has_same_ancestor(self, another_host): # for the entries with the same hierarchy_level value
    if self.hierarchy_level == another_host.hierarchy_level:
      ancestor_hierarchy_level = self.hierarchy_level-1
      if self.get_value_at_hierarchy_level(ancestor_hierarchy_level) == another_host.get_value_at_hierarchy_level(ancestor_hierarchy_level):
        return True
    return False

    
  # TODO, we can just check the host id
  def is_identical(self, another_host):
    if self.hierarchy_level != another_host.hierarchy_level:
      return(False)
    if self.hierarchy_level == another_host.hierarchy_level:
      for i in range(self.hierarchy_level):
        if another_host.host_data["hierarchy"][i] != self.host_data["hierarchy"][i]:
          return(False)
    return(True)

  
  
  def __repr__(self):
    return(str(self.host_data))


  def __str__(self):
    return(str(self.host_data))
  
   
 
  # def dict_depth(self, dic, level = 1):
  #   found_unknown = True
  #   if isinstance(dic, dict):
  #     for k in dic.keys():
  #       if "unknown" not in k:
  #         found_unknown = False
  #   if isinstance(dic, list):
  #     for k in dic:
  #       if "unknown" not in k:
  #         found_unknown = False
  #   if found_unknown:
  #     return level-1
  #   if not isinstance(dic, dict) or not dic or found_unknown:
  #     return level
  #   return max(self.dict_depth(dic[key], level + 1) for key in dic)
  #
  #
  # def get_max_hierarchy_level(self):
  #   return self.dict_depth(self.dict_data) 
  #
  #
  # def merge_with_other_dict_data(self, other_dict_data):
  #   def unnest(d, keys=[]):
  #     # input: d2 = {'bird': {'domestic bird': ['poultry', 'aaaa']}}
  #     # output: [('bird', 'domestic bird', ['poultry', 'aaaa'])]
  #     result = []
  #     for k, v in d.items():
  #         if isinstance(v, dict):
  #             result.extend(unnest(v, keys + [k]))
  #         else:
  #             result.append(tuple(keys + [k, v]))
  #     return result
  #   other_tuples = unnest(other_dict_data)
  #   final_other_tuples = []
  #   for other_tuple in other_tuples:
  #     for elt in other_tuple[-1]: # the last item is always a list
  #       final_other_tuples.append(tuple(list(other_tuple[0:-1])+[elt]))
  #   for other_tuple in other_tuples:
  #     self.add_host_info(other_tuple)
  #   # update hier level
  #   self.hierarchy_level = self.get_max_hierarchy_level()
  #   #self.nb_host_cases = other_dict_data.nb_host_cases + 1
      
      
  # def is_hierarchically_included(self, another_host):
  #   if list(self.dict_data.keys())[0] == list(another_host.dict_data.keys())[0]:
  #     if another_host.get_entry()["level"] < self.get_entry()["level"]: # e.g. level 1 < level 3
  #       return True
  #   return False
  #
  #
  # def hierarchically_includes(self, another_host):
  #   if list(self.dict_data.keys())[0] == list(another_host.dict_data.keys())[0]:
  #     if self.hierarchy_level < another_host.hierarchy_level: # e.g. level 1 < level 3
  #       return True
  #   return False
  #
  #
  # def is_identical(self, another_dis):
  #   def unnest(d, keys=[]):
  #     # input: d2 = {'bird': {'domestic bird': ['poultry', 'aaaa']}}
  #     # output: [('bird', 'domestic bird', ['poultry', 'aaaa'])]
  #     result = []
  #     for k, v in d.items():
  #         if isinstance(v, dict):
  #             result.extend(unnest(v, keys + [k]))
  #         else:
  #             result.append(tuple(keys + [k, v]))
  #     return result
  #   if self.hierarchy_level != another_dis.hierarchy_level:
  #     return False
  #   temp = unnest(self.get_entry())
  #   other_temp = unnest(another_dis.get_entry())
  #   if len(temp) != len(other_temp):
  #     return False
  #   for i in range(len(temp)):
  #     if temp[i] != other_temp[i]:                   
  #       return False
  #   return True 
  
  
