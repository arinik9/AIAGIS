'''
Created on Nov 14, 2021

@author: nejat
'''


class Symptom:

  def __init__(self):
      self.dict_data = {}
      
      
  def add_symptom_info(self, values, s_type):
    if isinstance(s_type, str) and isinstance(values, list) and len(values)>0:
      if s_type in self.dict_data:
        self.dict_data[s_type] = self.dict_data[s_type] + values
        self.dict_data[s_type] = list(set(self.dict_data[s_type]))
      else:
        self.dict_data[s_type] = values
    else:
      raise Exception('check the type of input parameters in add_host_info()') 
     
     
          
  def load_dict_data_from_str(self, str_values, str_type):
    self.dict_data = {}
    if str_type != "":
      values_list = [str_values]
      if ";" in str_values:
        values_list = str_values.split(";")
      type_list = [str_type]
      if ";" in str_type:
        type_list = str_type.split(";")
      for i in range(0,len(type_list)):
        self.dict_data[type_list[i]] = values_list[i].split("#")
      
      

  # it outputs in this form: [symptom_subtype, symptom]      
  def get_entry(self):
    if len(self.dict_data)>0:
      keys = self.dict_data.keys()
      vals = ""
      for idx, key in enumerate(keys):
        if idx>0:
          vals = vals + ":"
        vals = vals + "#".join(self.dict_data[key])
      return([vals, ":".join(keys)])
    return(["",""])
    
      
  # we stick to the host name in host 1. We just try to get extra information from host 2
  def extend_symptom_from_another_event_if_possible(self, sym2):
    if len(sym2.dict_data) > 0:
      sym2_keys = sym2.dict_data.keys()
      for idx, key in enumerate(sym2_keys):
        self.add_symptom_info(sym2.dict_data[key], key)
    #return(sym1)
  
  
  
  def __repr__(self):
    if len(self.dict_data)>0:
      keys = self.dict_data.keys()
      vals = ""
      for idx, key in enumerate(keys):
        if idx>0:
          vals = vals + ":"
        vals = vals + "#".join(self.dict_data[key])
      return "[" + vals + "," + ":".join(keys) + "]"
    return("[,]")
  

  def __str__(self):
    if len(self.dict_data)>0:
      keys = self.dict_data.keys()
      vals = ""
      for idx, key in enumerate(keys):
        if idx>0:
          vals = vals + ":"
        vals = vals + "#".join(self.dict_data[key])
        return "[" + vals + "," + ":".join(keys) + "]"
    return("[,]")
  
  