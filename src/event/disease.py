'''
Created on Nov 14, 2021

@author: nejat
'''


class Disease:

# source: https://www.cdc.gov/flu/avianflu/influenza-a-virus-subtypes.htm
#
# There are a few possibilities:
#
# level 0: avian influenza
# level 1: h5 subtype
# level 2: h5n1
#
# level 0: avian influenza
# level 1: h7 subtype
# level 2: h7n5
#
# level 0: avian influenza
# level 1: other subtypes than h5 and h7
# level 2: h5n3
#
# level 0: avian influenza
# level 1: h5 subtype
# level 2: unknown serotype
#
# level 0: avian influenza
# level 1: h7 subtype
# level 2: unknown serotype
#
# level 0: avian influenza
# level 1: unknown subtype
# level 2: unknown serotype

  def __init__(self, d_serotype, d_subtype, d_type, d_pathogenicity=None):
    self.d_type = d_type.lower() # we suppose that this variable has a single value, so not a list of values
    self.d_subtype = d_subtype.lower() # we suppose that this variable has a single value, so not a list of values
    self.d_serotype = d_serotype.lower() # we suppose that this variable has a single value, so not a list of values
    self.pathogenicity = d_pathogenicity
    if self.pathogenicity is not None:
      self.pathogenicity = self.pathogenicity.lower()
    #
    self.hierarchy_level = 0
    if not self.is_entry_empty():
      self.hierarchy_level = 1 # by default
    if self.d_subtype != "" and "unknown" not in self.d_subtype:
      self.hierarchy_level = 2
    if self.d_serotype != "" and "unknown" not in self.d_serotype:
      self.hierarchy_level = 3
    #  
    if self.d_serotype == "":
      self.d_serotype = "unknown serotype"
    if self.d_subtype == "":
      self.d_subtype = "unknown subtype"
    #
    if self.pathogenicity is None:
      if self.get_max_hierarchy_level() == len(self.get_hierarchy_as_list()): # i.e. self.d_serotype is not empty
        if self.d_serotype in ["h5n8", "h5n6", "h7n9", "h5n1"]: # we estimate if it is HPAI or LPAI based on the provided serotype
          self.pathogenicity = "hpai"
        else:
          self.pathogenicity = "lpai"
      else:
        self.pathogenicity = "unknown-pathogenicity"
      
      
  def get_disease_data(self):
    return({"serotype": self.d_serotype, "subtype": self.d_subtype, "type": self.d_type, "pathogenicity": self.pathogenicity})


  def get_hierarchy_as_list(self):
    return([self.d_serotype, self.d_subtype, self.d_type])
  
  
  def is_entry_empty(self):
    if (self.d_type == "" or "unknown" in self.d_type) and (self.d_subtype == "" or "unknown" in self.d_subtype) and (self.d_serotype == "" or "unknown" in self.d_serotype):
      return True
    return False
  
  # we stick to the disease name in disease 1. We just try to get extra information from disease 2
  def extend_disease_from_another_event_if_possible(self, dis2):
    #dis1 = Disease(self.d_subtype, self.d_type)
    if self.d_type != "" and self.d_type == dis2.d_type:
      if self.is_hierarchically_included(dis2):
        self.init(dis2.d_serotype, dis2.d_subtype, dis2.d_type, dis2.d_pathogenicity)
    #return(dis1)
  
  
  
  # def get_max_hierarchy_level(self):
  #   if self.d_subtype != "":
  #     return 1
  #   return 0
  
  def get_max_hierarchy_level(self):
    return self.hierarchy_level
  
  def set_max_hierarchy_level(self, level):
    self.hierarchy_level = level
    
    
  def get_value_at_hierarchy_level(self, hierarchy_level):
    entry = self.get_hierarchy_as_list() # example: ["h5n1", "h5 or h7 subtypes", "avian influenza"] (more specific to more general)
    if hierarchy_level>0 and hierarchy_level <= len(entry):
      value = entry[-hierarchy_level]
      return value
    return None
  
  
  # this function is not like "is_spatially_included" (i.e. the opposite logic)
  # here, we check if self.disease is more general than another_dis.another_dis
  def is_hierarchically_included(self, another_dis):
    if self.hierarchy_level < another_dis.hierarchy_level: # e.g. self.disease is at level 1 and another_dis.disease is at level 3
      more_general_hierarchy_level = self.hierarchy_level
      #print("more_general_hierarchy_level", more_general_hierarchy_level)
      # we check if there is a common ancestor at "more_general_hierarchy_level"
      #print("another_dis", another_dis.get_value_at_hierarchy_level(more_general_hierarchy_level))
      #print("self", self.get_value_at_hierarchy_level(more_general_hierarchy_level))
      if another_dis.get_value_at_hierarchy_level(more_general_hierarchy_level) == self.get_value_at_hierarchy_level(more_general_hierarchy_level):
        return True
    return False


  def hierarchically_includes(self, another_dis):
    if another_dis.hierarchy_level < self.hierarchy_level: # e.g. self.disease is at level 3 and another_dis.disease is at level 1
      more_general_hierarchy_level = another_dis.hierarchy_level
      # we check if there is a common ancestor at "more_general_hierarchy_level"
      if another_dis.get_value_at_hierarchy_level(more_general_hierarchy_level) == self.get_value_at_hierarchy_level(more_general_hierarchy_level):
        return True
    return False
  
  
  def has_same_ancestor(self, another_dis): # for the entries with the same hierarchy_level value
    if self.hierarchy_level == another_dis.hierarchy_level:
      ancestor_hierarchy_level = self.hierarchy_level-1
      if self.get_value_at_hierarchy_level(ancestor_hierarchy_level) == another_dis.get_value_at_hierarchy_level(ancestor_hierarchy_level):
        return True
    return False

    
  def is_identical(self, another_dis):
    if self.hierarchy_level == another_dis.hierarchy_level:
      if self.d_type == another_dis.d_type and self.d_subtype == another_dis.d_subtype and self.d_serotype == another_dis.d_serotype:
        return True
    return False
  
  
  
  
  def __repr__(self):
    entry = self.get_hierarchy_as_list()
    entry[-1] = entry[-1] + " (" + self.pathogenicity + ")"
    text = "[" + "'"+entry[0]+"'"
    for item in entry[1:]:
      text = text + "," + "'"+item+"'"
    return text + "]"
       
  def __str__(self):
    entry = self.get_hierarchy_as_list()
    entry[-1] = entry[-1] + " (" + self.pathogenicity + ")"
    text = "[" + "'"+entry[0]+"'"
    for item in entry[1:]:
      text = text + "," + "'"+item+"'"
    return text + "]"
  
       
