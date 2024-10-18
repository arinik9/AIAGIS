import itertools
from src.event.location import Location
from src.event.temporality import Temporality
from src.event.host import Host
from src.event.disease import Disease
from src.event.symptom import Symptom

import json

class Event:
  newid = itertools.count()

  def __init__(self, e_id, article_id, url, source, loc, t, \
                dis, h, sym, title, sentences):
    if e_id != -1:
      self.e_id = e_id
    else:
      self.e_id = next(self.newid)
    
    if not isinstance(article_id, int):
    # TypeError("Only integers are allowed.")
      self.article_id = str(article_id)
    else:
      self.article_id = article_id
    
    if not isinstance(url, str):
      TypeError("Only strings are allowed.")
    self.url = url
    
    if not isinstance(source, str):
      TypeError("Only strings are allowed.")
    self.source = source
    
    if not isinstance(title, str):
      TypeError("Only strings are allowed.")
    self.title = title
    
    if not isinstance(sentences, str):
      TypeError("Only strings are allowed.")
    self.sentences = sentences
    
    if not isinstance(loc, Location):
      TypeError("Only objects of type 'Location' are allowed.")
    self.loc = loc
    
    if not isinstance(t, Temporality):
      TypeError("Only objects of type 'Temporality' are allowed.")
    self.date = t
    
    if not isinstance(dis, Disease):
      TypeError("Only objects of type 'Disease' are allowed.")
    self.disease = dis
    
    if not isinstance(h, Host):
      TypeError("Only objects of type 'Host' are allowed.")
    self.host = h
    
    if not isinstance(sym, Symptom):
      TypeError("Only objects of type 'Symptom' are allowed.")
    self.symptom = sym  
    
    


  def get_event_entry(self):
    event_entry = [str(self.e_id), str(self.article_id), self.url, self.source]
    print(self.loc.get_entry())
    print(self.date.__str__())
    print(self.disease.get_hierarchy_as_list())
    host_entry = self.host.get_entry()
    event_entry = event_entry + self.loc.get_entry() + [self.date.__str__()] + [str(self.disease.__str__())] \
                 +  [str(host_entry)] + self.symptom.get_entry()
    return(event_entry)
  
  
  
  def is_identical(self, e):
    if self.loc.is_identical(e.loc) and self.date == e.date and self.disease.is_identical(e.disease) and self.host.is_identical(e.host):
      return(True)
    return(False)
  

  def __repr__(self):
    return "<Event id: %s, artice id: %s, source: %s, loc: %s, date: %s, disease: %s, host: %s, symptom: %s>" \
       % (str(self.e_id), self.article_id, self.source, self.loc, self.date, self.disease, self.host, self.symptom)
       
  def __str__(self):
    return "<Event id: %s, artice id: %s, source: %s, loc: %s, date: %s, disease: %s, host: %s, symptom: %s>" \
       % (str(self.e_id), self.article_id, self.source, self.loc, self.date, self.disease, self.host, self.symptom)
