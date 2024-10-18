'''
Created on Nov 14, 2021

@author: nejat
'''

from datetime import datetime
import dateutil.parser as parser


class Temporality:

  def __init__(self, t):
    if isinstance(t, datetime):
      self.date = t
    elif isinstance(t, str):
      # https://stackoverflow.com/questions/27800775/python-dateutil-parser-parse-parses-month-first-not-day
      # if it starts with year info
      if ("-" in t and len(t.split("-")[0])==4) or ("/" in t and len(t.split("/")[0])==4): 
        self.date = parser.parse(t, dayfirst=False)
      else: # if it starts with day info
        self.date = parser.parse(t, dayfirst=True)
    else:
      self.date = t
          
  def get_entry(self):
    return(self.date)

  
  def __repr__(self):
    #return "[" + self.get_entry().isoformat() + "]"
    return self.get_entry().isoformat()
       
  def __str__(self):
    #return "[" + self.get_entry().isoformat() + "]"
    return self.get_entry().isoformat()
 
