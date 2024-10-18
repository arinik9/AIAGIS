'''
Created on Nov 16, 2021

@author: nejat
'''

import numpy as np

from math import radians, cos, sin, asin, sqrt
import pandas as pd

import datetime

from datetime import date, datetime
import calendar


def unnest_dict(d, keys=[]):
  # input: 'a': {'b': [1, 2, 5], 'c': [4, 6, 7]}, 'r': {'d': [11, 12, 13], 'e': [66]}}
  # output: [('a', 'b', [1, 2, 5]), ('a', 'c', [4, 6, 7]), ('r', 'd', [11, 12, 13]), ('r', 'e', [66])]
  result = []
  for k, v in d.items():
      if isinstance(v, dict):
          result.extend(unnest_dict(v, keys + [k]))
      else:
          result.append(tuple(keys + [k, v]))
  return result


       
# https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
def haversine(lon1, lat1, lon2, lat2):
  """
  Calculate the great circle distance between two points 
  on the earth (specified in decimal degrees)
  """
  # convert decimal degrees to radians 
  lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
  # haversine formula 
  dlon = lon2 - lon1 
  dlat = lat2 - lat1 
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * asin(sqrt(a)) 
  # Radius of earth in kilometers is 6371
  km = 6371* c
  return km
  


def format_4digits(x):
  """This method formats the given number (either integer or float)
      by forcing to have 4 digits.
     
  :param x: number
  :type x: int or float
  """
  return ('%.4f' % x)


def which(values):
  """This program finds the indexes of the elements that are True.
  Logically, the type of 'values' has to be boolean.
     
  values: boolean values
  """
  for value in values:
      if not isinstance(value, (bool, np.bool_)):
          raise Exception('which() function expects only boolean values')
  indxs = [indx for indx, bool_elt in enumerate(values) if bool_elt]
  return indxs
  




def get_total_week_number_of_year(year):
  # source: https://stackoverflow.com/questions/29262859/the-number-of-calendar-weeks-in-a-year
  last_week = date(year, 12, 28)
  return last_week.isocalendar()[1]


def get_week_numbers_of_month(month_no, year):
  first_day = date(year, month_no, 4)
  last_day_no = calendar.monthrange(year, month_no)[1]
  last_day = date(year, month_no, last_day_no)
  return (first_day.isocalendar()[1], last_day.isocalendar()[1])
  
  
def prepare_all_periods_by_week(month_no_values, year_values):
  period_pairs = []
  for i in range(len(month_no_values)):
    month_no = month_no_values[i]
    year = year_values[i]
    week_no_begin, week_no_end = get_week_numbers_of_month(month_no, year)
    period_pairs.extend([(w_no,year) for w_no in range(week_no_begin, week_no_end+1)])
    
  sorted_period_pairs = sorted(list(set(period_pairs)), key=lambda p: (p[1],p[0]))
  periods = ["week="+str(pair[0])+"_year="+str(pair[1]) for pair in sorted_period_pairs]
  return periods






def retrieve_time_related_info(dates):
  # we apply a trick for getting week numbers.
  # for the last days of december, we can get sometimes week 1, but this is a bit confusing.
  # information: 28/12/YEAR is always the last week of a given year
  # workaround: we limit the dates up to 28/12/YEAR in order not to have the issue of "week 1"
  dates_adj = [datetime(d.year, d.month, 28) if d.month == 12 and d.day>28 else d for d in dates]
  dates_adj = [datetime(d.year, d.month, 4) if d.month == 1 and d.day<4 else d for d in dates_adj]
  
  day_no_list = [d.day for d in dates]
  week_no_list = [d.isocalendar()[1] for d in dates_adj]
  biweek_no_list = [np.ceil(d.isocalendar()[1]/float(2)) for d in dates_adj] # keep float(2), otherwise, it wont work
  month_no_list = [d.month for d in dates]
  year_list = [d.year for d in dates]
  dates_int = [d.month*100 + d.day for d in dates]
  # source: https://stackoverflow.com/questions/60285557/extract-seasons-from-datetime-pandas
  season_list = pd.cut(dates_int,[0,321,620,922,1220,1300], labels=['winter','spring','summer','autumn','winter '])
  # due to the last element "winter ", apply this postprocessing
  season_list = [s.strip() for s in season_list]
  return day_no_list, week_no_list, biweek_no_list, month_no_list, year_list, season_list



        
