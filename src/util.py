'''
Created on Nov 16, 2021

@author: nejat
'''


import numpy as np
import pandas as pd
import datetime




def retrieve_time_related_info(dates):
  # we apply a trick for getting week numbers.
  # for the last days of december, we can get sometimes week 1, but this is a bit confusing.
  # information: 28/12/YEAR is always the last week of a given year
  # workaround: we limit the dates up to 28/12/YEAR in order not to have the issue of "week 1"
  dates_adj = [datetime.datetime(d.year, d.month, 28) if d.month == 12 and d.day>28 else d for d in dates]
  dates_adj = [datetime.datetime(d.year, d.month, 4) if d.month == 1 and d.day<4 else d for d in dates]
  
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


        
