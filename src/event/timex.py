
import datetime

#
# dates = ['2018-07-09',
#          '2018-W15',
#          '2018-02',
#          '2018-04-06',
#          '2018-W15',
#          '2018-02',
#          '2015-09',
#          '2018-09-27 INTERSECT P5D',
#          'FUTURE_REF',
#          'FUTURE_REF',
#          'PXY',
#          'THIS P1D INTERSECT 2018-09-28',
#          {'end': 'XXXX-06', 'begin': 'XXXX-04'},
#          '2014-03-19',
#          '2018-08-02']

FORMAT = '%Y-%m-%d'
def get_simple_date(item, strformat=FORMAT):
  try:
    if not "-" in item and item.isnumeric():
      item = item+"-01-01"
    return (True, datetime.datetime.strptime(item, strformat))
  except (ValueError, TypeError):
    return (False, item)

def get_from_split(is_resolved, item):
  if is_resolved:
    return (is_resolved, item)
  try:
    tokens = item.split(' ')
    are_resolved, items = zip(*(get_simple_date(token) for token in tokens))
    if any(are_resolved):
      # assume one valid token
      result, = (item for item in items if isinstance(item, datetime.datetime))
      return (True, result)
  except (ValueError, AttributeError):
    pass
  return (False, item)

def get_from_no_day(is_resolved, item):
  if is_resolved:
      return (is_resolved, item)
  if not 'W' in item:
    try:
      return (True, datetime.datetime.strptime(f'{item}-01', FORMAT))
    except ValueError:
      pass
  return (False, item)

def get_from_w_date(is_resolved, item):
  if is_resolved:
    return (is_resolved, item)
  if 'W' in item:
    try:
      return (True, datetime.datetime.strptime(f'{item}-1', "%Y-W%W-%w"))
    except ValueError:
      pass
  return (is_resolved, item)



def convert_timex_str_to_datetime(dates):
  collection1 = (get_simple_date(item) for item in dates)
  collection2 = (get_from_split(*args) for args in collection1)
  collection3 = (get_from_no_day(*args) for args in collection2)
  collection4 = (get_from_w_date(*args) for args in collection3)
  return([d for is_resolved, d in collection4 if is_resolved])

