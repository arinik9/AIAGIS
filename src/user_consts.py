'''
Created on Nov 16, 2021

@author: nejat
'''

# ====================================================
# DISEASE NAMES (we define them here, and not in consts.py,
#                in order not to have a circular dependency import error)
# ====================================================
DISEASE_AVIAN_INFLUENZA = "AI"
DISEASE_WEST_NILE_VIRUS = "WNV"  # >> TODO not available, but do not comment for the variables defined in src/consts.py

# ====================================================
# HOST TYPES
# ====================================================
HOST_MAMMAL = "mammal"
HOST_AVIAN = "avian"
HOST_HUMAN = "human"


# ====================================================
# USER-SPECIFIC VARIABLES
# >> USER NEEDS TO CHANGE HERE ONLY <<
# ====================================================
#MAIN_FOLDER = "/home/nejat/eclipse/workspace/epidnews2event"
MAIN_FOLDER = "/Users/narinik/Mirror/workspace/GenomeDataPaper"


USER_DISEASE_NAME = DISEASE_AVIAN_INFLUENZA
#USER_DISEASE_NAME = DISEASE_WEST_NILE_VIRUS # >> TODO not available for now, we need to update the whole code for a new disease

##USER_HOST_TYPES = [HOST_MAMMAL]
USER_HOST_TYPES = [HOST_MAMMAL, HOST_AVIAN, HOST_HUMAN]



# GeoNames applies an hourly limit quota for API queries. 
# In order to take advantage the GeoNames' API as much as possible, 
# we are using multiple GeoNames accounts.
GEONAMES_API_USERNAME1 = "arinik9" # for geonames api
GEONAMES_API_USERNAME2 = "arinik1" # for geonames api
GEONAMES_API_USERNAME3 = "arinik2" # for geonames api
GEONAMES_API_USERNAME4 = "arinik10" # for geonames api
GEONAMES_API_USERNAME5 = "arinik11" # for geonames api
GEONAMES_API_USERNAME6 = "arinik12" # for geonames api
GEONAMES_API_USERNAME7 = "arinik13" # for geonames api
GEONAMES_API_USERNAME8 = "arinik14" # for geonames api


