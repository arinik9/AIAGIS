
import numpy as np
import dateutil.parser as parser
import src.consts as consts
import csv
import os
import pandas as pd

# 1) calculate the effectif year ??? >> for estimating cascades
# 2) (optional) consider only some periods
# 3) sort the data by pub time



def create_temporal_dist_matrix(df_events, output_temporal_dist_matrix_filepath):
    N = df_events.shape[0]
    temporal_dist_matrix = np.full(shape=(N,N), fill_value=np.nan)

    id_list = df_events[consts.COL_ID].to_numpy().flatten()
    date_list = df_events["published_at"].tolist()

    #print("temporal")
    for i, row1 in df_events.iterrows():
        print("i", i, "/", N)
        date1 = row1["published_at"]
        temporal_dist_matrix[i, i] = 0.0
        #print(date1, date_list[i])
        for j, row2 in df_events.iterrows():
            if i<j:
                date2 = row2["published_at"]
                temporal_dist_matrix[i,j] = abs((date1-date2).days)
                #print(date1, date2, abs((date1-date2).days))
                temporal_dist_matrix[j,i] = temporal_dist_matrix[i,j]

    df = pd.DataFrame(temporal_dist_matrix)
    df.index = id_list
    df.columns = id_list
    df.to_csv(output_temporal_dist_matrix_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)


# ======================================
#
# ======================================
def prepare_seasonal_periods(df_events):
    df_events["year_effectif"] = df_events["year"]
    idx = ((df_events["season"] == "winter") & (df_events["month_no_simple"] == 12))
    df_events.loc[idx, "year_effectif"] = df_events.loc[idx, "year"]+1
    #print(df_events.loc[idx, "year_effectif"])
    df_events = df_events[df_events["year_effectif"] != 2022]
    #print(df_events[idx])
    df_events["seasonal_year_period"] = df_events["year_effectif"].apply(str)+"_"+df_events["season"]
    del df_events["year_effectif"]
    return df_events


# ======================================
#
# ======================================
def prepare_timestamp(df_events, date_start, date_end):
    df_events["timestamp_in_hour"] = df_events["published_at"].apply(
        lambda t: (t - date_start).total_seconds() // 3600)  # in hours
    df_events["timestamp_in_day"] = df_events["published_at"].apply(
        lambda t: (t - date_start).total_seconds() // (3600*24))  # in hours
    print("max(timestamp_in_hour)", np.max(df_events["timestamp_in_hour"]))
    print("max(timestamp_in_day)", np.max(df_events["timestamp_in_day"]))
    return df_events


# ======================================
# MAIN FUNCTION
# ======================================
def perform_time_preprocessing(df_events, date_start, date_end, output_temporal_dist_matrix_from_events_filepath, force=False):
    # filtering according to date_start and date_end
    df_events = df_events[(df_events["published_at"] >= date_start) & (df_events["published_at"] <= date_end)]
    df_events.reset_index(drop=True, inplace=True)

    if os.path.exists(output_temporal_dist_matrix_from_events_filepath) or force:
        create_temporal_dist_matrix(df_events, output_temporal_dist_matrix_from_events_filepath)

    ## prepare a new column "timestamp" starting from the starting  date
    df_events = prepare_timestamp(df_events, date_start, date_end)
    # # prepare a new column "seasonal_year_period", indicating seasons, combined with years, as periods
    # df_events = prepare_seasonal_periods(df_events)
    # # sort data by date in ascending order
    # df_events.sort_values(by='published_at', ascending=True, inplace=True)
    # df_events.reset_index(drop=True, inplace=True)
    return df_events


if __name__ == '__main__':
    print('Starting')
    output_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
    events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events.csv") # only 2021 data
    df_events_prep_upd = pd.read_csv(events_filepath, sep=";", keep_default_na=False)

    df_events_prep_upd["published_at"] = df_events_prep_upd["published_at"].apply(lambda x: parser.parse(x))

    date_start = parser.parse("2020-12-31T00:00:00", dayfirst=False)
    date_end = parser.parse("2022-01-01T00:00:00", dayfirst=False)  # ending time

    perform_time_preprocessing(df_events_prep_upd, date_start, date_end)
