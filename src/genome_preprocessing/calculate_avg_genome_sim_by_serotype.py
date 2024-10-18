import os
import pandas as pd
import numpy as np
import csv


def calculate_avg_genome_sim_value(df):
    df2 = df[df["genome_state"] != "zero"]
    df2["sim_score"] = df2["sim_score"].astype(float)
    df3 = df2[df2["sim_score"] < 1.0]
    avg = np.mean(df3["sim_score"].to_numpy())
    return(avg)


def calculate_all_avg_genome_sim_values(in_result_folder, out_folder):
    serovar_list = ['h10n7', 'h10n8', 'h2n2', 'h3n1', 'h5n1', 'h5n2', 'h5n3', \
                    'h5n4', 'h5n5', 'h5n6', 'h5n8', 'h5n9', 'h6n2', 'h7n1', 'h7n2', \
                    'h7n3', 'h7n4', 'h7n6', 'h7n7', 'h7n8', 'h7n9', 'h9n2', 'h2', 'h3', \
                    'h5', 'h6', 'h7', 'h9', 'h10']


    serovar_pair_list = []
    avg_list = []
    for i, s1 in enumerate(serovar_list):
        for j, s2 in enumerate(serovar_list):
            if i <= j:
                ##in_result_folder = "/Users/narinik/Mirror/workspace/genome2event/out/preprocessing/BVBRC/analysis"
                curr_in_result_filepath = os.path.join(in_result_folder, "genome_sim_analysis"+"_"+s1+"_"+s2+".csv")
                if os.path.exists(curr_in_result_filepath):
                    print(curr_in_result_filepath)
                    serovar_pair_list.append(s1 + "_" + s2)
                    df = pd.read_csv(curr_in_result_filepath, sep=";", keep_default_na=False, dtype=str)
                    curr_avg = calculate_avg_genome_sim_value(df)
                    avg_list.append(curr_avg)

    df_res = pd.DataFrame(list(zip(serovar_pair_list, avg_list)),
          columns=['serovar_pair','avg_sim_score'])
    out_result_filepath = os.path.join(out_folder, "genome_sim_summary.csv")
    df_res.to_csv(out_result_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)




def calculate_avg_genome_sim_value_by_source_genome_name(df):
    df2 = df[df["genome_state"] != "zero"]
    df2["sim_score"] = df2["sim_score"].astype(float)
    df3 = df2[df2["sim_score"] < 1.0]
    desc_list = []
    for idx, row in df3.iterrows():
        desc_list.append(row["source_genome_name"])
    df3["desc"] = desc_list
    df_res = df3.groupby("desc", group_keys=True)[['sim_score']].mean().reset_index()
    return(df_res)

def calculate_avg_genome_sim_value_by_target_genome_name(df):
    df2 = df[df["genome_state"] != "zero"]
    df2["sim_score"] = df2["sim_score"].astype(float)
    df3 = df2[df2["sim_score"] < 1.0]
    desc_list = []
    for idx, row in df3.iterrows():
        desc_list.append(row["target_genome_name"])
    df3["desc"] = desc_list
    df_res = df3.groupby("desc", group_keys=True)[['sim_score']].mean().reset_index()
    return(df_res)


def calculate_all_avg_genome_sim_values_by_genome_name(in_result_folder, out_folder):
    serovar_list = ['h10n7', 'h10n8', 'h2n2', 'h3n1', 'h5n1', 'h5n2', 'h5n3', \
                    'h5n4', 'h5n5', 'h5n6', 'h5n8', 'h5n9', 'h6n2', 'h7n1', 'h7n2', \
                    'h7n3', 'h7n4', 'h7n6', 'h7n7', 'h7n8', 'h7n9', 'h9n2', 'h2', 'h3', \
                    'h5', 'h6', 'h7', 'h9', 'h10']

    # for i, s1 in enumerate(serovar_list):
    #     for j, s2 in enumerate(serovar_list):
    #         if i <= j:
    #             curr_in_result_filepath = os.path.join(in_result_folder, "genome_sim_analysis"+"_"+s1+"_"+s2+".csv")
    #             if os.path.exists(curr_in_result_filepath):
    #                 print(curr_in_result_filepath)
    #                 df = pd.read_csv(curr_in_result_filepath, sep=";", keep_default_na=False, dtype=str)
    #
    #                 curr_out_result_filepath = os.path.join(out_folder, "genome_sim_summary_by_genome_name" + "_" + s1 + "_" + s2 + ".csv")
    #                 if not os.path.exists(curr_out_result_filepath):
    #                     df_res1 = calculate_avg_genome_sim_value_by_source_genome_name(df)
    #                     df_res2 = calculate_avg_genome_sim_value_by_target_genome_name(df)
    #                     df_res = pd.concat([df_res1, df_res2])
    #                     df_res.to_csv(curr_out_result_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)

    df_list  = []
    for i, s1 in enumerate(serovar_list):
        for j, s2 in enumerate(serovar_list):
            if i <= j:
                curr_result_filepath = os.path.join(out_folder, "genome_sim_summary_by_genome_name" + "_" + s1 + "_" + s2 + ".csv")
                if os.path.exists(curr_result_filepath):
                    df = pd.read_csv(curr_result_filepath, sep=";", keep_default_na=False, dtype=str)
                    df["source_serovar"] = s1
                    df["target_serovar"] = s2
                    df_list.append(df)
    result_filepath = os.path.join(out_folder, "genome_sim_summary_by_genome_name.csv")
    df = pd.concat(df_list)
    df.to_csv(result_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)



def calculate_avg_genome_sim_value_by_country(df):
    df2 = df[df["genome_state"] != "zero"]
    df2["sim_score"] = df2["sim_score"].astype(float)
    df3 = df2[df2["sim_score"] < 1.0]
    desc_list = []
    for idx, row in df3.iterrows():
        desc_list.append(row["source_country_code"] + "_" + row["target_country_code"])
    df3["desc"] = desc_list
    df_res = df3.groupby("desc", group_keys=True)[['sim_score']].mean().reset_index()
    return(df_res)

def calculate_all_avg_genome_sim_values_by_country(in_result_folder, out_folder):
    print("calculate_all_avg_genome_sim_values_by_country")
    serovar_list = ['h10n7', 'h10n8', 'h2n2', 'h3n1', 'h5n1', 'h5n2', 'h5n3', \
                    'h5n4', 'h5n5', 'h5n6', 'h5n8', 'h5n9', 'h6n2', 'h7n1', 'h7n2', \
                    'h7n3', 'h7n4', 'h7n6', 'h7n7', 'h7n8', 'h7n9', 'h9n2', 'h2', 'h3', \
                    'h5', 'h6', 'h7', 'h9', 'h10']

    # for i, s1 in enumerate(serovar_list):
    #     for j, s2 in enumerate(serovar_list):
    #         if i <= j:
    #             curr_in_result_filepath = os.path.join(in_result_folder, "genome_sim_analysis"+"_"+s1+"_"+s2+".csv")
    #             if os.path.exists(curr_in_result_filepath):
    #                 print(curr_in_result_filepath)
    #                 df = pd.read_csv(curr_in_result_filepath, sep=";", keep_default_na=False, dtype=str)
    #
    #                 curr_out_result_filepath = os.path.join(out_folder, "genome_sim_summary_by_country" + "_" + s1 + "_" + s2 + ".csv")
    #                 if not os.path.exists(curr_out_result_filepath):
    #                     df_res = calculate_avg_genome_sim_value_by_country(df)
    #                     df_res.to_csv(curr_out_result_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)

    df_list  = []
    for i, s1 in enumerate(serovar_list):
        print(i)
        for j, s2 in enumerate(serovar_list):
            if i <= j:
                curr_result_filepath = os.path.join(out_folder, "genome_sim_summary_by_country" + "_" + s1 + "_" + s2 + ".csv")
                if os.path.exists(curr_result_filepath):
                    df = pd.read_csv(curr_result_filepath, sep=";", keep_default_na=False, dtype=str)
                    df["source_serovar"] = s1
                    df["target_serovar"] = s2
                    df_list.append(df)
    result_filepath = os.path.join(out_folder, "genome_sim_summary_by_country.csv")
    df = pd.concat(df_list)
    df.to_csv(result_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)



def calculate_avg_genome_sim_value_by_target_country_and_source_genome_name(df):
    df2 = df[df["genome_state"] != "zero"]
    df2["sim_score"] = df2["sim_score"].astype(float)
    df3 = df2[df2["sim_score"] < 1.0]
    desc_list = []
    for idx, row in df3.iterrows():
        desc_list.append(row["source_genome_name"] + "_" + row["target_country_code"])
    df3["desc"] = desc_list
    df_res = df3.groupby("desc", group_keys=True)[['sim_score']].mean().reset_index()
    return(df_res)

def calculate_avg_genome_sim_value_by_source_country_and_target_genome_name(df):
    df2 = df[df["genome_state"] != "zero"]
    df2["sim_score"] = df2["sim_score"].astype(float)
    df3 = df2[df2["sim_score"] < 1.0]
    desc_list = []
    for idx, row in df3.iterrows():
        desc_list.append(row["target_genome_name"] + "_" + row["source_country_code"])
    df3["desc"] = desc_list
    df_res = df3.groupby("desc", group_keys=True)[['sim_score']].mean().reset_index()
    return(df_res)

def calculate_all_avg_genome_sim_values_by_country_and_genome_name(in_result_folder, out_folder):
    serovar_list = ['h10n7', 'h10n8', 'h2n2', 'h3n1', 'h5n1', 'h5n2', 'h5n3', \
                    'h5n4', 'h5n5', 'h5n6', 'h5n8', 'h5n9', 'h6n2', 'h7n1', 'h7n2', \
                    'h7n3', 'h7n4', 'h7n6', 'h7n7', 'h7n8', 'h7n9', 'h9n2', 'h2', 'h3', \
                    'h5', 'h6', 'h7', 'h9', 'h10']

    # for i, s1 in enumerate(serovar_list):
    #     for j, s2 in enumerate(serovar_list):
    #         if i <= j:
    #             curr_in_result_filepath = os.path.join(in_result_folder, "genome_sim_analysis"+"_"+s1+"_"+s2+".csv")
    #             if os.path.exists(curr_in_result_filepath):
    #                 print(curr_in_result_filepath)
    #                 df = pd.read_csv(curr_in_result_filepath, sep=";", keep_default_na=False, dtype=str)
    #
    #                 curr_out_result_filepath = os.path.join(out_folder, "genome_sim_summary_by_country_and_genome_name" + "_" + s1 + "_" + s2 + ".csv")
    #                 if not os.path.exists(curr_out_result_filepath):
    #                     df_res1 = calculate_avg_genome_sim_value_by_target_country_and_source_genome_name(df)
    #                     df_res2 = calculate_avg_genome_sim_value_by_source_country_and_target_genome_name(df)
    #                     df_res = pd.concat([df_res1, df_res2])
    #                     df_res.to_csv(curr_out_result_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)

    df_list  = []
    for i, s1 in enumerate(serovar_list):
        for j, s2 in enumerate(serovar_list):
            if i <= j:
                curr_result_filepath = os.path.join(out_folder, "genome_sim_summary_by_country_and_genome_name" + "_" + s1 + "_" + s2 + ".csv")
                if os.path.exists(curr_result_filepath):
                    df = pd.read_csv(curr_result_filepath, sep=";", keep_default_na=False, dtype=str)
                    df["source_serovar"] = s1
                    df["target_serovar"] = s2
                    df_list.append(df)
    result_filepath = os.path.join(out_folder, "genome_sim_summary_by_country_and_genome_name.csv")
    df = pd.concat(df_list)
    df.to_csv(result_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)