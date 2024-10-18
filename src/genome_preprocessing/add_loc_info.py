import os
import pandas as pd
import src.consts as consts
import csv

out_result_filepath = os.path.join("/Users/narinik/Mirror/workspace/genome2event/out/preprocessing/BVBRC/analysis", "genome_sim_analysis.csv")

out_genome_preprocessing_folder = consts.OUT_GENOME_PREPROCESSING_BVBRC_FOLDER
out_genome_analysis_folder = os.path.join(out_genome_preprocessing_folder, "analysis")
preprocess_adj_output_filepath = os.path.join(out_genome_preprocessing_folder, "BVBRC_genome_preprocessed.csv")
df_genome = pd.read_csv(preprocess_adj_output_filepath, sep=";", keep_default_na=False)
print(df_genome["Genome Name"])
genomeName2countryCode = dict(zip(df_genome["Genome Name"], df_genome["country_code"]))

serovar_list = ['h10n7', 'h10n8', 'h2n2', 'h3n1', 'h5n1', 'h5n2', 'h5n3', 'h5n4', 'h5n5', 'h5n6', 'h5n8', 'h5n9', 'h6n2', 'h7n1', 'h7n2', 'h7n3', 'h7n4', 'h7n6', 'h7n7', 'h7n8', 'h7n9', 'h9n2', 'h2', 'h3', 'h5', 'h6', 'h7', 'h9', 'h10']
tot = 0
count = 0
for i, s1 in enumerate(serovar_list):
    for j, s2 in enumerate(serovar_list):
        if i <= j:
            tot += 1
            curr_out_result_filepath = out_result_filepath.replace(".csv", "_"+s1+"_"+s2+".csv")
            if os.path.exists(curr_out_result_filepath):
                count += 1
                print(curr_out_result_filepath)
                df = pd.read_csv(curr_out_result_filepath, sep=";", keep_default_na=False)
                df["source_country_code"] = df["source_genome_name"].apply(lambda x: genomeName2countryCode[x] if x in genomeName2countryCode else "-1")
                df["target_country_code"] = df["target_genome_name"].apply(lambda x: genomeName2countryCode[x] if x in genomeName2countryCode else "-1")
                df.to_csv(curr_out_result_filepath, index=False, sep=";",  quoting=csv.QUOTE_NONNUMERIC)

print(count, "/", tot)