'''
Created on Jul 20, 2023

@author: nejat
'''

from src.genome_preprocessing.preprocess_genome_data import preprocess_genome_data, handle_incomplete_dates_in_genome_data_after_preprocessing, split_final_results_into_precise_and_imprecise_serovar
import consts
import os
from src.genome_preprocessing.calculate_genome_similarity import calculate_genome_similarity_for_all_isolates
from src.genome_preprocessing.calculate_avg_genome_sim_by_serotype import calculate_all_avg_genome_sim_values_by_country_and_genome_name, calculate_all_avg_genome_sim_values_by_country, calculate_all_avg_genome_sim_values_by_genome_name, calculate_all_avg_genome_sim_values


if __name__ == '__main__':
  
  in_genome_folder = consts.IN_BVBRC_FOLDER
  out_genome_preprocessing_folder = consts.GENOME_PREPROCESSING_BVBRC_FOLDER
  out_genome_analysis_folder = os.path.join(out_genome_preprocessing_folder, "analysis")
  out_genome_analysis_summary_folder = os.path.join(out_genome_preprocessing_folder, "analysis_summary")
  try:
    if not os.path.exists(out_genome_preprocessing_folder):
      os.makedirs(out_genome_preprocessing_folder)
    if not os.path.exists(out_genome_analysis_folder):
      os.makedirs(out_genome_analysis_folder)
    if not os.path.exists(out_genome_analysis_summary_folder):
      os.makedirs(out_genome_analysis_summary_folder)
  except OSError as err:
    print(err)

  genome_filepath = os.path.join(in_genome_folder, "BVBRC_genome.csv")
  # # #genome_filepath = os.path.join(in_genome_folder, "BVBRC_genome_2021_test.csv")
  genome_seq_filepath = os.path.join(in_genome_folder, "genome_sequences.csv")
  preprocess_output_filepath = os.path.join(out_genome_preprocessing_folder, "BVBRC_genome_preprocessed.csv")
  preprocess_genome_data(in_genome_folder, out_genome_preprocessing_folder, genome_filepath, genome_seq_filepath, preprocess_output_filepath)

  preprocess_adj_output_filepath = os.path.join(out_genome_preprocessing_folder, "BVBRC_genome_preprocessed_adj.csv")
  handle_incomplete_dates_in_genome_data_after_preprocessing(preprocess_output_filepath, preprocess_adj_output_filepath)

  split_final_results_into_precise_and_imprecise_serovar(preprocess_adj_output_filepath, out_genome_preprocessing_folder)

  #   # ====================================
  out_result_filepath = os.path.join(out_genome_analysis_folder,"genome_sim_analysis.csv")
  # #isolates_raw_filepath = preprocess_output_filepath.split(".")[0] + "_aux.csv"
  isolates_filepath = preprocess_output_filepath
  calculate_genome_similarity_for_all_isolates(isolates_filepath, genome_seq_filepath, out_result_filepath)

  calculate_all_avg_genome_sim_values(out_genome_analysis_folder, out_genome_analysis_summary_folder)

  calculate_all_avg_genome_sim_values_by_country(out_genome_analysis_folder, out_genome_analysis_summary_folder)
  calculate_all_avg_genome_sim_values_by_country_and_genome_name(out_genome_analysis_folder, out_genome_analysis_summary_folder)
  calculate_all_avg_genome_sim_values_by_genome_name(out_genome_analysis_folder, out_genome_analysis_summary_folder)