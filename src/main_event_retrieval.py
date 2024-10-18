'''
Created on Nov 14, 2021

@author: nejat
'''

from src.event_preprocessing.preprocessing import PreprocessingPadiweb, PreprocessingPromed, PreprocessingEmpresi, PreprocessingWahis, PreprocessingApha, PreprocessingAphis, PreprocessingBVBRC
from src.event_retrieval.retrieve_doc_events import EventRetrievalPadiweb, EventRetrievalEmpresi, EventRetrievalWahis, EventRetrievalPromed, EventRetrievalAphis, EventRetrievalApha, EventRetrievalBVBRC
from src.event_clustering.event_clustering import EventClusteringPadiweb, EventClusteringPromed,\
                                                EventClusteringEmpresi, EventClusteringWahis, EventClusteringApha, EventClusteringAphis, EventClusteringBVBRC
from src.event_fusion.event_fusion import EventFusionPadiweb, EventFusionPromed, EventFusionEmpresi, EventFusionWahis, EventFusionAphis, EventFusionApha, EventFusionBVBRC
from src.event.event_duplicate_identification_strategy import EventDuplicateHierIdentificationStrategy
from src.event_clustering.event_clustering_strategy import EventClusteringStrategyHierDuplicate
from src.event_retrieval.event_retrieval_strategy import EventRetrievalStrategyRelevantSentence, EventRetrievalStrategyStructuredData
from src.event_fusion.event_fusion_strategy import EventFusionStrategyMaxOccurrence
from src.event_normalization.normalize_events import NormalizationPadiweb, NormalizationEmpresi, NormalizationWahis, NormalizationPromed, NormalizationApha, NormalizationAphis, NormalizationBVBRC

from src.event_retrieval.postprocessing_bvbrc import postprocessing_bvbrc
import os
import time
import src.consts as consts
import src.user_consts as user_consts


# to disable the following warning: "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"



if __name__ == '__main__':
  st = time.time()

  #########################################
  # BVBRC
  #########################################
  print("starting with BVBRC ....")

  for folder in [consts.PREPROCESSING_RESULT_BVBRC_FOLDER, consts.NORM_RESULT_BVBRC_FOLDER, consts.DOC_EVENTS_BVBRC_FOLDER, \
                 consts.EVENT_CLUSTERING_BVBRC_FOLDER, consts.CORPUS_EVENTS_BVBRC_FOLDER]:
      try:
          if not os.path.exists(folder):
              os.makedirs(folder)
      except OSError as err:
          print(err)

  consts.IN_BVBRC_FOLDER = consts.GENOME_PREPROCESSING_BVBRC_FOLDER
  prep = PreprocessingBVBRC("BVBRC_genome_preprocessed_adj.csv")
  prep.perform_preprocessing(user_consts.USER_DISEASE_NAME, consts.IN_BVBRC_FOLDER,
                                 consts.PREPROCESSING_RESULT_BVBRC_FOLDER, \
                             consts.DATA_FOLDER)

  norm = NormalizationBVBRC()
  norm.perform_normalization(user_consts.USER_DISEASE_NAME, consts.PREPROCESSING_RESULT_BVBRC_FOLDER, \
                             consts.NORM_RESULT_BVBRC_FOLDER, consts.DATA_FOLDER)

  structured_data_event_retrieval_strategy = EventRetrievalStrategyStructuredData()
  event_retrieval = EventRetrievalBVBRC(structured_data_event_retrieval_strategy)
  event_retrieval.perform_event_retrieval(user_consts.USER_DISEASE_NAME, consts.PREPROCESSING_RESULT_BVBRC_FOLDER, \
                                          consts.NORM_RESULT_BVBRC_FOLDER, consts.DOC_EVENTS_BVBRC_FOLDER,
                                          consts.DATA_FOLDER)

  # we perform a postprocessing to get the initial columns that are not used in this program
  # consts.PREPROCESSING_RESULT_BVBRC_FOLDER
  # consts.DOC_EVENTS_BVBRC_FOLDER,
  raw_events_filepath = os.path.join(consts.IN_BVBRC_FOLDER, "BVBRC_genome_preprocessed_adj.csv")
  processed_events_filepath = os.path.join(consts.DOC_EVENTS_BVBRC_FOLDER, "doc_events_bvbrc_task1=structured_data.csv")
  new_processed_events_filepath = os.path.join(consts.DOC_EVENTS_BVBRC_FOLDER, "doc_events_bvbrc.csv")
  postprocessing_bvbrc(raw_events_filepath, processed_events_filepath, new_processed_events_filepath)


  # event_duplicate_ident_strategy_manual = EventDuplicateHierIdentificationStrategy()
  # event_clustering_strategy = EventClusteringStrategyHierDuplicate()
  # event_clustering_bvbrc_manual = EventClusteringBVBRC(structured_data_event_retrieval_strategy, \
  #                                                    event_duplicate_ident_strategy_manual, event_clustering_strategy)
  # event_clustering_bvbrc_manual.perform_event_clustering(consts.DOC_EVENTS_BVBRC_FOLDER,
  #                                                       consts.EVENT_CLUSTERING_BVBRC_FOLDER)
  #
  # max_occurrence_fusion_strategy = EventFusionStrategyMaxOccurrence()
  # event_fusion_bvbrc = EventFusionBVBRC(structured_data_event_retrieval_strategy, event_clustering_bvbrc_manual, \
  #                                     max_occurrence_fusion_strategy)
  # event_fusion_bvbrc.perform_event_fusion(consts.DOC_EVENTS_BVBRC_FOLDER, consts.EVENT_CLUSTERING_BVBRC_FOLDER, \
  #                                        consts.CORPUS_EVENTS_BVBRC_FOLDER)

  print("ending with BVBRC ....")




  # #########################################
  # # Empres-i
  # #########################################
  print("starting with Empres-i ....")

  for folder in [consts.PREPROCESSING_RESULT_EMPRESI_FOLDER, consts.NORM_RESULT_EMPRESI_FOLDER,
                 consts.DOC_EVENTS_EMPRESI_FOLDER, \
                 consts.EVENT_CLUSTERING_EMPRESI_FOLDER, consts.CORPUS_EVENTS_EMPRESI_FOLDER]:
      try:
          if not os.path.exists(folder):
              os.makedirs(folder)
      except OSError as err:
          print(err)

  prep = PreprocessingEmpresi()
  prep.perform_preprocessing(user_consts.USER_DISEASE_NAME, consts.IN_EMPRESSI_FOLDER,
                             consts.PREPROCESSING_RESULT_EMPRESI_FOLDER, \
                             consts.DATA_FOLDER)

  norm = NormalizationEmpresi()
  norm.perform_normalization(user_consts.USER_DISEASE_NAME, consts.PREPROCESSING_RESULT_EMPRESI_FOLDER, \
                             consts.NORM_RESULT_EMPRESI_FOLDER, consts.DATA_FOLDER)

  structured_data_event_retrieval_strategy = EventRetrievalStrategyStructuredData()
  event_retrieval = EventRetrievalEmpresi(structured_data_event_retrieval_strategy)
  event_retrieval.perform_event_retrieval(user_consts.USER_DISEASE_NAME, consts.PREPROCESSING_RESULT_EMPRESI_FOLDER, \
                                          consts.NORM_RESULT_EMPRESI_FOLDER, consts.DOC_EVENTS_EMPRESI_FOLDER,
                                          consts.DATA_FOLDER)

  # event_duplicate_ident_strategy_manual = EventDuplicateHierIdentificationStrategy()
  # event_clustering_strategy = EventClusteringStrategyHierDuplicate()
  # event_clustering_empresi_manual = EventClusteringEmpresi(structured_data_event_retrieval_strategy, \
  #                                                          event_duplicate_ident_strategy_manual,
  #                                                          event_clustering_strategy)
  # event_clustering_empresi_manual.perform_event_clustering(consts.DOC_EVENTS_EMPRESI_FOLDER, \
  #                                                          consts.EVENT_CLUSTERING_EMPRESI_FOLDER)
  #
  # max_occurrence_fusion_strategy = EventFusionStrategyMaxOccurrence()
  # event_fusion_empresi = EventFusionEmpresi(structured_data_event_retrieval_strategy, event_clustering_empresi_manual, \
  #                                           max_occurrence_fusion_strategy)
  # event_fusion_empresi.perform_event_fusion(consts.DOC_EVENTS_EMPRESI_FOLDER, consts.EVENT_CLUSTERING_EMPRESI_FOLDER, \
  #                                           consts.CORPUS_EVENTS_EMPRESI_FOLDER)

  print("ending with Empres-i ....")



  
  elapsed_time = time.time()-st
  print('Execution time:', elapsed_time/60, 'minutes')
