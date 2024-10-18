

"""
6.6 Pairwise sequence alignment
Pairwise sequence alignment is the process of aligning two sequences to each other by optimizing the similarity
score between them. The Bio.Align module contains the PairwiseAligner class for global and local align-
ments using the Needleman-Wunsch, Smith-Waterman, Gotoh (three-state), and Waterman-Smith-Beyer
global and local pairwise alignment algorithms, with numerous options to change the alignment parameters.
We refer to Durbin et al. [16] for in-depth information on sequence alignment algorithms.

[16] Richard Durbin, Sean R. Eddy, Anders Krogh, Graeme Mitchison: “Biological sequence analysis: Prob-
abilistic models of proteins and nucleic acids”. Cambridge University Press, Cambridge, UK (1998).
"""

from Bio import Align
from Bio.Align import substitution_matrices
#from src.util_data import sigmoid, to_minus1_pos1_range
#from statistics import geometric_mean
import numpy as np

# https://stackoverflow.com/questions/76359855/unable-to-access-individual-alignment-strings-in-biopython-pairwise-align
# nucleotide >> section 6.6.6 in http://biopython.org/DIST/docs/tutorial/Tutorial.pdf

def calculate_isolate_raw_distance(isolate1, isolate2, id2seq):
    score, genome_state = calculate_isolate_raw_similarity(isolate1, isolate2, id2seq)
    return (1 - score)

def calculate_isolate_raw_similarity(isolate1, isolate2, id2seq):
    isolate1_keys = isolate1.keys()
    isolate2_keys = isolate2.keys()
    keys = [value for value in isolate1_keys if value in isolate2_keys]
    genome_state = "complete"
    if len(keys) == 0:
        return 0.0, "zero"
    elif len(keys) < 8:
        genome_state = "partial"

    score_list = []
    for k in keys:
        seq1 = id2seq[isolate1[k].replace("bvbrc", "")]
        seq2 = id2seq[isolate2[k].replace("bvbrc", "")]
        try:
            score = calculate_genome_raw_similarity(seq1, seq2)
            print(k, score)
            score_list.append(score)
        except ValueError as err:
            # We may have the following error : "ValueError: sequence contains letters not in the alphabet"
            print(err)
    final_score = -1.0
    if len(score_list) > 0:
        final_score = np.mean(np.array(score_list))

    return final_score, genome_state

# >>> similarity_smoothing_with_quality_scores(0.1, 0.6, 0.6)
# 0.07739679083827737
# >>> similarity_smoothing_with_quality_scores(0.1, 0.6, 0.9)
# 0.06221711100373239
# >>> similarity_smoothing_with_quality_scores(0.1, 0.9, 0.9)
# 0.05697960685200845
# >>> similarity_smoothing_with_quality_scores(0.1, 0.9, 0.99)
# 0.056445344673100264
# >>> similarity_smoothing_with_quality_scores(0.5, 0.9, 0.99)
# 0.42092296524845807
# >>> similarity_smoothing_with_quality_scores(0.5, 0.99, 0.99)
# 0.42046297526397447
# >>> similarity_smoothing_with_quality_scores(0.3, 0.99, 0.99)
# 0.22203838690604535
# >>> similarity_smoothing_with_quality_scores(0.2, 0.99, 0.99)
# 0.13375896898826548
# >>> similarity_smoothing_with_quality_scores(0.1, 0.99, 0.99)
# 0.056240694069047796
# >>> similarity_smoothing_with_quality_scores(0.1, 0.99, 0.01)
# 0.1
# >>> similarity_smoothing_with_quality_scores(0.1, 0.01, 0.01)
# 0.3055624615787688
# >>> similarity_smoothing_with_quality_scores(0.01, 0.01, 0.01)
# 0.09336841792607654
def similarity_smoothing_with_quality_scores(raw_dist_score, a, b):
    epsilon = 0.01 # when raw_dist_score = 0, this poses a problem in the calculation below
    if raw_dist_score == 0.0:
        raw_dist_score = epsilon
    # if (quality_score1+quality_score2-1) is positive, then dist score will be slightly improved
    # else if (quality_score1+quality_score2-1) is negative, then dist score will be slightly worsened
    # why quality_score1+quality_score2-1 ? because if the value of 1 for the sum seems to be the sufficient threshold
    # for instance, 0.3 + 0.8 = 1.1 > 1 (positive)
    # but, 0.3 + 0.4 = 0.7 < 1 (negative)
    born_factor = 4 / np.exp(2 - a - b)
    if (a+b-1)<0:
        born_factor = 4 / (2-a-b)
    return raw_dist_score**(1+((a+b-1)/born_factor))

def calculate_genome_raw_distance(seq1, seq2, scoring_matrix_name="GENETIC"): # seq1_conf_score, seq2_conf_score,
    return 1 - calculate_genome_raw_similarity(seq1, seq2, scoring_matrix_name)

def calculate_genome_raw_similarity(seq1, seq2, scoring_matrix_name="GENETIC"): # seq1_conf_score, seq2_conf_score,
    all_scoring_matrix_names = substitution_matrices.load()
    # ['BENNER22', 'BENNER6', 'BENNER74', 'BLOSUM45', 'BLOSUM50', 'BLOSUM62', 'BLOSUM80', 'BLOSUM90',
    #   'DAYHOFF', 'FENG', 'GENETIC', 'GONNET1992', 'HOXD70', 'JOHNSON', 'JONES', 'LEVIN', 'MCLACHLAN', 'MDM78',
    #       'NUC.4.4', 'PAM250', 'PAM30', 'PAM70', 'RAO', 'RISLER', 'SCHNEIDER', 'STR', 'TRANS']
    if scoring_matrix_name not in all_scoring_matrix_names:
        raise Exception('scoring matrix name is not in the list of available names')
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load(scoring_matrix_name)
    raw_score = aligner.score(seq1.upper(), seq2.upper())
    score = raw_score/max(aligner.score(seq1.upper(), seq1.upper()), aligner.score(seq2.upper(), seq2.upper()))
    #quality_score = to_minus1_pos1_range(seq1_conf_score) + to_minus1_pos1_range(seq2_conf_score)
    #smoothing_factor = sigmoid(quality_score, k=0.33)
    return score
    #return 1- (score*smoothing_factor)

def calculate_genome_distance_with_imprecise_genome_info(articlePair2distScore,
                                               event1_bvbrc_article_id_list, event2_bvbrc_article_id_list,
                                               event1_genome_quality_scores, event2_genome_quality_scores,
                                               event1_genome_dist_values, event2_genome_dist_values,
                                               event1_genome_temp_values, event2_genome_temp_values):
    print("girdi calculate_genome_distance_with_imprecise_genome_info")
    final_dist_score = np.nan
    print("event1", event1_bvbrc_article_id_list, len(event1_bvbrc_article_id_list))
    print("event2", event2_bvbrc_article_id_list, len(event2_bvbrc_article_id_list))
    if len(event1_bvbrc_article_id_list)==0 or len(event2_bvbrc_article_id_list)==0:
        return final_dist_score

    event1_genome_temp_values = [float(x) for x in event1_genome_temp_values]
    event1_genome_dist_values = [float(x) for x in event1_genome_dist_values]
    event1_genome_quality_scores = [float(x) for x in event1_genome_quality_scores]
    event2_genome_temp_values = [float(x) for x in event2_genome_temp_values]
    event2_genome_dist_values = [float(x) for x in event2_genome_dist_values]
    event2_genome_quality_scores = [float(x) for x in event2_genome_quality_scores]


    # # -----------------------
    # TODO: need to find out an appropriate formula
    # epsilon = 0.01  # when this equals 0, this poses a problem in the calculation below
    # event1_genome_dist_values = np.array(event1_genome_dist_values)
    # event1_genome_temp_values = np.array(event1_genome_temp_values)
    # event1_genome_dist_values[event1_genome_dist_values == 0.0] = epsilon
    # event1_genome_temp_values[event1_genome_temp_values == 0.0] = epsilon
    # event2_genome_dist_values = np.array(event2_genome_dist_values)
    # event2_genome_temp_values = np.array(event2_genome_temp_values)
    # event2_genome_dist_values[event2_genome_dist_values == 0.0] = epsilon
    # event2_genome_temp_values[event2_genome_temp_values == 0.0] = epsilon
    #
    # st_sum_values1 = np.sum(event1_genome_temp_values*event1_genome_dist_values)
    # event1_prop_st_values = (event1_genome_temp_values*event1_genome_dist_values)/st_sum_values1
    # event1_prop_st_values = 1 - event1_prop_st_values # favoring less time and spatial distances
    #
    # st_sum_values2 = np.sum(event2_genome_temp_values*event2_genome_dist_values)
    # event2_prop_st_values = (event2_genome_temp_values*event2_genome_dist_values)/st_sum_values2
    # print("event2_prop_st_values", event2_prop_st_values)
    # event2_prop_st_values = 1 - event2_prop_st_values  # favoring less time and spatial distances
    # # weighted multiplication
    # event2_genome_upd_quality_scores = np.array(event2_genome_quality_scores) * event2_prop_st_values
    # weighted multiplication
    # event1_genome_upd_quality_scores = np.array(event1_genome_quality_scores)*event1_prop_st_values
    # # -----------------------

    #print("event1_genome_quality_scores", event1_genome_quality_scores)
    #print("event2_genome_quality_scores", event2_genome_quality_scores)

    event1_genome_upd_quality_scores = np.array(event1_genome_quality_scores)
    event2_genome_upd_quality_scores = np.array(event2_genome_quality_scores)

    dist_scores = []
    for i in range(len(event1_bvbrc_article_id_list)):
        article_id1 = event1_bvbrc_article_id_list[i]
        genome_quality_score1 = event1_genome_upd_quality_scores[i]
        for j in range(len(event2_bvbrc_article_id_list)):
            article_id2 = event2_bvbrc_article_id_list[j]
            genome_quality_score2 = event2_genome_upd_quality_scores[j]
            raw_dist_score = np.nan
            pair1 = str(article_id1) + "_" + str(article_id2)
            pair2 = str(article_id2) + "_" + str(article_id1)
            if pair1 in articlePair2distScore:
                raw_dist_score = articlePair2distScore[pair1]
            elif pair2 in articlePair2distScore:
                raw_dist_score = articlePair2distScore[pair2]
            #print(pair1, pair2, raw_dist_score)
            #print("genome_quality_score1", genome_quality_score1)
            #print("genome_quality_score2", genome_quality_score2)
            smoothed_dist_score = similarity_smoothing_with_quality_scores(raw_dist_score, genome_quality_score1, genome_quality_score2)
            #print("smoothed_dist_score", smoothed_dist_score)
            dist_scores.append(smoothed_dist_score)

    if len(dist_scores) > 0:
        # final_dist_score = geometric_mean(dist_scores)
        final_dist_score = np.mean(dist_scores)
    return final_dist_score


