

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



