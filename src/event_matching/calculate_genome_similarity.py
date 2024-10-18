import numpy as np
import pandas as pd
import json
from src.event_matching.genome_similarity import calculate_genome_raw_similarity
import csv
import os
import multiprocessing
import time

def calculate_isolate_similarity(isolate1, isolate2, id2seq):
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
            score_list.append(score)
        except ValueError as err:
            # We may have the following error : "ValueError: sequence contains letters not in the alphabet"
            print(err)
    final_score = -1.0
    if len(score_list) > 0:
        final_score = np.mean(np.array(score_list))

    return final_score, genome_state

