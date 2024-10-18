import multiprocessing
import pandas as pd

def calculate_matrix_sim_values_in_parallel(events1_list, events2_list, sim_strategy, nb_processes=8):
    entry_list = []
    N = len(events1_list)
    for i, event1 in enumerate(events1_list):
        print(i, "/", N)
        event1_id = event1.e_id
        event1_dis = event1.disease
        for j, event2 in enumerate(events2_list):
            event2_id = event2.e_id
            event2_dis = event2.disease
            if event1_dis.is_identical(event2_dis) or event1_dis.is_hierarchically_included(event2_dis) or event2_dis.is_hierarchically_included(event1_dis):
                # -----------------------------------------------------
                entry = (event1_id, event2_id, event1, event2, sim_strategy)
                entry_list.append(entry)

    list_grouped = [[] for i in range(nb_processes)]
    for i, entry in enumerate(entry_list):
        list_grouped[i % nb_processes].append((entry))
    list_grouped = [(i, l) for i, l in enumerate(list_grouped)]

    with multiprocessing.Pool(processes=nb_processes) as pool:
        # call the function for each item in parallel
        list_result = pool.map(calculate_matrix_sim_values_for_list, list_grouped)
        df_result = pd.concat(list_result)
        #df_result.sort_values(by=['seq1_bvbrc_id', 'seq1', 'seq2_bvbrc_id', 'seq2', 'score'], ascending=True, inplace=True)
        return df_result


def calculate_matrix_sim_values_for_list(params):
    i = params[0]
    entry_list = params[1]

    event1_id_list = []
    event2_id_list = []
    sim_score_list = []
    for entry in entry_list:
        event1_id = entry[0]
        event2_id = entry[1]
        event1 = entry[2]
        event2 = entry[3]
        sim_strategy = entry[4]
        sim_score = sim_strategy.perform_event_similarity(event1, event2, False)
        event1_id_list.append(event1_id)
        event2_id_list.append(event2_id)
        sim_score_list.append(sim_score)

    df = pd.DataFrame(list(zip(event1_id_list, event2_id_list, sim_score_list)),
                      columns = ["event1_id", "event2_id", "sim_score"])
    return df