# import os
# import pandas as pd
# import csv
# import numpy as np
# import json
# import multiprocessing
# from itertools import repeat
# import dateutil.parser as parser
# import seaborn as sns # library for visualization
#
# import matplotlib.pyplot as plt
# from datetime import date, datetime
#
# #from scipy.stats import rayleigh
# import geopandas as gpd
# from itertools import permutations
# from src.util_gis import read_map_shapefilepath
# from src.util_event import build_disease_instance
#
# from src.plot.plot_graph import plot_graph_on_map2
#
# from src.inference.MultiC.obj_functions import objective_single_layer_phase_with_dist_survival, objective_single_layer_phase_with_spatial_survival
#
#
# from scipy.special import logit
# from scipy.stats import spearmanr
# import networkx as nx
# import random
# import time
# import pickle
# from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score, precision_recall_curve, auc
# import torch
# from tqdm import tqdm
#
# import src.consts as consts
#
#
# # from torch.utils.tensorboard import SummaryWriter
# # from tensorboard import notebook
# #
# # TB_PATH = "/tmp/logs/module2"
# # %load_ext tensorboard
# # %tensorboard --logdir {TB_PATH}
# #
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(device)
#
#
# # if torch.backends.mps.is_available():
# #     mps_device = torch.device("mps")
# #     x = torch.ones(1, device=mps_device)
# #     print (x)
# # else:
# #     print ("MPS device not found.")
#
#
#
#
# # ======================================================================
# # preprocessing
# # ======================================================================
#
#
# # TODO: Recognize cascades of size above the cascades size threshold. In the original paper, it was 8.
# def prepare_non_zero_cascade_data(df_events, cascade_list, map_info_data, N):
#     id_list = map_info_data["gn_id"].to_numpy().flatten()
#     id2geonameid = dict(zip(range(N), id_list))
#     geonameid2id = dict(zip(id_list, range(N)))
#     df_events["newid"] = df_events["ADM1_geonameid"].apply(lambda x: geonameid2id[x])
#
#     C = len(cascade_list)
#     data_nz = {}  # nz: nonzero
#     for i in range(C):
#         event_ids_str = cascade_list[i]
#         event_ids = [int(id) for id in event_ids_str.split(",")]
#         # C_nz_idx.append(i)
#         df_sub = df_events[df_events[consts.COL_ID].isin(event_ids)]
#         df_sub.sort_values(by=["timestamp_in_day"], inplace=True)
#         data_nz[i] = list(zip(df_sub["timestamp_in_day"], df_sub["newid"]))  # (infection time, location id)
#     return data_nz
#
#
# # input:
# #  - N: nb nodes
# #  - C: nb cascades
# #
# # output:
# #  - time_dict (keys: edge indexes with i*N+j, values: delta T)
# #  - mask_dict (keys: index of each cascade c, values: edge indexes of cascade c)
# #  - succ_edges_list: [edge idx1, edge idx2, ...]
# #  - succ_edges_dict: (key: init edge idx, value: new edge idx) >> example: {edge idx1: 0, edge idx2: 1, ..., edge idx ?: nb succ edges}
# def construct_tensors_from_nonzero_cascades_single_layer_phase(data_nz, N, C, T_end):
#     print("T_end", T_end)
#     # Construct data tensors from nonzero cascades
#     #time_dict = {e: 0.0 for e in range(N * N)}  # transmission edge'lere index atiyo hash yontemiyle, o yuzden init yaparak N*N yapiyo
#     #time_dict = {e: 0.0 for e in range(N * N)}
#     succ_mask_dict = {c: set() for c in range(C)}
#     fail_mask_dict = {c: set() for c in range(C)}
#
#     succ_edges = set()
#     fail_edges = set()
#     for c in range(C):
#         if c % 5 == 0 and c != 0:
#             print('Finished parsing %d cascades' % c)
#
#         sim_logs = data_nz[c]
#         succ_users = []
#         fail_users = set(range(N))
#         #print(sim_logs)
#
#         for (t,j) in sim_logs:  # for each cascade c, it returns a list of tuples (t,j) stating that node j is infected at time t.
#             if t > T_end:
#                 break
#             # Instead of adding all edge pairs, the authors notice that it is unlikely for an edge to exist
#             # between two nodes that never occur in the same cascade.
#             # Therefore, in the single layer phase, we only consider the
#             # set of “possible” edges – i.e., edges between nodes that co-occur
#             # in at least one cascade
#             if j in fail_users:  # it ensures that we do not add the same location id twice
#                 for (u, t_u) in succ_users:  # the previously infected locations (succ_users) might infect the location j
#                     if u != j:
#                         idx = u * N + j  # index of the directed edge (u, j)
#                         # nejat.append((u, j, idx))
#                         succ_mask_dict[c].add(idx)
#                         succ_edges.add(idx)
#                         #time_dict[idx] += t - t_u
#                     else:
#                         break
#                 succ_users.insert(0, (j, t))  # node j is activated at time t on cascade c
#                 fail_users.remove(j)
#
#         for (j, t_j) in succ_users:  # for each activated node j on cascade c
#             # note that t_j is always <= T_end
#             for n in fail_users:
#                 idx = j * N + n
#                 fail_mask_dict[c].add(idx)
#                 fail_edges.add(idx)
#                 #time_dict[idx] += T_end - t_j
#
#     succ_edges_list = list(succ_edges)
#     fail_edges_list = list(fail_edges)
#     #return time_dict, succ_mask_dict, fail_mask_dict, succ_edges_list, fail_edges_list
#     return succ_mask_dict, fail_mask_dict, succ_edges_list, fail_edges_list
#
#
# # N: nb nodes/locations
# # D: spatial distance matrix
# def construct_dist_list_for_succ_edges_single_layer_phase(N, D, succ_edges_list, map_info_data=None, debug=False):
#     n_succ_edges = len(succ_edges_list)
#
#     dist_list_for_succ_edges = []
#     n_poss_edges = N * (N - 1)
#     print(n_succ_edges, "/", n_poss_edges, "=", n_succ_edges / n_poss_edges)
#
#     # prepare dist list for succ edges
#     for i in range(n_succ_edges):
#         e_idx = succ_edges_list[i]
#         # e_idx hash value gibi calculate edildigi icin, o hash value'dan node index'leri geri buluyo
#         i = e_idx // N
#         j = e_idx % N
#         dist = D[i, j]
#         if debug and dist == '':
#             #print(i,j)
#             id_list = map_info_data["gn_id"].to_numpy().flatten()
#             id2geonameid = dict(zip(range(N), id_list))
#             print(id2geonameid[i], id2geonameid[j])
#         dist_list_for_succ_edges.append(dist)
#
#     # print(dist_for_succ_edges[:10])
#     return dist_list_for_succ_edges
#
#
# def construct_temporal_dist_list_for_succ_edges_single_layer_phase(N, D, succ_edges_list, map_info_data=None, debug=False):
#     n_succ_edges = len(succ_edges_list)
#
#     dist_list_for_succ_edges = []
#     n_poss_edges = N * (N - 1)
#     print(n_succ_edges, "/", n_poss_edges, "=", n_succ_edges / n_poss_edges)
#
#     # prepare dist list for succ edges
#     for i in range(n_succ_edges):
#         e_idx = succ_edges_list[i]
#         # e_idx hash value gibi calculate edildigi icin, o hash value'dan node index'leri geri buluyo
#         i = e_idx // N
#         j = e_idx % N
#         dist = D[i, j]
#         if debug and dist == '':
#             #print(i,j)
#             id_list = map_info_data["gn_id"].to_numpy().flatten()
#             id2geonameid = dict(zip(range(N), id_list))
#             print(id2geonameid[i], id2geonameid[j])
#         dist_list_for_succ_edges.append(dist)
#
#     # print(dist_for_succ_edges[:10])
#     return dist_list_for_succ_edges
#
# # def construct_dist_list_for_succ_edges_single_layer_phase(N, D, succ_edges_list):
# #     n_succ_edges = len(succ_edges_list)
# #
# #     dist_list_for_succ_edges = []
# #     n_poss_edges = N * (N - 1)
# #     print(n_succ_edges, "/", n_poss_edges, "=", n_succ_edges / n_poss_edges)
# #
# #     # prepare dist list for succ edges
# #     for i in range(n_succ_edges):
# #         e_idx = succ_edges_list[i]
# #         # e_idx hash value gibi calculate edildigi icin, o hash value'dan node index'leri geri buluyo
# #         i = e_idx // N
# #         j = e_idx % N
# #         dist = D[i, j]
# #         dist_list_for_succ_edges.append(dist)
# #
# #     # print(dist_for_succ_edges[:10])
# #     return dist_list_for_succ_edges
#
#
#
# def construct_mask_and_scatter_idx_single_layer_phase(N, C, time_dict, succ_mask_dict, succ_edges_list,
#                                                         fail_mask_dict, fail_edges_list,
#                                                       spatial_dist_list_for_succ_edges, genome_dist_list_for_succ_edges):
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     overall_start_time = time.time()
#
#     succ_edges_dict = {k: v for v, k in enumerate(succ_edges_list)}  # {'a': 0, 'c': 1}
#     fail_edges_dict = {k: v for v, k in enumerate(fail_edges_list)}  # {'a': 0, 'c': 1}
#
#     # Construct time tensor
#     succ_time_val = np.array(list(time_dict.values()))[succ_edges_list]  # keep only successful activation times
#     succ_delta_t = torch.FloatTensor(succ_time_val).to(device)
#     fail_time_val = np.array(list(time_dict.values()))[fail_edges_list]  # keep only successful activation times
#     fail_delta_t = torch.FloatTensor(fail_time_val).to(device)
#
#     # Construct spatial dist tensor
#     spatial_dist_val = np.array(spatial_dist_list_for_succ_edges)  # only successful activation times
#     spatial_delta_dist = torch.FloatTensor(spatial_dist_val).to(device)
#
#     # Construct genome dist tensor
#     genome_dist_val = np.array(genome_dist_list_for_succ_edges)  # only successful activation times
#     genome_delta_dist = torch.FloatTensor(genome_dist_val).to(device)
#
#     # Construct mask and scatter tensor
#     succ_mask_idx = []
#     succ_scatter_idx = []
#     succ_node_idx = []
#     fail_mask_idx = []
#     fail_scatter_idx = []
#     for c in range(C):
#         # --------------------------------------------
#         # succ edges
#         # --------------------------------------------
#         succ_mask_idx_list = []
#         succ_scatter_idx_list = []
#         succ_node_idx_list = []
#         for old_idx in succ_mask_dict[c]:  # succ edges on cascade c
#             # we retrieve the node indexes i and j from the previously calculated hash value
#             i = old_idx // N
#             j = old_idx % N
#             new_index = j * C + c  # in this new index, node i is not that important, because it is node j which is activated
#             # node j can be involved in one or multiple cascades
#             succ_scatter_idx_list.append(new_index)
#             succ_node_idx_list.append(j)
#             # succ_edges_dict = {8194: 0, 3: 1, 8196: 2, 5: 3, 8198: 4, 6 ... }
#             succ_mask_idx_list.append(succ_edges_dict[old_idx])
#         succ_mask_idx.append(torch.LongTensor(succ_mask_idx_list).to(device))
#         succ_scatter_idx.append(torch.LongTensor(succ_scatter_idx_list).to(device))
#         succ_node_idx.append(torch.LongTensor(succ_node_idx_list).to(device))
#         # --------------------------------------------
#         # fail edges
#         # --------------------------------------------
#         fail_mask_idx_list = []
#         fail_scatter_idx_list = []
#         for old_idx in fail_mask_dict[c]:  # fail edges on cascade c
#             # we retrieve the node indexes i and j from the previously calculated hash value
#             i = old_idx // N
#             j = old_idx % N
#             new_index = j * C + c  # in this new index, node i is not that important, because it is node j which is activated
#             # node j can be involved in one or multiple cascades
#             fail_scatter_idx_list.append(new_index)
#             fail_mask_idx_list.append(fail_edges_dict[old_idx])
#         fail_mask_idx.append(torch.LongTensor(fail_mask_idx_list).to(device))
#         fail_scatter_idx.append(torch.LongTensor(fail_scatter_idx_list).to(device))
#
#     return succ_mask_idx, succ_scatter_idx, succ_delta_t, fail_mask_idx, fail_scatter_idx, fail_delta_t, spatial_delta_dist, genome_delta_dist, succ_node_idx
#     #return succ_mask_idx, scatter_idx, node_idx, spatial_delta_dist, delta_t, genome_delta_dist
#
#
#
#
# def perform_data_parsing_and_tensor_construction_single_layer_phase(df_events, date_start, date_end,
#                                                                     cascade_list, D, genome_dist_matrix, map_info_data, out_dist_values_filepath=None):
#     # init
#     T_end = (date_end - date_start).total_seconds() // (3600*24)
#     #T_end = 360
#     C = len(cascade_list)
#     N = D.shape[0]  # distance matrix is square matrix, the values in each column/row represents all existing locations
#     # df_map_nz = df_map[df_map["gn_id"] != -1]
#     # id_list = df_map_nz["gn_id"].to_numpy().flatten()
#     # N = len(id_list)
#     max_dist = np.max(D)
#
#     # processing
#     data_nz = prepare_non_zero_cascade_data(df_events, cascade_list, map_info_data, N)
#     #time_dict, succ_mask_dict, fail_mask_dict, succ_edges_list, fail_edges_list = construct_tensors_from_nonzero_cascades_single_layer_phase(data_nz, N, C, T_end)
#     succ_mask_dict, fail_mask_dict, succ_edges_list, fail_edges_list = construct_tensors_from_nonzero_cascades_single_layer_phase(data_nz, N, C, T_end)
#     spatial_dist_list_for_succ_edges_init = construct_dist_list_for_succ_edges_single_layer_phase(N, D, succ_edges_list)
#     spatial_dist_list_for_succ_edges = [(float(dist)/max_dist)**(1/2) for dist in spatial_dist_list_for_succ_edges_init]
#     genome_dist_list_for_succ_edges_init = construct_dist_list_for_succ_edges_single_layer_phase(N, genome_dist_matrix, succ_edges_list, map_info_data, False)
#     #print(genome_dist_list_for_succ_edges)
#     print("nb null values", len([dist for dist in genome_dist_list_for_succ_edges_init if dist == '']))
#     print("nb no null values", len([dist for dist in genome_dist_list_for_succ_edges_init if dist != '']))
#     #genome_dist_list_for_succ_edges = [float(dist)  if dist != '' else 1.0 for dist in genome_dist_list_for_succ_edges]
#     genome_dist_list_for_succ_edges = [(1.0-(1.0-float(dist))**2) if dist != '' else 1.0 for dist in genome_dist_list_for_succ_edges_init]
#     #genome_dist_list_for_succ_edges = [float(dist)*5000 if dist != '' else 1.0*5000 for dist in genome_dist_list_for_succ_edges]
#
#     temporal_dist_list_for_succ_edges
#
#     if out_dist_values_filepath is not None:
#         df = pd.DataFrame(list(zip(spatial_dist_list_for_succ_edges_init, spatial_dist_list_for_succ_edges, genome_dist_list_for_succ_edges_init, genome_dist_list_for_succ_edges)), columns=["spatial raw", "spatial norm", "genome raw", "genome norm"])
#         df.to_csv(out_dist_values_filepath, sep=";", index=False)
#
#     succ_mask_idx, succ_scatter_idx, succ_delta_t, fail_mask_idx, fail_scatter_idx, fail_delta_t, spatial_delta_dist, genome_delta_dist, succ_node_idx = construct_mask_and_scatter_idx_single_layer_phase(
#                                                                             N, C, time_dict, succ_mask_dict, succ_edges_list,
#                                                                             fail_mask_dict, fail_edges_list,
#                                                                              spatial_dist_list_for_succ_edges,
#                                                                              genome_dist_list_for_succ_edges)
#
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     T_end = np.max(df_events["timestamp_in_day"].tolist())
#     succ_delta_t = [t/T_end for t in list(succ_delta_t.cpu())]
#     succ_delta_t = torch.FloatTensor(succ_delta_t).to(device)
#     return succ_edges_list, succ_mask_idx, succ_scatter_idx, succ_delta_t, fail_edges_list, fail_mask_idx, fail_scatter_idx, fail_delta_t, spatial_delta_dist, genome_delta_dist, succ_node_idx
#
#
# # ====================================================================================
#
# # ======================================================================
# # optimization
# # ======================================================================
#
#
#
#
#
# def perform_optimization_single_layer_phase(obj_func, opt_params, N, C,
#                                             succ_mask_idx, succ_scatter_idx, succ_node_idx, succ_delta_t,
#                                             fail_mask_idx, fail_scatter_idx, fail_delta_t,
#                                             n_succ_edges, n_fail_edges,
#                                             delta_spatial_dist, delta_genome_dist,
#                                             include_spatial_dist, include_genome_dist, include_epsilon,
#                                             include_unactivated_nodes, beta, gamma):
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#
#     # reproducibility purposes
#     random.seed(0)
#     np.random.seed(0)
#     torch.manual_seed(0)
#
#     # Optimization parameters
#     max_iter = opt_params["max_iter"]
#     min_iter = opt_params["min_iter"]
#     learning_rate = opt_params["learning_rate"]
#     tol = opt_params["tol"]
#
#     params_size = n_succ_edges
#     if include_unactivated_nodes:
#         params_size += n_fail_edges
#     if include_epsilon:
#         params_size += N
#     #if beta is None:
#     #    params_size += 1
#
#         # Initialize parameters: we put all parameters into the same array to optimize them togethr in the Adam opt.
#     params_init = np.random.uniform(-5, 5, params_size)  # sigmoid'e koycagi icin boyle init yapiyo
#     #params_init = np.random.uniform(-5, 5, n_succ_edges) # sigmoid'e koycagi icin boyle init yapiyo
#                                                         # sigmoid(-5) = 0.0067, sigmoid(5) = 0.9933
#     params_g = torch.tensor(params_init, requires_grad=True, device=device, dtype=torch.float)
#
#     # Initialize optimizer
#     opt = torch.optim.Adam([params_g], lr=learning_rate)
#
#     infer_time = 0
#     lik_list = []
#     # Conduct optimization
#     print("Conduct optimization ..")
#
#     for it in range(max_iter):
#         # Calculate objective
#         start_time = time.time()
#         # ==========================
#         loss = np.nan
#         if obj_func == "dist_survival":
#             loss = objective_single_layer_phase_with_dist_survival(params_g, N, C,
#                                                                    n_succ_edges, succ_mask_idx, succ_scatter_idx, succ_node_idx, succ_delta_t,
#                                                                    n_fail_edges, fail_mask_idx, fail_scatter_idx, fail_delta_t,
#                                                                     delta_spatial_dist, include_spatial_dist, include_epsilon,
#                                                                    include_genome_dist, delta_genome_dist,
#                                                                    include_unactivated_nodes, beta, gamma)
#         # elif obj_func == "spatial_survival":
#         #     loss = objective_single_layer_phase_with_spatial_survival(params_g, mask_idx, scatter_idx, N, C, n_succ_edges,
#         #                                         delta_spatial_dist, delta_t, delta_genome_dist,
#         #                                         include_spatial_dist, include_genome_dist, include_epsilon, beta)
#         # ==========================
#         loss_val = loss.item()
#         lik_list.append(loss_val)
#         end_time = time.time()
#         infer_time += end_time - start_time
#
#         print('Iteration %d loss: %.4f' % (it + 1, loss_val))
#
#         # Stop optimization when relative decrease in objective value lower than threshold
#         if it > min_iter and len(lik_list) >= 2 and (lik_list[-2] - lik_list[-1]) / lik_list[-2] < tol:
#             break
#
#         # Loss propagation
#         start_time = time.time()
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         end_time = time.time()
#         infer_time += end_time - start_time
#
#     print('\nOptimization time for N=%d, C=%d' % (N, C))
#     if torch.backends.mps.is_available():
#         print('Max cuda memory allocated: %.4f' % torch.mps.current_allocated_memory()) # TDDO: check
#
#     return params_g, lik_list
#
#
#
# def postprocess_optimization_result_single_layer_phase(params_g, succ_edges_list, lik_list, sample_size, N, n_succ_edges, n_fail_edges,
#                                     out_folder, raw_res_filename, inferred_edgelist_filename,
#                                     map_info_data, graph_single_layer_filepath, include_epsilon=True,
#                                     include_unactivated_nodes=False, include_beta=False):
#     alpha_inferred_nonzero = torch.sigmoid(params_g[:n_succ_edges]).cpu().detach().numpy()
#     #beta_inferred_nonzero = torch.sigmoid(params_g[(n_succ_edges):(2 * n_succ_edges)]).cpu().detach().numpy()
#     if include_unactivated_nodes == False:
#         n_fail_edges = 0
#
#     beta_inferred = None
#     eps_inferred_nonzero = None
#     if include_epsilon:
#         eps_inferred_nonzero = torch.sigmoid(params_g[(n_succ_edges+n_fail_edges):(n_succ_edges + n_fail_edges + N)]).cpu().detach().numpy()
#         if include_beta:
#             beta_inferred = torch.sigmoid(params_g[(n_succ_edges+n_fail_edges+ N):(n_succ_edges + n_fail_edges + N+1)]).cpu().detach().numpy()
#     if beta_inferred is None and include_beta:
#         beta_inferred = torch.sigmoid(params_g[(n_succ_edges+n_fail_edges):(n_succ_edges + n_fail_edges +1)]).cpu().detach().numpy()
#
#     # Save results to file
#     res_dict = {}
#     res_dict['succ_edges_list'] = succ_edges_list
#     res_dict['alpha_inferred_nonzero'] = alpha_inferred_nonzero
#     #res_dict['beta_inferred_nonzero'] = beta_inferred_nonzero
#     if include_epsilon:
#         res_dict['eps_inferred_nonzero'] = eps_inferred_nonzero
#     if include_beta:
#         res_dict['beta_inferred'] = beta_inferred
#     res_dict['lik'] = lik_list
#     # res_dict['time'] = {'parse': parse_time, 'tensor': tensor_time, 'infer': infer_time, 'eval': eval_time}
#     if torch.cuda.is_available():
#         res_dict['memory'] = torch.torch.cuda.max_memory_allocated()
#
#     K = params_g.shape[0]
#     #K = N+2
#     print("L", lik_list[-1], ", n_succ_edges", n_succ_edges, ", sample_size", sample_size, ", K", K)
#     print(2*K*(sample_size/(sample_size-K-1)))
#     AICc = 2*lik_list[-1] + 2*K*(sample_size/(sample_size-K-1))
#     print("AICc", AICc)
#     df_AICc = pd.DataFrame([AICc], columns=["AICc"])
#     AICc_filename = "AICc.csv"
#     AICc_filepath = os.path.join(out_folder, AICc_filename)
#     df_AICc.to_csv(AICc_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)
#
#
#     id_list = map_info_data["gn_id"].to_numpy().flatten()
#     id2geonameid = dict(zip(range(N), id_list))
#     geonameid2id = dict(zip(id_list, range(N)))
#     geonameid2name = dict(zip(id_list, map_info_data["name"]))
#     geonameid2country = dict(zip(id_list, map_info_data["admin"]))
#     geonameid2lon = dict(zip(id_list, map_info_data["lon"]))
#     geonameid2lat = dict(zip(id_list, map_info_data["lat"]))
#
#
#     print("nb nan values in alpha:", len(res_dict['alpha_inferred_nonzero'][np.isnan(res_dict['alpha_inferred_nonzero'])]))
#     print("alpha min", np.min(res_dict['alpha_inferred_nonzero']))
#     print("alpha max", np.max(res_dict['alpha_inferred_nonzero']))
#     #print(np.max(res_dict['beta_inferred_nonzero']))
#     #print(np.max(res_dict['eps_inferred_nonzero']))
#
#     if include_epsilon:
#         print("epsilon min", np.min(res_dict['eps_inferred_nonzero']))
#         print("epsilon max", np.max(res_dict['eps_inferred_nonzero']))
#
#     if include_beta:
#         print("beta", res_dict['beta_inferred'])
#
#     # write raw results into pickle file
#     raw_res_filepath = os.path.join(out_folder, raw_res_filename)
#     with open(raw_res_filepath, 'wb') as fp:
#         pickle.dump(res_dict, fp)
#
#     threshold = 0.0
#     E_ratio = 1.1
#     E_infer = int(E_ratio * N)  # we expect slightly more than nb nodes
#     print( "len(alpha_inferred_nonzero)", len(alpha_inferred_nonzero), "E_infer", E_infer)
#     if len(alpha_inferred_nonzero) > E_infer:
#         print("entered")
#         # Rank the edges by inferred edge weight in the single layer phase
#         print(len(alpha_inferred_nonzero))
#         print(E_infer)
#         alpha_inferred_sorted = np.partition(alpha_inferred_nonzero, len(alpha_inferred_nonzero) - E_infer)
#         threshold = alpha_inferred_sorted[len(alpha_inferred_nonzero) - E_infer]
#         print("threshold:",threshold)
#         # edge weight threshold above which there are <E_infer> edges
#
#     # -----------------
#
#     A = np.array(res_dict['alpha_inferred_nonzero'])
#     A[A < 0.01] = None
#
#     #B = np.array(res_dict['beta_inferred_nonzero'])
#     #B[B < 0.01] = None
#
#     if include_epsilon:
#         eps = np.array(res_dict['eps_inferred_nonzero'])
#         #eps[eps < 0.01] = None
#         print("len(eps)", eps.shape)
#
#         gn_id_list = [id2geonameid[i] for i in range(N)]
#         name_list = [geonameid2name[i] for i in gn_id_list]
#         country_list = [geonameid2country[i] for i in gn_id_list]
#         df_inferred_epsilon = pd.DataFrame({
#             "unobseved_source_proba": eps,
#             "geonameId": gn_id_list,
#             "loc_name": name_list,
#             "country": country_list
#         })
#         inferred_epsilon_filename = "inferred_epsilon.csv"
#         inferred_epsilon_filepath = os.path.join(out_folder, inferred_epsilon_filename)
#         df_inferred_epsilon.to_csv(inferred_epsilon_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)
#
#     if include_beta:
#         beta = np.array(res_dict['beta_inferred'])
#         df_inferred_beta = pd.DataFrame([beta], columns=["beta_inferred"])
#         inferred_beta_filename = "inferred_beta.csv"
#         inferred_beta_filepath = os.path.join(out_folder, inferred_beta_filename)
#         df_inferred_beta.to_csv(inferred_beta_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)
#
#     #write edgelist into file
#     inferred_src_gn_id_list = []
#     inferred_src_lon_list = []
#     inferred_src_lat_list = []
#     inferred_src_name_list = []
#     inferred_src_country_list = []
#     inferred_trgt_gn_id_list = []
#     inferred_trgt_lon_list = []
#     inferred_trgt_lat_list = []
#     inferred_trgt_name_list = []
#     inferred_trgt_country_list = []
#     inferred_w_a_list = []
#     #inferred_w_b_list = []
#
#     for new_idx, old_idx in enumerate(succ_edges_list):
#         i = old_idx // N
#         j = old_idx % N
#         if A[new_idx]>=threshold: # << threshold is determined by the previous step
#             src_gn_id = id2geonameid[i]
#             trgt_gn_id = id2geonameid[j]
#             inferred_src_gn_id_list.append(src_gn_id)
#             inferred_src_lon_list.append(geonameid2lon[src_gn_id])
#             inferred_src_lat_list.append(geonameid2lat[src_gn_id])
#             inferred_src_name_list.append(geonameid2name[src_gn_id])
#             inferred_src_country_list.append(geonameid2country[src_gn_id])
#             inferred_trgt_gn_id_list.append(trgt_gn_id)
#             inferred_trgt_lon_list.append(geonameid2lon[trgt_gn_id])
#             inferred_trgt_lat_list.append(geonameid2lat[trgt_gn_id])
#             inferred_trgt_name_list.append(geonameid2name[trgt_gn_id])
#             inferred_trgt_country_list.append(geonameid2country[trgt_gn_id])
#             inferred_w_a_list.append(A[new_idx])
#             #inferred_w_b_list.append(B[new_idx])
#     df_inferred_edgelist = pd.DataFrame({
#         "source_geonameId":inferred_src_gn_id_list,
#         "source_lon":inferred_src_lon_list,
#         "source_lat":inferred_src_lat_list,
#         "source_geonameId":inferred_src_gn_id_list,
#         "source_name":inferred_src_name_list,
#         "source_country":inferred_src_country_list,
#         "target_geonameId":inferred_trgt_gn_id_list,
#         "target_lon":inferred_trgt_lon_list,
#         "target_lat":inferred_trgt_lat_list,
#         "target_name":inferred_trgt_name_list,
#         "target_country":inferred_trgt_country_list,
#         "weight_alpha": inferred_w_a_list,
#         #"weight_beta": inferred_w_b_list
#     })
#     print(df_inferred_edgelist)
#
#     inferred_edgelist_filepath = os.path.join(out_folder, inferred_edgelist_filename)
#     df_inferred_edgelist.to_csv(inferred_edgelist_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)
#
#     construct_digraph_from_inferred_edgelist_single_layer_phase(graph_single_layer_filepath, df_inferred_edgelist)
#
#
#
#
# # TODO: add other node and edge attributes into graphml
# def construct_digraph_from_inferred_edgelist_single_layer_phase(graph_single_layer_filepath, df_inferred_edgelist, df_map_data=None):
#     graph = nx.DiGraph()
#
#     # create nodes
#     for index, row in df_inferred_edgelist.iterrows():
#         src_geonameId = row["source_geonameId"]
#         src_lat = row["source_lat"]
#         src_lon = row["source_lon"]
#         trgt_geonameId = row["target_geonameId"]
#         trgt_lat = row["target_lat"]
#         trgt_lon = row["target_lon"]
#         weight = row["weight_alpha"]
#
#         if not graph.has_node(src_geonameId):
#             graph.add_node(src_geonameId, size=1, lat=src_lat, lng=src_lon)
#         if not graph.has_node(trgt_geonameId):
#             graph.add_node(trgt_geonameId, size=1, lat=trgt_lat, lng=trgt_lon)
#         if not graph.has_edge(src_geonameId, trgt_geonameId):
#             graph.add_edge(src_geonameId, trgt_geonameId, weight=weight, index=index)
#
#     ## print(graph[1859133][1852553])
#     ## print(graph.has_edge(1859133, 1852552))
#     ## print(graph.edges(data=True))
#     nx.write_graphml(graph, graph_single_layer_filepath)
#     return graph
#
#
# # ---------------------------------------------------------------------------------
#
#
#
#
# # ======================================================================
# # MAIN METHOD: Inference
# # ======================================================================
#
# # # Spreading parameters
# # gamma = 2  # <RW> recovery rate in the SIR process
# # epsilon_max = 0  # <RW> maximum layer mixing
# # ratio = 3  # <W> cascade-edge ratio
#
#
# def perform_MultiC_single_layer_phase(preprocessed_events_filepath, date_start, date_end, cascades_info_filepath, \
#                    spatial_dist_matrix_filepath, genome_dist_matrix_filepath, world_map_filepath, world_map_shape_filepath,
#                                       out_folder, out_graph_single_layer_filename, out_dist_values_filepath, obj_func,
#                                       include_spatial_dist, include_genome_dist, include_epsilon, include_unactivated_nodes,
#                                       beta, gamma):
#     df_events = pd.read_csv(preprocessed_events_filepath, sep=";", keep_default_na=False)
#     df_events[consts.COL_PUBLISHED_TIME] = df_events[consts.COL_PUBLISHED_TIME].apply(lambda x: parser.parse(x))
#     df_events["hierarchy_data"] = df_events["hierarchy_data"].apply(lambda x: eval(x))
#
#
#     df_cascades = pd.read_csv(cascades_info_filepath, sep=";", keep_default_na=False)
#     cascade_list = df_cascades["cascade"].to_list()
#
#     map_shapefile_data = gpd.read_file(world_map_shape_filepath, encoding="utf-8")
#     map_shapefile_data = map_shapefile_data.to_crs(3857)
#
#     #map_info = read_map_shapefilepath(world_map_shapefilepath)
#     map_info = pd.read_csv(world_map_filepath, sep=";", keep_default_na=False)
#     print(world_map_filepath)
#     print(map_info.columns)
#     #id_list = map_info["gn_id"].to_numpy().flatten()
#     #print(id_list)
#
#
#     #map_info = map_info[map_info["gn_id"] != -1]
#     # TODO: in the map data, there are some -1 values for geonames id.
#     # But, If I exclude these values, we will have N != len(map). So, maybe remove in the end these -1 values as postprocessing
#         ## example:
#         ##id_list = map_info["gn_id"].to_numpy().flatten()
#         ##id2geonameid = dict(zip(range(N), id_list))
#         ##print(id2geonameid)
#         ##gn_id_list = [id2geonameid[i] for i in range(N)]
#
#
#     spatial_dist_matrix = pd.read_csv(spatial_dist_matrix_filepath, sep=";", keep_default_na=False, index_col=0, header=0).to_numpy()
#     np.fill_diagonal(spatial_dist_matrix, 2)
#
#     genome_dist_matrix = pd.read_csv(genome_dist_matrix_filepath, sep=";", keep_default_na=False, index_col=0, header=0).to_numpy()
#     #print(genome_dist_matrix.shape)
#     #print(spatial_dist_matrix.shape)
#
#     # map_info_data = map_info_data[map_info_data["gn_id"] != -1]
#     C = len(cascade_list)
#     N = spatial_dist_matrix.shape[0]  # distance matrix is square matrix, the values in each column/row represents all existing locations
#
#     sample_size = np.sum([len(c) for c in cascade_list]) # TODO, not sure on that
#
#     # optimization params
#     #max_iter = 200  # <W> maximum number of optimization iterations
#     max_iter = 5000  # <W> maximum number of optimization iterations
#     min_iter = 100  # minimum number of optimization iterations
#     learning_rate = 0.5  # initial learning rate of the Adam optimizer
#     tol = 0.0001  # threshold of relative objective value change for stopping the optimization
#     opt_params = {"max_iter": max_iter, "min_iter": min_iter, "learning_rate": learning_rate, "tol": tol}
#
#     succ_edges_list, succ_mask_idx, succ_scatter_idx, succ_delta_t, fail_edges_list, fail_mask_idx, fail_scatter_idx, fail_delta_t, delta_spatial_dist, delta_genome_dist, succ_node_idx = perform_data_parsing_and_tensor_construction_single_layer_phase(\
#                     df_events, date_start, date_end, cascade_list, spatial_dist_matrix, genome_dist_matrix, map_info, out_dist_values_filepath)
#     n_succ_edges = len(succ_edges_list)
#     n_fail_edges = len(fail_edges_list)
#     print(n_succ_edges, n_fail_edges)
#
#     params_g, lik_list = perform_optimization_single_layer_phase(obj_func, opt_params, N, C,
#                                                                  succ_mask_idx, succ_scatter_idx, succ_node_idx, succ_delta_t,
#                                                                  fail_mask_idx, fail_scatter_idx, fail_delta_t,
#                                                                  n_succ_edges, n_fail_edges,
#                                                                  delta_spatial_dist, delta_genome_dist,
#                                                                  include_spatial_dist, include_genome_dist, include_epsilon,
#                                                                  include_unactivated_nodes, beta, gamma)
#
#     #out_single_layer_folder = os.path.join(out_folder, 'inference_results_s_%d_%d_%d_%d' % (N, len(succ_edges_list), C, max_iter))
#     #os.makedirs(out_single_layer_folder, exist_ok=True)
#     raw_res_filename = 'raw_results.pkl'
#     inferred_edgelist_filename = "inference_edgelist.csv"
#
#     include_beta = False
#     #if beta is None:
#     #    include_beta = True
#     out_graph_single_layer_filepath = os.path.join(out_folder, out_graph_single_layer_filename)
#     postprocess_optimization_result_single_layer_phase(params_g, succ_edges_list, lik_list, sample_size, N, n_succ_edges, n_fail_edges,
#                                     out_folder, raw_res_filename, inferred_edgelist_filename,
#                                     map_info, out_graph_single_layer_filepath, include_epsilon,
#                                     include_unactivated_nodes, include_beta)
#
#     out_graph_plot_filepath = out_graph_single_layer_filepath.replace(".graphml", ".png")
#     plot_graph_on_map2(out_graph_single_layer_filepath, map_shapefile_data, out_graph_plot_filepath)
#
#
#
#
# def perform_MultiC_single_layer_phase_with_all_configs(preprocessed_events_filepath, out_folder, date_start, date_end, cascades_info_filepath, \
#                    spatial_dist_matrix_filepath, genome_dist_matrix_filepath, world_map_csv_filepath, world_map_shape_filepath,
#                                       out_graph_single_layer_filename, out_dist_values_filepath, infer_params_list, force=False):
#
#     start_time = time.time()
#
#     for infer_params in infer_params_list:
#         obj_func = infer_params["obj_func"]
#         include_spatial_dist = infer_params["include_spatial_dist"]
#         include_genome_dist = infer_params["include_genome_dist"]
#         include_epsilon = infer_params["include_epsilon"]
#         include_unactivated_nodes = infer_params["include_unactivated_nodes"]
#         beta = infer_params["beta"]
#         gamma = infer_params["gamma"]
#
#         spatial_str = str(include_spatial_dist)
#         if include_spatial_dist:
#             spatial_str = str(beta)
#         genome_str = str(include_genome_dist)
#         if include_genome_dist:
#             genome_str = str(gamma)
#         folder_params_prefix = "obj=" + obj_func + "_spatial="+spatial_str+"_genome="+genome_str+"_epsilon="+str(include_epsilon)+"_unactivated="+str(include_unactivated_nodes)
#
#         output_folder_all = os.path.join(out_folder, folder_params_prefix)
#         try:
#             if not os.path.exists(output_folder_all):
#                 os.makedirs(output_folder_all)
#         except OSError as err:
#             print(err)
#         print(output_folder_all)
#
#         if not os.path.exists(os.path.join(output_folder_all, out_graph_single_layer_filename)) or force:
#             perform_MultiC_single_layer_phase(preprocessed_events_filepath, date_start, date_end, cascades_info_filepath, \
#                                               spatial_dist_matrix_filepath, genome_dist_matrix_filepath,
#                                               world_map_csv_filepath, world_map_shape_filepath,
#                                               output_folder_all, out_graph_single_layer_filename, out_dist_values_filepath, obj_func,
#                                               include_spatial_dist, include_genome_dist, include_epsilon,
#                                               include_unactivated_nodes, beta, gamma)
#
#         map_shapefile_data = gpd.read_file(world_map_shape_filepath, encoding="utf-8")
#         map_shapefile_data = map_shapefile_data.to_crs(3857)
#         out_graph_single_layer_filepath = os.path.join(output_folder_all, out_graph_single_layer_filename)
#         out_graph_plot_filepath = out_graph_single_layer_filepath.replace(".graphml", ".png")
#         print("girdi")
#         plot_graph_on_map2(out_graph_single_layer_filepath, map_shapefile_data, out_graph_plot_filepath)
#
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     tot_exec_mins = elapsed_time / 60
#     print('Total execution time:', tot_exec_mins, 'minutes')
#     print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
#
#     # --------------------------------------
#
#     # clusters = set()
#     # for cl in df_events["disease_cluster"].tolist():
#     #     for c in cl:
#     #         clusters.add(c)
#     # nb_clusters = len(clusters)
#     #
#     # for cluster_id in range(nb_clusters):
#     #     #print("-- cluster id:", cluster_id)
#     #     df_events["disease_cluster_"+str(cluster_id)] = df_events_prep_upd["disease_cluster"].apply(lambda x: 1 if cluster_id in x else 0)
#     #     df_events_by_serotype = df_events[df_events_prep_upd["disease_cluster_"+str(cluster_id)] == 1].copy(deep=True)
#     #     del df_events_by_serotype["disease_cluster_"+str(cluster_id)]
#     #     del df_events["disease_cluster_" + str(cluster_id)]
#     #
#     #     serotype = None
#     #     for i in df_events_by_serotype["disease"].tolist():
#     #         serotype = build_disease_instance(i).get_disease_data()["serotype"]
#     #         if serotype != "unknown serotype":
#     #             break
#     #     if serotype in ["h5n1", "h5n5", "h5n8"]:
#     #         print("-- serotype:", serotype)
#     #
#     #         events_by_serotype_filepath = os.path.join(out_folder, "processed_empres-i_events_updated_serotype="+serotype+".csv")
#     #         df_events_by_serotype.to_csv(events_by_serotype_filepath, sep=";", index=False)
#     #
#     #         output_folder_by_serotype = os.path.join(out_folder, "serotype="+serotype, folder_params_prefix)
#     #
#     #         try:
#     #             if not os.path.exists(output_folder_by_serotype):
#     #                 os.makedirs(output_folder_by_serotype)
#     #         except OSError as err:
#     #             print(err)
#     #
#     #         out_graph_single_layer_filename = "single_layer_graph.graphml"
#     #         cascades_info_filepath = os.path.join(output_preprocessing_folder, "cascades", "cascades_disease=" + serotype + ".csv")
#     #         #cascades_info_filepath = os.path.join(output_preprocessing_folder, "cascades_from_flyway_movements_serotype=" + serotype + ".csv")
#     #         #cascades_info_filepath = os.path.join(output_preprocessing_folder, "cascades_from_st_clustering_disease=" + serotype + ".csv")
#     #
#     #         perform_MultiC_single_layer_phase(events_by_serotype_filepath, date_start, date_end, cascades_info_filepath, \
#     #                        spatial_dist_matrix_filepath, genome_dist_matrix_filepath,
#     #                                           world_map_csv_filepath, map_shapefile_data,
#     #                                           output_folder_by_serotype, out_graph_single_layer_filename, obj_func,
#     #                                           include_spatial_dist, include_genome_dist, include_epsilon,
#     #                                           include_unactivated_nodes, beta)
#
#
# # if __name__ == '__main__':
# #     print('Starting')
# #     #print("torch version:", torch.__version__)
# #     start_time = time.time()
# #
# #     output_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
# #     preprocessed_events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events.csv") # only 2021 data
# #     df_events_prep_upd = pd.read_csv(preprocessed_events_filepath, sep=";", keep_default_na=False)
# #     df_events_prep_upd["disease_cluster"] = df_events_prep_upd["disease_cluster"].apply(lambda x: eval(x))
# #     df_events_prep_upd["timestamp"] = df_events_prep_upd["timestamp"].apply(lambda x: x/24)
# #     print(np.max(df_events_prep_upd["timestamp"]))
# #
# #     in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
# #     world_map_shape_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1_with_fixed_geometries.shp")
# #     #world_map_shape_filepath = os.path.join(in_map_folder, "naturalearth_adm1_with_fixed_geometries_and_flyway.shp")
# #     #world_map_shape_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1.shp")
# #     #world_map_shape_filepath = os.path.join(in_map_folder, "naturalearth_adm1_with_fixed_geometries_and_flyway.shp")
# #
# #     #map_shapefile_data.loc[map_shapefile_data["flyway_inf"].isna(), "flyway_inf"] = "[]"
# #
# #     #map_shapefile_data["flyway_inf"] = map_shapefile_data["flyway_inf"].apply(lambda x: eval(x) if x is not None else [])
# #
# #
# #     include_spatial_dist = True
# #     include_genome_dist = True
# #     include_epsilon = False
# #     include_unactivated_nodes = True
# #     obj_func = "dist_survival"
# #     #obj_func = "spatial_survival"
# #     beta = 0.1
# #     #beta = None
# #     gamma = 0.1
# #
# #     #date_start = parser.parse("2016-12-31T00:00:00", dayfirst=False)
# #     #date_end = parser.parse("2017-07-01T00:00:00", dayfirst=False)  # ending time
# #     date_start = parser.parse("2020-12-31T00:00:00", dayfirst=False)
# #     date_end = parser.parse("2021-07-01T00:00:00", dayfirst=False)  # ending time
# #     ##date_end = parser.parse("2022-01-01T00:00:00", dayfirst=False)  # ending time
# #
# #
# #     in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
# #     world_map_csv_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1.csv")
# #     #world_map_csv_filepath = os.path.join(in_map_folder, "naturalearth_adm1_with_fixed_geometries_and_flyway.csv")
# #     spatial_dist_matrix_filepath = os.path.join(output_preprocessing_folder, "spatial_dist_matrix_from_map.csv")
# #     genome_dist_matrix_filepath = os.path.join(output_preprocessing_folder, "genome_dist_matrix_from_map.csv")
# #
# #
# #     out_graph_single_layer_filename = "single_layer_graph.graphml"
# #     #cascades_info_filepath = os.path.join(output_preprocessing_folder, "single_cascade", "single_cascade_by_serotype_and_flyway_intersection.csv")
# #     cascades_info_filepath = os.path.join(output_preprocessing_folder, "single_cascade", "cascades_flyway_and_st_clustering.csv")
# #     #cascades_info_filepath = os.path.join(output_preprocessing_folder, "single_cascade", "single_cascade.csv")
# #     #cascades_info_filepath = os.path.join(output_preprocessing_folder, "single_cascade", "single_cascade_by_serotype.csv")
# #     #cascades_info_filepath = os.path.join(output_preprocessing_folder, "single_cascade", "single_cascade_by_flyway.csv")
# #     # cascades_info_filepath = os.path.join(output_preprocessing_folder, "cascades_from_flyway_movements_serotype=" + serotype + ".csv")
# #     # cascades_info_filepath = os.path.join(output_preprocessing_folder, "cascades_from_st_clustering_disease=" + serotype + ".csv")
# #
# #     out_folder = consts.OUT_INFERENCE_EMPRESS_I_FOLDER + "_ver2"
# #     perform_MultiC_single_layer_phase_with_all_configs(preprocessed_events_filepath, out_folder, date_start, date_end, cascades_info_filepath, \
# #                                       spatial_dist_matrix_filepath, genome_dist_matrix_filepath,
# #                                       world_map_csv_filepath, world_map_shape_filepath, out_graph_single_layer_filename, obj_func,
# #                                       include_spatial_dist, include_genome_dist, include_epsilon, include_unactivated_nodes, beta, gamma)
#
