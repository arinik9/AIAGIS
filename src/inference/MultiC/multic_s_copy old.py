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
#         data_nz[i] = list(zip(df_sub["timestamp"], df_sub["newid"]))  # (infection time, location id)
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
#     # Construct data tensors from nonzero cascades
#     time_dict = {e: 0.0 for e in range(N * N)}  # transmission edge'lere index atiyo hash yontemiyle, o yuzden init yaparak N*N yapiyo
#     mask_dict = {c: set() for c in range(C)}
#
#     succ_edges = set()
#     for c in range(C):
#         if c % 5 == 0 and c != 0:
#             print('Finished parsing %d cascades' % c)
#
#         sim_logs = data_nz[c]
#
#         succ_users = []
#         fail_users = set(range(N))
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
#                         mask_dict[c].add(idx)
#                         succ_edges.add(idx)
#                         time_dict[idx] += t - t_u # TODO: why '+=' ?
#                     else:
#                         break
#                 succ_users.insert(0, (j, t))  # node j is activated at time t on cascade c
#                 fail_users.remove(j)
#
#         for (j, t_j) in succ_users:  # for each activated node j on cascade c
#             # TODO: I dont understand why we update the activation time of the fail_users .. ?
#             for n in fail_users:
#                 idx = j * N + n
#                 time_dict[idx] += T_end - t_j  # TODO: check if these values are used later in the code !!
#
#     succ_edges_list = list(succ_edges)
#     return time_dict, mask_dict, succ_edges_list
#
#
# # N: nb nodes/locations
# # D: spatial distance matrix
# def construct_dist_list_for_succ_edges_single_layer_phase(N, D, succ_edges_list):
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
#         dist_list_for_succ_edges.append(dist)
#
#     # print(dist_for_succ_edges[:10])
#     return dist_list_for_succ_edges
#
#
#
# def construct_mask_and_scatter_idx_single_layer_phase(N, C, time_dict, mask_dict, succ_edges_list, dist_list_for_succ_edges):
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     overall_start_time = time.time()
#
#     succ_edges_dict = {k: v for v, k in enumerate(succ_edges_list)}  # {'a': 0, 'c': 1}
#
#     # Construct time tensor
#     time_val = np.array(list(time_dict.values()))[succ_edges_list]  # keep only successful activation times
#     delta_t = torch.FloatTensor(time_val).to(device)
#
#     # Construct dist tensor
#     dist_val = np.array(dist_list_for_succ_edges)  # only successful activation times
#     delta_dist = torch.FloatTensor(dist_val).to(device)
#
#     # Construct mask and scatter tensor
#     mask_idx = []
#     scatter_idx = []
#     succ_cnt = 0
#     for c in range(C):
#         mask_idx_list = []
#         scatter_idx_list = []
#         succ_cnt += len(mask_dict[c])
#         for old_idx in mask_dict[c]:  # succ edges on cascade c
#             # old_index hash value gibi calculate edildigi icin, o hash value'dan node index'leri geri buluyo
#             i = old_idx // N
#             j = old_idx % N
#             new_index = j * C + c  # bu index'te node i'nin onemi yok, cunku activated olan node j
#             # node j, sadece bir tane cascade'de de bulunabilir, ya da hepsinde
#             scatter_idx_list.append(new_index)
#             # succ_edges_dict = {8194: 0, 3: 1, 8196: 2, 5: 3, 8198: 4, 6 ... }
#             mask_idx_list.append(succ_edges_dict[old_idx])
#         mask_idx.append(torch.LongTensor(mask_idx_list).to(device))
#         scatter_idx.append(torch.LongTensor(scatter_idx_list).to(device))
#
#     return mask_idx, scatter_idx, delta_dist, delta_t
#
#
#
#
# def perform_data_parsing_and_tensor_construction_single_layer_phase(df_events, date_start, date_end, cascade_list, D, map_info_data):
#     # init
#     T_end = (date_end - date_start).total_seconds() // 3600
#     C = len(cascade_list)
#     N = D.shape[0]  # distance matrix is square matrix, the values in each column/row represents all existing locations
#     # df_map_nz = df_map[df_map["gn_id"] != -1]
#     # id_list = df_map_nz["gn_id"].to_numpy().flatten()
#     # N = len(id_list)
#
#     # processing
#     data_nz = prepare_non_zero_cascade_data(df_events, cascade_list, map_info_data, N)
#     time_dict, mask_dict, succ_edges_list = construct_tensors_from_nonzero_cascades_single_layer_phase(data_nz, N, C, T_end)
#     dist_list_for_succ_edges = construct_dist_list_for_succ_edges_single_layer_phase(N, D, succ_edges_list)
#     mask_idx, scatter_idx, delta_dist, delta_t = construct_mask_and_scatter_idx_single_layer_phase(N, C, time_dict, mask_dict, succ_edges_list, dist_list_for_succ_edges)
#     return mask_idx, scatter_idx, delta_dist, delta_t, succ_edges_list
#
#
# # ====================================================================================
#
# # ======================================================================
# # optimization
# # ======================================================================
#
# # Conduct Inference
# # Define objective function
# def objective_single_layer_phase(params, mask_idx, scatter_idx, N, C, n_succ_edges, delta_dist, delta_t):
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#
#     alpha_p = torch.sigmoid(params[:n_succ_edges])
#     beta_p = torch.sigmoid(params[(n_succ_edges):(2 * n_succ_edges)])
#     eps_p = torch.sigmoid(params[(2 * n_succ_edges):(2 * n_succ_edges + N)])
#
#     alpha_p_delta_t_beta_p = alpha_p * delta_t * beta_p
#
#     H0 = eps_p
#
#     H = torch.zeros(N * C, device=device)  # hazard func
#     for c in range(C):
#         # mask_idx[c]: index of succ edges on cascade c
#         # H.scatter_add_(dim=0, index=scatter_idx[c], src=alpha_p.index_select(0, mask_idx[c])) # dim 0: line-wise
#         # TODO: perform the log operation before doing the summation
#         H.scatter_add_(dim=0, index=scatter_idx[c],
#                        src=alpha_p_delta_t_beta_p.index_select(0, mask_idx[c]))  # dim 0: line-wise
#
#     H_nonzero = H[H != 0]
#
#     S0 = eps_p
#
#     S = 0.5 * (delta_t ** 2) * alpha_p * delta_dist * beta_p  # survival func
#
#     return torch.sum(S0) + torch.sum(S) - torch.sum(torch.log(H_nonzero)) - torch.sum(torch.log(H0))
#
#
#
# def perform_optimization_single_layer_phase(opt_params, mask_idx, scatter_idx, N, C, n_succ_edges, delta_dist, delta_t):
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
#     # Initialize parameters: we put all parameters into the same array to optimize them togethr in the Adam opt.
#     params_init = np.random.uniform(-5, 5, 2*n_succ_edges+N) # sigmoid'e koycagi icin boyle init yapiyo
#                                                         # sigmoid(-5) = 0.0067, sigmoid(5) = 0.9933
#     params_g = torch.tensor(params_init, requires_grad=True, device=device, dtype=torch.float)
#
#     # Initialize optimizer
#     opt = torch.optim.Adam([params_g], lr=learning_rate)
#
#     infer_time = 0
#     lik_list = []
#     # Conduct optimization
#     for it in range(max_iter):
#         # Calculate objective
#         start_time = time.time()
#         # ==========================
#         loss = objective_single_layer_phase(params_g, mask_idx, scatter_idx, N, C, n_succ_edges, delta_dist, delta_t)
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
# def postprocess_optimization_result_single_layer_phase(params_g, succ_edges_list, lik_list, N, n_succ_edges,
#                                     out_folder, raw_res_filename, inferred_edgelist_filename,
#                                     map_info_data, graph_single_layer_filepath):
#     alpha_inferred_nonzero = torch.sigmoid(params_g[:n_succ_edges]).cpu().detach().numpy()
#     beta_inferred_nonzero = torch.sigmoid(params_g[(n_succ_edges):(2 * n_succ_edges)]).cpu().detach().numpy()
#     eps_inferred_nonzero = torch.sigmoid(params_g[(2 * n_succ_edges):(2 * n_succ_edges + N)]).cpu().detach().numpy()
#
#     # Save results to file
#     res_dict = {}
#     res_dict['succ_edges_list'] = succ_edges_list
#     res_dict['alpha_inferred_nonzero'] = alpha_inferred_nonzero
#     res_dict['beta_inferred_nonzero'] = beta_inferred_nonzero
#     res_dict['eps_inferred_nonzero'] = eps_inferred_nonzero
#     res_dict['lik'] = lik_list
#     # res_dict['time'] = {'parse': parse_time, 'tensor': tensor_time, 'infer': infer_time, 'eval': eval_time}
#     if torch.cuda.is_available():
#         res_dict['memory'] = torch.torch.cuda.max_memory_allocated()
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
#     print(np.max(res_dict['alpha_inferred_nonzero']))
#     print(np.max(res_dict['beta_inferred_nonzero']))
#     print(np.max(res_dict['eps_inferred_nonzero']))
#
#
#     # write raw results into pickle file
#     raw_res_filepath = os.path.join(out_folder, raw_res_filename)
#     with open(raw_res_filepath, 'wb') as fp:
#         pickle.dump(res_dict, fp)
#
#     # Rank the edges by inferred edge weight in the single layer phase
#     E_ratio = 1.1
#     E_infer = int(E_ratio * N) # we expect slightly more than nb nodes
#     alpha_inferred_sorted = np.partition(alpha_inferred_nonzero, len(alpha_inferred_nonzero) - E_infer)
#     threshold = alpha_inferred_sorted[len(alpha_inferred_nonzero) - E_infer]
#     print("threshold:",threshold)
#     # edge weight threshold above which there are <E_infer> edges
#
#     # -----------------
#
#     A = np.array(res_dict['alpha_inferred_nonzero'])
#     A[A < 0.01] = None
#
#     B = np.array(res_dict['beta_inferred_nonzero'])
#     B[B < 0.01] = None
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
#     inferred_w_b_list = []
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
#             inferred_w_b_list.append(B[new_idx])
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
#         "weight_beta": inferred_w_b_list
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
# def construct_digraph_from_inferred_edgelist_single_layer_phase(graph_single_layer_filepath, df_inferred_edgelist,
#                                                                 df_map_data=None):
#
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
#                    spatial_dist_matrix_filepath, world_map_filepath, out_folder, out_graph_single_layer_filename):
#     df_events = pd.read_csv(preprocessed_events_filepath, sep=";", keep_default_na=False)
#     df_events[consts.COL_PUBLISHED_TIME] = df_events[consts.COL_PUBLISHED_TIME].apply(lambda x: parser.parse(x))
#     df_events["hierarchy_data"] = df_events["hierarchy_data"].apply(lambda x: eval(x))
#
#     df_cascades = pd.read_csv(cascades_info_filepath, sep=";", keep_default_na=False)
#     cascade_list = df_cascades["cascade"].to_list()
#
#     #map_info = read_map_shapefilepath(world_map_shapefilepath)
#     map_info = pd.read_csv(world_map_filepath, sep=";", keep_default_na=False)
#     map_info = map_info[map_info["gn_id"] != -1]
#
#     dist_matrix = pd.read_csv(spatial_dist_matrix_filepath, sep=";", keep_default_na=False, index_col=0, header=0).to_numpy()
#     np.fill_diagonal(dist_matrix, 2)
#
#     # map_info_data = map_info_data[map_info_data["gn_id"] != -1]
#     C = len(cascade_list)
#     N = dist_matrix.shape[0]  # distance matrix is square matrix, the values in each column/row represents all existing locations
#
#     # optimization params
#     max_iter = 500  # <W> maximum number of optimization iterations
#     min_iter = 100  # minimum number of optimization iterations
#     learning_rate = 0.5  # initial learning rate of the Adam optimizer
#     tol = 0.0001  # threshold of relative objective value change for stopping the optimization
#     opt_params = {"max_iter": max_iter, "min_iter": min_iter, "learning_rate": learning_rate, "tol": tol}
#
#     mask_idx, scatter_idx, delta_dist, delta_t, succ_edges_list = perform_data_parsing_and_tensor_construction_single_layer_phase(\
#                     df_events, date_start, date_end, cascade_list, dist_matrix, map_info)
#     n_succ_edges = len(succ_edges_list)
#
#     params_g, lik_list = perform_optimization_single_layer_phase(opt_params, mask_idx, scatter_idx, N, C, n_succ_edges, delta_dist, delta_t)
#
#     #out_single_layer_folder = os.path.join(out_folder, 'inference_results_s_%d_%d_%d_%d' % (N, len(succ_edges_list), C, max_iter))
#     #os.makedirs(out_single_layer_folder, exist_ok=True)
#     raw_res_filename = 'raw_results.pkl'
#     inferred_edgelist_filename = "inference_edgelist.csv"
#
#     out_graph_single_layer_filepath = os.path.join(out_folder, out_graph_single_layer_filename)
#     postprocess_optimization_result_single_layer_phase(params_g, succ_edges_list, lik_list, N, n_succ_edges,
#                                     out_folder, raw_res_filename, inferred_edgelist_filename,
#                                     map_info, out_graph_single_layer_filepath)
#
