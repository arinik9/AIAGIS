import os
import pandas as pd
import csv
import numpy as np
import dateutil.parser as parser

#from scipy.stats import rayleigh
from itertools import permutations

import networkx as nx
import random
import time
import torch

import src.consts as consts

from src.inference.MultiC.multic_s import prepare_non_zero_cascade_data


# input:
#  - N: nb nodes
#  - C: nb cascades
#
# output:
#  - time_dict (keys: edge indexes with i*N+j, values: delta T)
#  - mask_dict (keys: index of each cascade c, values: edge indexes of cascade c)
#  - succ_edges_list: [edge idx1, edge idx2, ...]
#  - succ_edges_dict: (key: init edge idx, value: new edge idx) >> example: {edge idx1: 0, edge idx2: 1, ..., edge idx ?: nb succ edges}
def construct_tensors_from_nonzero_cascades_multilayer_phase(data_nz, graph, D, N, C, T_end, map_info_data):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    N_infer = graph.number_of_nodes()
    E_infer = graph.number_of_edges()
    print("N:", N, "N_infer:", N_infer, ", E:", E_infer)
    # C: nb cascades
    print(C)

    id_list = map_info_data["gn_id"].to_numpy().flatten()
    id2geonameid = dict(zip(range(N), id_list))
    geonameid2id = dict(zip(id_list, range(N)))

    mask_dict = {e: 0 for e in range(E_infer * C)}  # either 0 or 1
    time_dict = {e: 0 for e in range(E_infer * C)}
    dist_dict = {e: 0 for e in range(E_infer * C)}  # TODO: init with 0 is correct ?

    scatter_index_list = []
    succ_edges_by_cascade = {}

    start_time = time.time()
    for c in range(C):
        succ_edges_by_cascade[c] = []

        if c % 5 == 0 and c != 0:
            print('Finished parsing %d cascades' % c)
        sim_logs = data_nz[c]

        for e in graph.edges(data=True):  # it is normal to repeat this loop for each cascade
            geonameid_j = int(e[1])
            j = geonameid2id[geonameid_j]  # node idx
            scatter_index_list.append(j * C + c)

        succ_users = set()
        fail_users = set(range(N))

        for (t, j) in sim_logs:  # for each cascade c, it returns a list of tuples (t,j) stating that node j is infected at time t.
            geonameid_j = id2geonameid[j]
            if j in fail_users:
                if t > T_end:
                    break
                for (u, t_u) in succ_users:
                    geonameid_u = id2geonameid[u]
                    # e = N * u + j
                    # print(u,"->",j)
                    if graph.has_edge(str(geonameid_u), str(geonameid_j)):  # according to the result of the single layer phase
                        # print("has edge")
                        idx = c * E_infer + graph[str(geonameid_u)][str(geonameid_j)]["index"]
                        mask_dict[idx] = 1
                        time_dict[idx] += t - t_u # TODO: why '+=' ?
                        dist_dict[idx] = D[u, j]
                        #succ_edges_by_cascade[c].append("("+str(t)+","+str(u)+","+str(j)+")")
                        e_idx = N * u + j
                        succ_edges_by_cascade[c].append(graph[str(geonameid_u)][str(geonameid_j)]["index"])
                succ_users.add((j, t))
                fail_users.remove(j)

        for (j, t_j) in succ_users:
            for n in fail_users:
                # e = N * j + n
                if graph.has_edge(str(geonameid_u), str(geonameid_j)):
                    idx = c * E_infer + graph[str(geonameid_u)][str(geonameid_j)]["index"]
                    time_dict[idx] += T_end - t_j

    end_time = time.time()
    parse_time = end_time - start_time
    print('parse_time:', parse_time)

    start_time = time.time()
    mask = torch.tensor(list(mask_dict.values()), dtype=torch.uint8).view(C, E_infer).to(device)
    delta_t = torch.tensor(list(time_dict.values())).view(C, E_infer).to(device)
    delta_dist = torch.tensor(list(dist_dict.values()), dtype=torch.float32).view(C, E_infer).to(device)
    scatter_index = torch.tensor(scatter_index_list, dtype=torch.int64).to(device)
    end_time = time.time()
    tensor_time = end_time - start_time
    print('tensor_time:', tensor_time)

    print('Finished data parsing and tensor construction')
    return mask, delta_dist, delta_t, scatter_index, succ_edges_by_cascade




# Conduct Inference
# Define objective function
def objective_multilayer_phase(params, graph, N, C, K, mask, delta_dist, delta_t, scatter_index):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    N_infer = graph.number_of_nodes()
    E_infer = graph.number_of_edges()
    print("N:", N, "N_infer:", N_infer, ", E:", E_infer, ", K:", K, ", C:", C)

    alpha_p = torch.sigmoid(params[:(E_infer * K)]).view(K, E_infer) # >> K x E_infer
    beta_p = torch.sigmoid(params[(E_infer * K):(2 * E_infer * K)]).view(K, E_infer) # >> K x E_infer
    eps_p = torch.sigmoid(params[(2 * E_infer * K):(2 * E_infer * K + N)]).view(N) # >> 1 x N
    alpha_beta_p = alpha_p * beta_p

    pi_list = []
    for k in range(K - 1):
        pi_rm = 1
        for i in range(k):
            pi_rm -= pi_list[i]
        pi_list.append(torch.sigmoid(params[(2 * E_infer * K + N + C * k):(2 * E_infer * K + N + C * (k + 1))]) * pi_rm)
    pi_rm = 1
    for pi in pi_list:
        pi_rm -= pi
    pi_list.append(pi_rm) # for the last layer, i.e. layer K
    pi_p = torch.stack(pi_list, dim=-1).to(device) # -1 indicates the last dimension >> C x K

    alpha_beta_pi_prod = torch.matmul(pi_p, alpha_beta_p) # >> C x E_infer

    # hazard func
    H0 = eps_p

    #H = torch.zeros(N * C).to(device).scatter_add_(0, scatter_index, torch.flatten(mask * delta_t * alpha_beta_pi_prod))
    H = torch.zeros(N * C, dtype=torch.float32).to(device).scatter_add_(0, scatter_index, torch.flatten(mask * delta_t * alpha_beta_pi_prod))
    H_nonzero = H[H != 0]

    # survival func
    S0 = eps_p

    # S = delta_t * alpha_pi_prod # survival func
    S = 0.5 * (delta_t**2) * alpha_beta_pi_prod * delta_dist # survival func

    return torch.sum(S0) + torch.sum(S) - torch.sum(torch.log(H_nonzero)) - torch.sum(torch.log(H0))




def perform_optimization_multilayer_phase(opt_params, graph, N, C, K, mask, delta_dist, delta_t, scatter_index):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    N_infer = graph.number_of_nodes()
    E_infer = graph.number_of_edges()
    print("N:", N, "N_infer:", N_infer, ", E:", E_infer, ", K:", K, ", C:", C)

    # reproducibility purposes
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Optimization parameters
    max_iter = opt_params["max_iter"]
    min_iter = opt_params["min_iter"]
    learning_rate = opt_params["learning_rate"]
    tol = opt_params["tol"]

    print('Starting optimization')

    # Initialize parameters: we put all parameters into the same array to optimize them togethr in the Adam opt.

    params_init = np.random.uniform(-5, 5, size=(2 * E_infer * K + N + C * (K - 1)))
    params_g = torch.tensor(params_init, requires_grad=True, device=device, dtype=torch.float32)

    # Initialize optimizer
    opt = torch.optim.Adam([params_g], lr=learning_rate)

    infer_time = 0
    lik_list = [] # loss value list
    pi_acc_list = []
    orders = list(permutations(range(K))) # layer orders
    print("orders", orders)

    # Conduct optimization
    for it in range(max_iter):
        # Calculate objective
        start_time = time.time()
        loss = objective_multilayer_phase(params_g, graph, N, C, K, mask, delta_dist, delta_t, scatter_index)
        loss_val = loss.item()
        lik_list.append(loss_val)
        end_time = time.time()
        infer_time += end_time - start_time

        print('Iteration %d loss: %.4f' % (it + 1, loss_val))

        # Stop optimization when relative decrease in objective value lower than threshold
        if it > min_iter and len(lik_list) >= 2 and (lik_list[-2] - lik_list[-1]) / lik_list[-2] < tol:
            break

        # Loss propagation
        start_time = time.time()
        opt.zero_grad()
        loss.backward()
        opt.step()

    end_time = time.time()
    infer_time += end_time - start_time
    print("infer time:", infer_time)

    return params_g


def postprocess_optimization_result_multilayer_phase(params_g, graph, N, C, K,
                                    out_folder, inferred_cascade_layer_membership_filename):
    N_infer = graph.number_of_nodes()
    E_infer = graph.number_of_edges()
    print("N:", N, "N_infer:", N_infer, ", E:", E_infer, ", K:", K, ", C:", C)

    # alpha_p = torch.sigmoid(params[:(E_infer * K)]).view(K, E_infer) # >> K x E_infer

    # Parse inferred pi values
    alpha_inferred = torch.sigmoid(params_g[:(E_infer * K)]).view(K, E_infer).cpu().detach().numpy().T # >> K x E_infer
    beta_inferred = torch.sigmoid(params_g[(E_infer * K):(2 * E_infer * K)]).view(K, E_infer).cpu().detach().numpy().T
    eps_inferred = torch.sigmoid(params_g[(2 * E_infer * K):(2 * E_infer * K + N)]).view(N).cpu().detach().numpy().T
    pi_list = []
    for k in range(K - 1):
        pi_rm = 1
        for i in range(k):
            pi_rm -= pi_list[i]
        pi_list.append(torch.sigmoid(params_g[(E_infer * K + C * k):(E_infer * K + C * (k + 1))]).cpu().detach() * pi_rm)
    pi_rm = 1
    for pi in pi_list:
        pi_rm -= pi
    pi_list.append(pi_rm)
    pi_inferred = torch.stack(pi_list, dim=-1)

    # hangi cascade'in hangi layer'de oldugunu gosteriyo
    print(np.around(pi_inferred.numpy(), decimals=3))
    print(pi_inferred.shape)

    # when dim=0, get maximum index along columns with argmax
    # when dim=1, get maximum index along rows with argmax
    print(torch.argmax(pi_inferred, dim=0).numpy())
    print(torch.argmax(pi_inferred, dim=1).numpy())

    df = pd.DataFrame(np.around(pi_inferred.numpy(), decimals=3), columns = ['layer'+str(k) for k in range(K)])
    print(df.shape)

    inferred_cascade_layer_membership_filepath = os.path.join(out_folder, inferred_cascade_layer_membership_filename)
    df.to_csv(inferred_cascade_layer_membership_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)

    print('alpha')
    df_alpha = pd.DataFrame(np.around(alpha_inferred, decimals=3), columns=['layer' + str(k) for k in range(K)])
    print(df_alpha)
    inferred_alpha_filepath = os.path.join(out_folder, "inferred_alpha.csv")
    df_alpha.to_csv(inferred_alpha_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)

    print('beta')
    df_beta = pd.DataFrame(np.around(beta_inferred, decimals=3), columns=['layer' + str(k) for k in range(K)])
    print(df_beta)
    inferred_beta_filepath = os.path.join(out_folder, "inferred_beta.csv")
    df_beta.to_csv(inferred_beta_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)

    print('epsilon')
    df_eps = pd.DataFrame(np.around(eps_inferred, decimals=3), columns=['eps'])
    print(df_eps)
    inferred_eps_filepath = os.path.join(out_folder, "inferred_epsilon.csv")
    df_eps.to_csv(inferred_eps_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)



# Prepare for multilayer edge list
# for each layer, recognize cascades that spread on it and remove edges that are not activated by these cascades
def induce_final_edges_multilayer_phase(K, C, graph, succ_edges_by_cascade, \
                                        inferred_cascade_layer_membership_filepath, alpha_inferred_filepath,\
                                        out_folder, final_edges_filepath):
    start_time = time.time()

    cols = ['layer' + str(k) for k in range(K)]
    df_cascade_layer_membership = pd.read_csv(inferred_cascade_layer_membership_filepath, usecols= cols, sep=";", keep_default_na=False)
    pi_inferred = df_cascade_layer_membership.values
    # axis = 0 >> by column
    # axis = 1 >> by row
    pi_b = np.argmax(pi_inferred, axis=1)

    df_alpha_inferred = pd.read_csv(alpha_inferred_filepath, sep=";", keep_default_na=False)

    index2NodePair = {}
    for e in graph.edges(data=True):  # it is normal to repeat this loop for each cascade
        e_idx = e[2]['index']
        print(e[0], e[1], )
        index2NodePair[e_idx] = (e[0], e[1])

    N = graph.number_of_nodes()
    E_infer = graph.number_of_edges()
    print("N:", N, ", E:", E_infer, ", K:", K, ", C:", C)

    c_cluster = {k:set() for k in range(K)}
    for c in range(C): # cascade'lari K cluster'a ayiriyo
        c_cluster[pi_b[c]].add(c)

    source_list = []
    target_list = []
    layer_list = []
    weight_list = []
    for k in range(K):
        for c in c_cluster[k]:
            for e_idx in succ_edges_by_cascade[c]:
                i, j = index2NodePair[e_idx]
                w = df_alpha_inferred.loc[e_idx, 'layer' + str(k)]
                source_list.append(i)
                target_list.append(j)
                layer_list.append(k)
                weight_list.append(w)

    data = {'source': source_list, 'target': target_list, 'layer': layer_list, 'weight': weight_list}
    df = pd.DataFrame.from_dict(data)
    print(df.shape)
    final_edges_filepath = os.path.join(out_folder, final_edges_filepath)
    df.to_csv(final_edges_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)




# TODO
def construct_multilayer_graph_from_inferred_edgelist_multilayer_phase(params_g, succ_edges_list, lik_list, N, n_succ_edges,
                                    out_folder, raw_res_filename, inferred_edgelist_filename,
                                    map_info_data):
    pass





# K: nb desired layers in output
def perform_MultiC_multilayer_phase(preprocessed_events_filepath, graph_single_layer_filepath, date_start, date_end, cascades_info_filepath, \
                   spatial_dist_matrix_filepath, world_map_filepath, out_folder, K):
    out_folder = os.path.join(out_folder, "multilayer_phase")
    os.makedirs(out_folder, exist_ok=True)

    df_events = pd.read_csv(preprocessed_events_filepath, sep=";", keep_default_na=False)
    df_events[consts.COL_PUBLISHED_TIME] = df_events[consts.COL_PUBLISHED_TIME].apply(lambda x: parser.parse(x))
    df_events["hierarchy_data"] = df_events["hierarchy_data"].apply(lambda x: eval(x))

    df_cascades = pd.read_csv(cascades_info_filepath, sep=";", keep_default_na=False)
    cascade_list = df_cascades["cascade"].to_list()
    C = len(cascade_list)

    T_end = (date_end - date_start).total_seconds() // 3600

    dist_matrix = pd.read_csv(spatial_dist_matrix_filepath, sep=";", keep_default_na=False, index_col=0, header=0).to_numpy()
    np.fill_diagonal(dist_matrix, 2)
    N = dist_matrix.shape[0]  # distance matrix is square matrix, the values in each column/row represents all existing locations

    #map_info = read_map_shapefilepath(world_map_shapefilepath)
    map_info = pd.read_csv(world_map_filepath, sep=";", keep_default_na=False)
    map_info = map_info[map_info["gn_id"] != -1]

    # Read the graph, which is from the single layer inference results
    print(graph_single_layer_filepath)
    graph = nx.read_graphml(graph_single_layer_filepath)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        print("empty graph or only isolated nodes")
        return


    N_infer = graph.number_of_nodes()
    E_infer = graph.number_of_edges()
    print("N:", N, ", N_infer:", N_infer, ", E_infer:", E_infer, ", C:", C)

    # optimization params
    max_iter = 500  # <W> maximum number of optimization iterations
    min_iter = 100  # minimum number of optimization iterations
    learning_rate = 0.5  # initial learning rate of the Adam optimizer
    tol = 0.0001  # threshold of relative objective value change for stopping the optimization
    opt_params = {"max_iter": max_iter, "min_iter": min_iter, "learning_rate": learning_rate, "tol": tol}

    data_nz = prepare_non_zero_cascade_data(df_events, cascade_list, map_info, N)

    mask, delta_dist, delta_t, scatter_index, succ_edges_by_cascade = construct_tensors_from_nonzero_cascades_multilayer_phase(\
        data_nz, graph, dist_matrix, N, C, T_end, map_info)
    print(succ_edges_by_cascade)

    params_g = perform_optimization_multilayer_phase(opt_params, graph, N, C, K, mask, delta_dist, delta_t, scatter_index)

    inferred_layer_membership_filename = 'results_multilayer_cascade_membership_m_%d_%d_%d_%d.csv' % (N_infer, E_infer, K, C) # seed, s_max_iter
    postprocess_optimization_result_multilayer_phase(params_g, graph, N, C, K, out_folder, inferred_layer_membership_filename)

    inferred_layer_membership_filepath = os.path.join(out_folder, inferred_layer_membership_filename)
    final_edges_filepath = os.path.join(out_folder, "final_edges.csv")
    alpha_inferred_filepath = os.path.join(out_folder, "inferred_alpha.csv")
    induce_final_edges_multilayer_phase(K, C, graph, succ_edges_by_cascade, \
                                            inferred_layer_membership_filepath, alpha_inferred_filepath, \
                                            out_folder, final_edges_filepath)