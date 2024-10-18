

import torch
import numpy as np
import pandas as pd
#from src.truncated_normal import TruncatedNormal

# mu = torch.tensor([0.03], requires_grad=False)
# sigma = torch.tensor([0.1], requires_grad=False)
# x = torch.tensor([0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]).view(-1, 1, 1)
# print(Ad.log_prob(x).view(1, -1))
# tensor([[  1.8431,   1.8458,   1.8678,   1.8678,   1.6428,  -0.5322,  -9.1572,     -45.1572]])
# Ad = TruncatedNormal(loc=mu, scale=sigma, a=0, b=0.5) #  tanh_loc=False, upscale=20
# print(Ad.log_prob(x).view(1, -1))
# tensor([[  1.8204,   1.8230,   1.8451,   1.8451,   1.6201,  -0.5549,  -9.1799,     -45.1799]])
# x = torch.tensor([0.0001, 0.001, 0.01, 0.03, 0.1, 0.25, 0.5, 1]).view(-1, 1, 1)
# print(Ad.log_prob(x).view(1, -1))


# Conduct Inference
# Define objective function
def objective_single_layer_phase_with_dist_survival(params, N, C,
                                                    n_succ_edges, succ_mask_idx, succ_scatter_idx, succ_node_idx,
                                                    succ_tp_dict,
                                                    n_fail_edges, fail_mask_idx, fail_scatter_idx, fail_tp_dict,
                                                    dp, include_spatial_dist=True, include_epsilon=True,
                                                    include_genome_dist=False, succ_gp_dict=None,
                                                    include_unactivated_nodes=False,
                                                    b=0.0001, g=0.0001):

    # mask_idx[c]: index of succ edges on cascade c
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if include_unactivated_nodes == False:
        n_fail_edges = 0

    A_succ = torch.sigmoid(params[:n_succ_edges])
    A_fail = torch.sigmoid(params[(n_succ_edges):(n_succ_edges + n_fail_edges)])
    if include_epsilon:
        eps = torch.sigmoid(params[(n_succ_edges + n_fail_edges):(n_succ_edges + n_fail_edges + N)])
        if b is None:
            b = torch.sigmoid(params[(n_succ_edges + n_fail_edges + N):(n_succ_edges + n_fail_edges + N + 1)])
    else:
        print("epsilon ignored")
        if b is None:
            b = torch.sigmoid(params[(n_succ_edges + n_fail_edges):(n_succ_edges + n_fail_edges + 1)])

    # # =====================================================
    # # TEMPORARY
    # epsilon_filepath = "/Users/narinik/Mirror/workspace/MulNetFer/out/preprocessing/epsilon_scores_from_map.csv"
    # eps = pd.read_csv(epsilon_filepath, sep=";", keep_default_na=False)["uncertainty_score"].to_numpy()
    # #eps = np.random.rand(eps.shape[0])
    # print(eps)
    # eps = torch.LongTensor(eps).to(device)
    # include_epsilon = True
    # # =====================================================

    #max_val = torch.max(dp)
    #for c in range(C):
    #    succ_gp_dict[c] = succ_gp_dict[c]*max_val

    # ----------------------------------------------------
    # hazard
    # ----------------------------------------------------
    H_succ_edges = torch.zeros_like(A_succ)
    for c in range(C):
        if include_spatial_dist:
            H_succ_edges += b * A_succ * succ_tp_dict[c] * torch.exp(-b * dp)
            if include_genome_dist:
                H_succ_edges += g * b * A_succ * succ_tp_dict[c] * torch.exp(-b * dp) * torch.exp(-g * succ_gp_dict[c])
        else:
            if include_genome_dist:
                H_succ_edges += g * A_succ * succ_tp_dict[c] * torch.exp(-g * succ_gp_dict[c])
            else: # time only
                H_succ_edges += A_succ * succ_tp_dict[c]


    H_succ = torch.zeros(N * C, device=device)  # hazard func
    # TODO: we add epsilon to H for each cascade, or only once ?
    for c in range(C):
        H_succ.scatter_add_(dim=0, index=succ_scatter_idx[c], src=H_succ_edges.index_select(0, succ_mask_idx[c]))  # dim 0: line-wise
        if include_epsilon:
            # postprocessing for unobserved source factor, see this page for more info: https://github.com/IzzyRou/spatial_rcs/blob/main/function_genRC.R#L198
            #node_idx = torch.LongTensor([i * C + c for i in range(N)]).to(device)
            #H_succ.scatter_add_(dim=0, index=node_idx, src=eps/C)  # dim 0: line-wise
            H_succ.scatter_add_(dim=0, index=succ_scatter_idx[c], src=(eps/C).index_select(0, succ_node_idx[c]))  # dim 0: line-wise
    H_succ_nonzero = H_succ[H_succ != 0]
    nll1 = torch.sum(torch.log(H_succ_nonzero))

    # ----------------------------------------------------
    # log survival for activated nodes, i.e. succ edges
    # ----------------------------------------------------
    S_succ_edges = torch.zeros_like(A_succ)
    for c in range(C):
        if include_spatial_dist:
            if torch.is_tensor(b):
                S_succ_edges += -0.5 * A_succ * succ_tp_dict[c] * succ_tp_dict[c] - torch.log(b)
            else:
                S_succ_edges += -0.5 * A_succ * succ_tp_dict[c] * succ_tp_dict[c] - np.log(b)
            #
            if include_genome_dist:
                if torch.is_tensor(g):
                    S_succ_edges += -0.5 * A_succ * succ_tp_dict[c] * succ_tp_dict[c] - torch.log(b) - torch.log(g)
                else:
                    S_succ_edges += -0.5 * A_succ * succ_tp_dict[c] * succ_tp_dict[c] - np.log(b) - np.log(g)
        else:
            if include_genome_dist:
                if torch.is_tensor(g):
                    S_succ_edges += -0.5 * A_succ * succ_tp_dict[c] * succ_tp_dict[c] - torch.log(g)
                else:
                    S_succ_edges += -0.5 * A_succ * succ_tp_dict[c] * succ_tp_dict[c] - np.log(g)
            else: # time only
                S_succ_edges += -0.5 * A_succ * succ_tp_dict[c] * succ_tp_dict[c]  # survival func

    S_succ = torch.zeros(N * C, device=device)  # hazard func
    # TODO: we substract epsilon from S for each cascade, or only once ?
    for c in range(C):
        S_succ.scatter_add_(dim=0, index=succ_scatter_idx[c], src=S_succ_edges.index_select(0, succ_mask_idx[c]))  # dim 0: line-wise
        if include_epsilon:
            # postprocessing for unobserved source factor, see this page for more info: https://github.com/IzzyRou/spatial_rcs/blob/main/function_genRC.R#L202
            #node_idx = torch.LongTensor([i * C + c for i in range(N)]).to(device)
            #S_succ.scatter_add_(dim=0, index=node_idx, src=-eps/C)  # dim 0: line-wise
            S_succ.scatter_add_(dim=0, index=succ_scatter_idx[c], src=(-eps/C).index_select(0, succ_node_idx[c]))  # dim 0: line-wise
    nll2 = torch.sum(S_succ)

    nll = -(nll1 + nll2)

    # ----------------------------------------------------
    # log survival for unactivated nodes, i.e. fail edges
    # ----------------------------------------------------
    if include_unactivated_nodes == True:
        S_fail_edges = torch.zeros_like(A_fail)
        for c in range(C):
            S_fail_edges += -0.5 * A_fail * fail_tp_dict[c] * fail_tp_dict[c] # survival func
            if include_spatial_dist:
                if torch.is_tensor(b):
                    S_fail_edges += -0.5 * A_fail * fail_tp_dict[c] * fail_tp_dict[c] - torch.log(b)
                else:
                    S_fail_edges += -0.5 * A_fail * fail_tp_dict[c] * fail_tp_dict[c] - np.log(b)
                #
                if include_genome_dist:
                    if torch.is_tensor(g):
                        S_fail_edges += -0.5 * A_fail * fail_tp_dict[c] * fail_tp_dict[c] - torch.log(b) - torch.log(g)
                    else:
                        S_fail_edges += -0.5 * A_fail * fail_tp_dict[c] * fail_tp_dict[c] - np.log(b) - np.log(g)
            else:
                if include_genome_dist:
                    if torch.is_tensor(g):
                        S_fail_edges += -0.5 * A_fail * fail_tp_dict[c] * fail_tp_dict[c] - torch.log(g)
                    else:
                        S_fail_edges += -0.5 * A_fail * fail_tp_dict[c] * fail_tp_dict[c] - np.log(g)

        S_fail = torch.zeros(N * C, device=device)  # hazard func
        for c in range(C):
            S_fail.scatter_add_(dim=0, index=fail_scatter_idx[c], src=S_fail_edges.index_select(0, fail_mask_idx[c]))  # dim 0: line-wise
        nll3 = torch.sum(S_fail)
        nll = -(nll1 + nll2 + nll3) # negation for neg log likelihood


    # # # ----------------------------------------------------
    # # # prior
    # # # ----------------------------------------------------
    # # see: https://github.com/IzzyRou/spatial_rcs/blob/main/function_genRC.R#L105
    # # also see: https://github.com/IzzyRou/spatial_rcs/blob/main/function_genRC.R#L207
    # mu = torch.tensor([0.003], requires_grad=False, device=device)
    # sigma = torch.tensor([0.1], requires_grad=False, device=device)
    # Ad = TruncatedNormal(loc=mu, scale=sigma, a=0, b=0.01)
    # #print(A_succ)
    # prior_A = torch.sum(Ad.log_prob(torch.clip(A_succ, 0, 0.01)).view(1, -1))
    # # #
    # # # print(eps)
    # mu = torch.tensor([0.01], requires_grad=False, device=device)
    # sigma = torch.tensor([0.1], requires_grad=False, device=device)
    # Ed = TruncatedNormal(loc=mu, scale=sigma, a=0.000001, b=0.1)
    # prior_E = torch.sum(Ed.log_prob(torch.clip(eps, 0.000001, 0.1)).view(1, -1))
    # #
    # prior = -(prior_A + prior_E)
    # #prior = -(prior_E)
    # return nll+prior

    return nll


def objective_single_layer_phase_with_spatial_survival(params, mask_idx, scatter_idx, node_idx, N, C, n_succ_edges,
                                 delta_spatial_dist, delta_t, delta_genome_dist,
                                 include_spatial_dist=True, include_genome_dist=True, include_epsilon=True, beta=0.0001):
    pass

# def objective_single_layer_phase_with_spatial_survival(params, mask_idx, scatter_idx, node_idx, N, C, n_succ_edges,
#                                  delta_spatial_dist, delta_t, delta_genome_dist,
#                                  include_spatial_dist=True, include_genome_dist=True, include_epsilon=True, beta=0.0001):
#     #print("started objective_single_layer_phase")
#     print("-------")
#
#     #print("include_spatial_dist:", include_spatial_dist)
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     ##beta = -0.001 # fixed for now >> for spatial dist
#     #beta = 0.05  # fixed for now >> for spatial dist
#     #beta = 0.0005
#     #gamma = -1  # fixed for now >> for genome dist
#     gamma = 0.05
#
#     # normalize = False
#     # if normalize:
#     #     delta_t = delta_t/360 # to normalize
#     #     delta_spatial_dist = 1 / (1 + (delta_spatial_dist / 1000))
#     alpha_p = torch.sigmoid(params[:n_succ_edges])
#     #print("!! alpha_p", alpha_p)
#     #print("!! delta_t", delta_t)
#     #print("!! surv",  alpha_p * 0.5 * (delta_t))
#     #print("!! delta_norm_spatial_dist", delta_norm_spatial_dist)
#     #print("!! surv **2", alpha_p * 0.5 * (delta_t ** 2))
#
#
#     if include_epsilon:
#         eps_p = torch.sigmoid(params[(n_succ_edges):(n_succ_edges + N)])
#
#     # ----------------------------------------------------
#     # hazard function
#     # ----------------------------------------------------
#     alpha_p_adj = alpha_p # not including 'delta_spatial_dist' and 'delta_spatial_dist'
#     if include_spatial_dist:
#         #delta_norm_spatial_dist = torch.sigmoid((torch.log(delta_spatial_dist / 1000)))
#         if not include_genome_dist:
#             #print("--!--")
#             #print(delta_norm_spatial_dist)
#             #print(torch.exp(-beta * delta_norm_spatial_dist))
#             alpha_p_adj = alpha_p * torch.exp(-beta*delta_spatial_dist)
#         elif include_genome_dist:
#             alpha_p_adj = alpha_p * torch.exp((-beta*delta_spatial_dist) + (-gamma*delta_spatial_dist))
#
#     hazard_alpha_p_adj_delta_t = alpha_p_adj * delta_t
#
#     H = torch.zeros(N * C, device=device)  # hazard func
#     for c in range(C):
#         # mask_idx[c]: index of succ edges on cascade c
#         H.scatter_add_(dim=0, index=scatter_idx[c], src=hazard_alpha_p_adj_delta_t.index_select(0, mask_idx[c]))  # dim 0: line-wise
#
#     if include_epsilon:
#         # postprocessing for unobserved source factor (multiplying each node j by its epsilon value)
#         for c in range(C):
#             # contribution from H_0 for unobserved source factor
#             H.scatter_add_(dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]))  # dim 0: line-wise
#
#     if include_epsilon:
#         # postprocessing for unobserved source factor (multiplying each node j by its epsilon value)
#         # since MPS does not support torch.scatter_reduce, this is a workaround
#         S0_on_H = torch.zeros(N * C, device=device)  # hazard func
#         for c in range(C):
#             # contribution from S_0 for unobserved source factor
#             S0_on_H.scatter_add_(dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]))
#             #H = torch.scatter_reduce(H, dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]), reduce="prod", include_self=True)
#         H = H*S0_on_H
#
#     H_nonzero = H[H != 0]
#
#     # ----------------------------------------------------
#     # log survival function
#     # ----------------------------------------------------
#     survival_alpha_p_adj_delta_t = 0.5*alpha_p*delta_t  # survival func
#     #survival_alpha_p_adj_delta_t = alpha_p * 0.5 * (delta_t ** 2) # survival func
#     if include_spatial_dist:
#         #delta_norm_spatial_dist = torch.sigmoid((torch.log(delta_spatial_dist / 1000)))
#         if not include_genome_dist:
#             survival_alpha_p_adj_delta_t = survival_alpha_p_adj_delta_t ** torch.exp(-beta * delta_spatial_dist)
#         elif include_genome_dist:
#             survival_alpha_p_adj_delta_t = survival_alpha_p_adj_delta_t ** torch.exp((-beta*delta_spatial_dist) + (-gamma*delta_genome_dist))
#
#     #print("before S")
#     #print(survival_alpha_p_adj_delta_t)
#     S = torch.zeros(N * C, device=device)  # hazard func
#     for c in range(C):
#         # mask_idx[c]: index of succ edges on cascade c
#         S.scatter_add_(dim=0, index=scatter_idx[c], src=survival_alpha_p_adj_delta_t.index_select(0, mask_idx[c]))  # dim 0: line-wise
#
#     if include_epsilon:
#         # postprocessing for unobserved source factor (multiplying each node j by its epsilon value)
#         S0_on_S = torch.zeros(N * C, device=device)  # hazard func
#         for c in range(C):
#             # contribution from S_0 for unobserved source factor
#             S0_on_S.scatter_add_(dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]))
#             #S = torch.scatter_reduce(S, dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]), reduce="prod", include_self=True)
#         S = S * S0_on_S
#
#     S = torch.clip(S, 0.000001, 1.0)
#     #S = torch.clamp(S, 0.000001, 1.0)  # clip
#     #H = torch.clamp(H, 0.000001, 1.0)
#
#     #print("!! S")
#     #S_nonzero = S[S!=0]
#     #print(S_nonzero)
#     #print("!! H")
#     #print(H_nonzero)
#     # torch.clamp(torch.sum(S) - torch.sum(torch.log(H_nonzero)), 0.0, 1.0)
#     return torch.sum(S) - torch.sum(torch.log(H_nonzero))
