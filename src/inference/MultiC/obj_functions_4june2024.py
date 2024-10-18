

import torch
import numpy as np


# Conduct Inference
# Define objective function
def objective_single_layer_phase_with_dist_survival(params, mask_idx, scatter_idx, node_idx, N, C, n_succ_edges,
                                 delta_spatial_dist, delta_t, delta_genome_dist,
                                 include_spatial_dist=True, include_genome_dist=True, include_epsilon=True, beta=0.0001):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    #beta = 0.0005  # fixed for now >> for spatial dist
    gamma = 0.0001 # fixed for now >> for genome dist

    # #normalize = True
    # if normalize:
    #     delta_t = delta_t/360 # to normalize
    #     delta_spatial_dist = 1 / (1 + (delta_spatial_dist / 1000))

    alpha_p = torch.sigmoid(params[:n_succ_edges])
    if include_epsilon:
        eps_p = torch.sigmoid(params[(n_succ_edges):(n_succ_edges + N)])

    # ----------------------------------------------------
    # hazard function
    # ----------------------------------------------------
    alpha_p_adj = alpha_p # not including 'delta_spatial_dist' and 'delta_spatial_dist'
    if include_spatial_dist:
        print("!!!!!!!!!! SPATIAL")
        if not include_genome_dist:
            alpha_p_adj = beta * alpha_p * torch.exp(-beta * delta_spatial_dist)
        elif include_genome_dist:
            alpha_p_adj = gamma * beta * alpha_p * torch.exp(-beta * delta_spatial_dist) * torch.exp(-gamma * delta_genome_dist)

    hazard_alpha_p_adj_delta_t = alpha_p_adj * delta_t

    H = torch.zeros(N * C, device=device)  # hazard func
    for c in range(C):
        # mask_idx[c]: index of succ edges on cascade c
        H.scatter_add_(dim=0, index=scatter_idx[c], src=hazard_alpha_p_adj_delta_t.index_select(0, mask_idx[c]))  # dim 0: line-wise

    if include_epsilon:
        # postprocessing for unobserved source factor (multiplying each node j by its epsilon value)
        for c in range(C):
            # contribution from H_0 for unobserved source factor
            H.scatter_add_(dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]))  # dim 0: line-wise

    if include_epsilon:
        # postprocessing for unobserved source factor (multiplying each node j by its epsilon value)
        # since MPS does not support torch.scatter_reduce, this is a workaround
        S0_on_H = torch.zeros(N * C, device=device)  # hazard func
        for c in range(C):
            # contribution from S_0 for unobserved source factor
            S0_on_H.scatter_add_(dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]))
            #H = torch.scatter_reduce(H, dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]), reduce="prod", include_self=True)
        H = H*S0_on_H

    H_nonzero = H[H != 0]

    # ----------------------------------------------------
    # log survival function
    # ----------------------------------------------------
    #survival_alpha_p_adj_delta_t = alpha_p * 0.5 * (delta_t ** 2)  # survival func
    survival_alpha_p_adj_delta_t = alpha_p * 0.5 * (delta_t ** 1) # survival func
    if include_spatial_dist:
        if not include_genome_dist:
            survival_alpha_p_adj_delta_t = np.log((1/beta)) * survival_alpha_p_adj_delta_t
        elif include_genome_dist:
            survival_alpha_p_adj_delta_t = (1 / beta) * (1 / gamma) * survival_alpha_p_adj_delta_t

    S = torch.zeros(N * C, device=device)  # hazard func
    for c in range(C):
        # mask_idx[c]: index of succ edges on cascade c
        S.scatter_add_(dim=0, index=scatter_idx[c], src=survival_alpha_p_adj_delta_t.index_select(0, mask_idx[c]))  # dim 0: line-wise

    if include_epsilon:
        # postprocessing for unobserved source factor (multiplying each node j by its epsilon value)
        S0_on_S = torch.zeros(N * C, device=device)  # hazard func
        for c in range(C):
            # contribution from S_0 for unobserved source factor
            S0_on_S.scatter_add_(dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]))
            #S = torch.scatter_reduce(S, dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]), reduce="prod", include_self=True)
        S = S * S0_on_S

    #print(S[S!=0])
    #S = torch.clip(S, 0.000001, 1.0)

    return torch.sum(S) - torch.sum(torch.log(H_nonzero))


def objective_single_layer_phase_with_spatial_survival(params, mask_idx, scatter_idx, node_idx, N, C, n_succ_edges,
                                 delta_spatial_dist, delta_t, delta_genome_dist,
                                 include_spatial_dist=True, include_genome_dist=True, include_epsilon=True, beta=0.0001):
    #print("started objective_single_layer_phase")
    print("-------")

    #print("include_spatial_dist:", include_spatial_dist)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ##beta = -0.001 # fixed for now >> for spatial dist
    #beta = 0.05  # fixed for now >> for spatial dist
    #beta = 0.0005
    #gamma = -1  # fixed for now >> for genome dist
    gamma = 0.05

    # normalize = False
    # if normalize:
    #     delta_t = delta_t/360 # to normalize
    #     delta_spatial_dist = 1 / (1 + (delta_spatial_dist / 1000))
    alpha_p = torch.sigmoid(params[:n_succ_edges])
    #print("!! alpha_p", alpha_p)
    #print("!! delta_t", delta_t)
    #print("!! surv",  alpha_p * 0.5 * (delta_t))
    #print("!! delta_norm_spatial_dist", delta_norm_spatial_dist)
    #print("!! surv **2", alpha_p * 0.5 * (delta_t ** 2))


    if include_epsilon:
        eps_p = torch.sigmoid(params[(n_succ_edges):(n_succ_edges + N)])

    # ----------------------------------------------------
    # hazard function
    # ----------------------------------------------------
    alpha_p_adj = alpha_p # not including 'delta_spatial_dist' and 'delta_spatial_dist'
    if include_spatial_dist:
        print("!!!!!!!!!! SPATIAL")
        #delta_norm_spatial_dist = torch.sigmoid((torch.log(delta_spatial_dist / 1000)))
        if not include_genome_dist:
            #print("--!--")
            #print(delta_norm_spatial_dist)
            #print(torch.exp(-beta * delta_norm_spatial_dist))
            alpha_p_adj = alpha_p * torch.exp(-beta*delta_spatial_dist)
        elif include_genome_dist:
            alpha_p_adj = alpha_p * torch.exp((-beta*delta_spatial_dist) + (-gamma*delta_spatial_dist))

    hazard_alpha_p_adj_delta_t = alpha_p_adj * delta_t

    H = torch.zeros(N * C, device=device)  # hazard func
    for c in range(C):
        # mask_idx[c]: index of succ edges on cascade c
        H.scatter_add_(dim=0, index=scatter_idx[c], src=hazard_alpha_p_adj_delta_t.index_select(0, mask_idx[c]))  # dim 0: line-wise

    if include_epsilon:
        # postprocessing for unobserved source factor (multiplying each node j by its epsilon value)
        for c in range(C):
            # contribution from H_0 for unobserved source factor
            H.scatter_add_(dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]))  # dim 0: line-wise

    if include_epsilon:
        # postprocessing for unobserved source factor (multiplying each node j by its epsilon value)
        # since MPS does not support torch.scatter_reduce, this is a workaround
        S0_on_H = torch.zeros(N * C, device=device)  # hazard func
        for c in range(C):
            # contribution from S_0 for unobserved source factor
            S0_on_H.scatter_add_(dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]))
            #H = torch.scatter_reduce(H, dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]), reduce="prod", include_self=True)
        H = H*S0_on_H

    H_nonzero = H[H != 0]

    # ----------------------------------------------------
    # log survival function
    # ----------------------------------------------------
    survival_alpha_p_adj_delta_t = 0.5*alpha_p*delta_t  # survival func
    #survival_alpha_p_adj_delta_t = alpha_p * 0.5 * (delta_t ** 2) # survival func
    if include_spatial_dist:
        #delta_norm_spatial_dist = torch.sigmoid((torch.log(delta_spatial_dist / 1000)))
        if not include_genome_dist:
            survival_alpha_p_adj_delta_t = survival_alpha_p_adj_delta_t ** torch.exp(-beta * delta_spatial_dist)
        elif include_genome_dist:
            survival_alpha_p_adj_delta_t = survival_alpha_p_adj_delta_t ** torch.exp((-beta*delta_spatial_dist) + (-gamma*delta_genome_dist))

    #print("before S")
    #print(survival_alpha_p_adj_delta_t)
    S = torch.zeros(N * C, device=device)  # hazard func
    for c in range(C):
        # mask_idx[c]: index of succ edges on cascade c
        S.scatter_add_(dim=0, index=scatter_idx[c], src=survival_alpha_p_adj_delta_t.index_select(0, mask_idx[c]))  # dim 0: line-wise

    if include_epsilon:
        # postprocessing for unobserved source factor (multiplying each node j by its epsilon value)
        S0_on_S = torch.zeros(N * C, device=device)  # hazard func
        for c in range(C):
            # contribution from S_0 for unobserved source factor
            S0_on_S.scatter_add_(dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]))
            #S = torch.scatter_reduce(S, dim=0, index=scatter_idx[c], src=eps_p.index_select(0, node_idx[c]), reduce="prod", include_self=True)
        S = S * S0_on_S

    S = torch.clip(S, 0.000001, 1.0)
    #S = torch.clamp(S, 0.000001, 1.0)  # clip
    #H = torch.clamp(H, 0.000001, 1.0)

    #print("!! S")
    #S_nonzero = S[S!=0]
    #print(S_nonzero)
    #print("!! H")
    #print(H_nonzero)
    # torch.clamp(torch.sum(S) - torch.sum(torch.log(H_nonzero)), 0.0, 1.0)
    return torch.sum(S) - torch.sum(torch.log(H_nonzero))
