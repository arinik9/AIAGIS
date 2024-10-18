import numpy as np
from model import MyModel
from utils import LoadGraph, BCELoss, MAE
import time
import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import random
import uuid
import os
import pandas as pd
import src.consts as consts
import csv

from src.util_event import build_disease_instance

from src.inference.MultiC.multic_s_NEW import construct_digraph_from_inferred_edgelist_single_layer_phase
from src.plot.plot_graph import plot_graph_on_map2

from src.preprocessing.create_cascades_nodes_matrix import create_cascades_nodes_matrix

import geopandas as gpd

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#device = "cpu"



def train(model, batch_x, SlotNum, optimizer, tau, l1_lambda, norm, act_fn):
    model.train()
    time_epoch = 0 
    loss_train = 0
    t1 = time.time()        
    optimizer.zero_grad()
    output = batch_x[:,0,:]
    for slot in range(1,SlotNum):                        
        output = model(output, tau)
        loss_train += BCELoss(output, batch_x[:,slot,:])
    loss_train += l1_lambda*torch.norm(model.weight.data, 1)
    loss_train.backward()     
    nn.utils.clip_grad_norm_(model.parameters(), norm)
    optimizer.step()    
    act_fn(model.weight.data)    
    time_epoch += (time.time()-t1)
    return model, loss_train.item(), time_epoch

def validate(model, ValiData, SlotNum, tau):
    model.eval()
    micro_val = 0
    mae=0
    with torch.no_grad(): 
        output=ValiData[:,0,:]
        for slot in range(1,SlotNum):
            output = model(output, tau)
            micro_val += BCELoss(output, ValiData[:,slot,:], False)
            mae+=MAE(output, ValiData[:,slot,:])
            #print("--",slot, micro_val)
        #print(micro_val)
        return micro_val, mae

def test(model, checkpt_file, TestData, SlotNum, tau):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    micro_val = 0
    mae=0
    with torch.no_grad():
        output=TestData[:,0,:]
        for slot in range(1,SlotNum):
            output = model(output, tau)
            micro_val += BCELoss(output, TestData[:,slot,:]) 
            mae+=MAE(output, TestData[:,slot,:]) 
        return micro_val, mae







# def inference_FIM(in_folder, out_folder, time_horizon, tau=1, seed=25190, batch=50, steps=2500, norm=2.0, l1_lambda=0.001,
#                   patience=1000, lr=0.001, train_per=0.8, val_per=0.1, dev=1):
def inference_FIM(in_folder, out_folder, time_horizon, tau=1, seed=25190, batch=50, steps=500, norm=2.0, l1_lambda=0.001,
                  patience=200, lr=0.001, train_per=0.8, val_per=0.1, dev=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)

    SlotNum = int(time_horizon / tau + 1)

    #Load Graph
    #cascades_time_filepath = os.path.join(in_folder, "cascades_time.npy")
    TrainData, ValiData, TestData=LoadGraph(in_folder, tau, time_horizon, train_per, val_per)
    TrainSize=TrainData.shape[0]
    nNodes=TrainData.shape[2]

    #### initial to torch
    TrainData = torch.FloatTensor(TrainData)
    ValiData = torch.FloatTensor(ValiData)
    TestData = torch.FloatTensor(TestData)
    print(TrainData.shape)

    # Model and Optimizer
    model=MyModel(nNodes=nNodes)
    optimizer = optim.Adam(model.parameters(), lr=lr) #weight_decay=args.weight_decay
    act_fn = nn.ReLU(inplace=True)

    #immigrate labels into GPU
    TrainData = TrainData.to(device)
    ValiData = ValiData.to(device)
    TestData = TestData.to(device)

    torch_dataset = Data.TensorDataset(TrainData)
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=batch,shuffle=False,num_workers=0)  # cannot shuffle

    train_time = 0
    train_time1 = 0
    bad_counter = 0
    best = 10000
    best_epoch = 0

    #for filename in glob.glob('*.pt'):
        #os.remove(filename)
    checkpt_file = os.path.join(out_folder, uuid.uuid4().hex+'.pt')

    for epoch in range(steps):
        start=(epoch*batch)%TrainSize
        end=min(start+batch, TrainSize)
        model, loss_tra,train_ep = train(model, TrainData[start:end], SlotNum, optimizer, tau, l1_lambda, norm, act_fn)
        f1_val, mae_val = validate(model, ValiData, SlotNum, tau)
        train_time += train_ep
        if epoch == 0:
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                '| val',
                'acc:{:.3f}'.format(f1_val),
                'mae:{:.3f}'.format(mae_val),
                '| cost:{:.3f}'.format(train_time))
        if(epoch+1)%100 == 0:
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                '| val',
                'acc:{:.3f}'.format(f1_val),
                'mae:{:.3f}'.format(mae_val),
                '| cost:{:.3f}'.format(train_time))
        if f1_val < best:
            best = f1_val
            best_epoch = epoch
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

    weight_savefile = os.path.join(out_folder, "weights.npy")
    weights =  model.weight.data.cpu() #dict(model.named_parameters())["weight"].data
    np.save(weight_savefile, weights)

    # import numpy as np
    # w = np.load("weights.npy")
    f1_test, mae_test = test(model, checkpt_file, TestData, SlotNum, tau)
    print("Train cost: {:.4f}s".format(train_time), 'and {:.4f}'.format(train_time1))
    print('Load {}th epoch'.format(best_epoch))
    print("Test f1:{:.3f}".format(f1_test), "Test mae:{:.3f}".format(mae_test))
    print("--------------------------")



def postprocessing_FIM(weight_savefile, threshold, df_world_map_info, inferred_edgelist_filepath, graph_single_layer_filepath,
                        map_data, output_plot_filepath):
    W = np.load(weight_savefile)
    #N = W.shape[0]

    N = map_data.shape[0]
    id_list = df_world_map_info_nz["gn_id"].to_numpy().flatten()
    id2geonameid = dict(zip(range(N), id_list))
    geonameid2name = dict(zip(id_list, df_world_map_info["name"]))
    geonameid2country = dict(zip(id_list, df_world_map_info["admin"]))
    geonameid2lon = dict(zip(id_list, df_world_map_info["lon"]))
    geonameid2lat = dict(zip(id_list, df_world_map_info["lat"]))

    # write edgelist into file
    inferred_src_gn_id_list = []
    inferred_src_lon_list = []
    inferred_src_lat_list = []
    inferred_src_name_list = []
    inferred_src_country_list = []
    inferred_trgt_gn_id_list = []
    inferred_trgt_lon_list = []
    inferred_trgt_lat_list = []
    inferred_trgt_name_list = []
    inferred_trgt_country_list = []
    inferred_w_a_list = []
    # inferred_w_b_list = []

    for i in range(N):
        for j in range(N):
            if i!=j and W[i,j] >= threshold:  # << threshold is determined by the previous step
                src_gn_id = id2geonameid[i]
                trgt_gn_id = id2geonameid[j]
                inferred_src_gn_id_list.append(src_gn_id)
                inferred_src_lon_list.append(geonameid2lon[src_gn_id])
                inferred_src_lat_list.append(geonameid2lat[src_gn_id])
                inferred_src_name_list.append(geonameid2name[src_gn_id])
                inferred_src_country_list.append(geonameid2country[src_gn_id])
                inferred_trgt_gn_id_list.append(trgt_gn_id)
                inferred_trgt_lon_list.append(geonameid2lon[trgt_gn_id])
                inferred_trgt_lat_list.append(geonameid2lat[trgt_gn_id])
                inferred_trgt_name_list.append(geonameid2name[trgt_gn_id])
                inferred_trgt_country_list.append(geonameid2country[trgt_gn_id])
                inferred_w_a_list.append(W[i,j])
                # inferred_w_b_list.append(B[new_idx])

    df_inferred_edgelist = pd.DataFrame({
        "source_geonameId": inferred_src_gn_id_list,
        "source_lon": inferred_src_lon_list,
        "source_lat": inferred_src_lat_list,
        "source_geonameId": inferred_src_gn_id_list,
        "source_name": inferred_src_name_list,
        "source_country": inferred_src_country_list,
        "target_geonameId": inferred_trgt_gn_id_list,
        "target_lon": inferred_trgt_lon_list,
        "target_lat": inferred_trgt_lat_list,
        "target_name": inferred_trgt_name_list,
        "target_country": inferred_trgt_country_list,
        "weight_alpha": inferred_w_a_list,
        # "weight_beta": inferred_w_b_list
    })
    print(df_inferred_edgelist)

    df_inferred_edgelist.to_csv(inferred_edgelist_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)

    construct_digraph_from_inferred_edgelist_single_layer_phase(graph_single_layer_filepath, df_inferred_edgelist)

    plot_graph_on_map2(graph_single_layer_filepath, map_data, output_plot_filepath)



def inference_FIM_with_all_configs(events_filepath, in_cascades_folder, out_preprocessing_folder,
                                   out_inference_folder, map_data,
                                   df_world_map_info_nz, inference_threshold):
    df_events = pd.read_csv(events_filepath, sep=";", keep_default_na=False)
    df_events["disease_cluster"] = df_events["disease_cluster"].apply(lambda x: eval(x))

    # # ==========================================================
    # # ALL
    # # ===========================================================
    #
    # out_preprocessing_all_folder = os.path.join(out_preprocessing_folder, "All")
    # out_all_folder = os.path.join(out_inference_folder, "All")
    # try:
    #     if not os.path.exists(out_preprocessing_all_folder):
    #         os.makedirs(out_preprocessing_all_folder)
    #     if not os.path.exists(out_all_folder):
    #         os.makedirs(out_all_folder)
    # except OSError as err:
    #     print(err)
    #
    # # PREPROCESSING
    # cascades_all_filepath = os.path.join(in_folder, "cascades", "cascades.csv")
    # create_cascades_nodes_matrix(events_filepath, cascades_all_filepath, df_world_map_info_nz, out_preprocessing_all_folder)
    #
    # # TRAINING
    # weight_savefile = os.path.join(out_all_folder, "weights.npy")
    # if not os.path.exists(weight_savefile):
    #     time_horizon = 30 # since the data from 2021 only
    #     inference_FIM(out_preprocessing_all_folder, out_all_folder, time_horizon, tau=1, seed=25190)
    #     # batch=50, steps=2500, norm=2.0, l1_lambda=0.001, patience=1000, lr=0.001, train_per=0.8, val_per=0.1, dev=1
    #
    # # PLOT
    # inferred_edgelist_filepath = os.path.join(out_all_folder, "inference_edgelist.csv")
    # graph_single_layer_filepath = os.path.join(out_all_folder, "single_layer_graph.graphml")
    # output_plot_filepath = graph_single_layer_filepath.replace(".graphml", ".png")
    # postprocessing_FIM(weight_savefile, inference_threshold, df_world_map_info_nz, inferred_edgelist_filepath, graph_single_layer_filepath,
    #                    map_data, output_plot_filepath)


    # ===========================================================
    # BY SEROTYPE
    # ==========================================================

    clusters = set()
    for cl in df_events["disease_cluster"].tolist():
        for c in cl:
            clusters.add(c)
    nb_clusters = len(clusters)

    for cluster_id in range(nb_clusters):
        print("-- cluster id:", cluster_id)
        df_events["disease_cluster_" + str(cluster_id)] = df_events["disease_cluster"].apply(
            lambda x: 1 if cluster_id in x else 0)
        df_events_by_serotype = df_events[df_events["disease_cluster_" + str(cluster_id)] == 1].copy(deep=True)
        del df_events_by_serotype["disease_cluster_" + str(cluster_id)]
        del df_events["disease_cluster_" + str(cluster_id)]

        serotype = None
        for i in df_events_by_serotype["disease"].tolist():
            serotype = build_disease_instance(i).get_disease_data()["serotype"]
            if serotype != "unknown serotype":
                break
        if serotype in ["h5n1", "h5n5", "h5n8"]:
            print("-- serotype:", serotype)

            out_preprocessing_folder_by_serotype = os.path.join(out_preprocessing_folder, "serotype=" + serotype)
            output_folder_by_serotype = os.path.join(out_inference_folder, "serotype=" + serotype)
            try:
                if not os.path.exists(out_preprocessing_folder_by_serotype):
                    os.makedirs(out_preprocessing_folder_by_serotype)
                if not os.path.exists(output_folder_by_serotype):
                    os.makedirs(output_folder_by_serotype)
            except OSError as err:
                print(err)

            events_by_serotype_filepath = os.path.join(output_folder_by_serotype,
                                                       "processed_empres-i_events_updated_serotype=" + serotype + ".csv")
            df_events_by_serotype.to_csv(events_by_serotype_filepath, sep=";", index=False)


            # PREPROCESSING
            cascades_by_serotype_filepath = os.path.join(in_cascades_folder, "cascades_disease=" + serotype + ".csv")
            create_cascades_nodes_matrix(events_by_serotype_filepath, cascades_by_serotype_filepath, df_world_map_info_nz,
                                         out_preprocessing_folder_by_serotype)

            # TRAINING
            weight_savefile = os.path.join(output_folder_by_serotype, "weights.npy")
            if not os.path.exists(weight_savefile):
                time_horizon = 30  # since the data from 2021 only
                inference_FIM(out_preprocessing_folder_by_serotype, output_folder_by_serotype, time_horizon, tau=1, seed=25190)
                # batch=50, steps=2500, norm=2.0, l1_lambda=0.001, patience=1000, lr=0.001, train_per=0.8, val_per=0.1, dev=1

            # PLOT
            inferred_edgelist_filepath = os.path.join(output_folder_by_serotype, "inference_edgelist.csv")
            graph_single_layer_filepath = os.path.join(output_folder_by_serotype, "single_layer_graph.graphml")
            output_plot_filepath = graph_single_layer_filepath.replace(".graphml", ".png")
            postprocessing_FIM(weight_savefile, inference_threshold, df_world_map_info_nz, inferred_edgelist_filepath,
                               graph_single_layer_filepath,
                               map_data, output_plot_filepath)


# # Training settings
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default="HR",help='Dataset to use.')
# parser.add_argument('--seed', type=int, default=25190, help='Random seed.')
# parser.add_argument('--steps', type=int, default=2500, help='Number of steps to train.')
# parser.add_argument('--patience', type=int, default=1000, help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
# parser.add_argument('--dev', type=int, default=1, help='device id')
# parser.add_argument('--batch', type=int, default=50, help='batch size')
# parser.add_argument('--tau', type=float, default=1, help='infinitesimal value')
# parser.add_argument('--train_per', type=float, default=0.8, help='percentage of training size')
# parser.add_argument('--val_per', type=float, default=0.1, help='percentage of validation size')
# parser.add_argument('--l1_lambda', type=float, default=0.001, help='l1 regularization coefficient')
# parser.add_argument('--norm', type=float, default=2.0, help='gradient clip norm')

# args = parser.parse_args()


if __name__ == '__main__':
    print('Starting')
    #print("torch version:", torch.__version__)
    start_time = time.time()

    in_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
    preprocessed_events_filepath = os.path.join(in_folder,
                                                "processed_empres-i_events_updated.csv")  # only 2021 data

    in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
    world_map_csv_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces",
                                          "naturalearth_adm1.csv")
    df_world_map_info = pd.read_csv(world_map_csv_filepath, usecols=["gn_id", "name", "admin", "lon", "lat"], sep=";",
                                    keep_default_na=False)
    # df_world_map_info_nz = df_world_map_info[df_world_map_info["gn_id"] != -1]
    df_world_map_info_nz = df_world_map_info

    world_map_shape_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1_with_fixed_geometries.shp")
    map_data = gpd.read_file(world_map_shape_filepath, encoding = "utf-8")
    map_data = map_data.to_crs(4326)

    in_cascades_folder = os.path.join(in_folder, "cascades")

    map_folder = consts.IN_MAP_SHAPEFILE_FOLDER

    out_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "preprocessing", "FIM")
    out_inference_folder = os.path.join(consts.OUT_INFERENCE_EMPRESS_I_FOLDER + "_ver2", "FIM")

    inference_threshold = 0.03
    # folder_params_prefix
    inference_FIM_with_all_configs(preprocessed_events_filepath, in_cascades_folder,
                                   out_preprocessing_folder, out_inference_folder, map_data,
                                   df_world_map_info_nz, inference_threshold)

    #
    # # ====================================================
    # # PREPROCESSING
    # # ====================================================
    # create_cascades_nodes_matrix(preprocessed_events_filepath, cascades_filepath, map_folder, cascades_folder)
    #
    #
    # # ====================================================
    # # TRAINING
    # # ====================================================
    # weight_savefile = os.path.join(out_folder, "weights.npy")
    # if not os.path.exists(weight_savefile):
    #     time_horizon = 30 # since the data from 2021 only
    #     inference_FIM(cascades_folder, out_folder, time_horizon, tau=1, seed=25190)
    #     # batch=50, steps=2500, norm=2.0, l1_lambda=0.001, patience=1000, lr=0.001, train_per=0.8, val_per=0.1, dev=1
    #
    #
    # # ====================================================
    # # PLOT
    # # ===================================================
    # threshold = 0.03
    # inferred_edgelist_filepath = os.path.join(out_folder, "inference_edgelist.csv")
    # graph_single_layer_filepath = os.path.join(out_folder, "single_layer_graph.graphml")
    # output_plot_filepath = graph_single_layer_filepath.replace(".graphml", ".png")
    # postprocessing_FIM(weight_savefile, threshold, df_world_map_info_nz, inferred_edgelist_filepath, graph_single_layer_filepath,
    #                    map_data, output_plot_filepath)
