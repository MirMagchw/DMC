import argparse, glob, os, torch, warnings, time
from models.tools import *
from models.ECAPAModel import ECAPAModel
from models.ECAPA_CNNModel import ECAPA_CNNModel
from models.CRNN_featureModel import CRNN_featureModel
from models.CRNNModel import CRNNModel
from config.config_ECAPA import ECAPA_opt
from config.config_CRNN import CRNN_opt
from config.config_CRNN_feature import CRNN_feature_opt
import soundfile as sf
import pandas as pd
import numpy as np
from fcmeans import FCM
from DSP.Co_NMF import NMF_clustering
from matplotlib import pyplot as plt
from DSP.Mod_MFCC import Mod_MFCC
import ast, os
from tools import sort_wav

room_num = 3
scenes = 10

torch.multiprocessing.set_sharing_strategy('file_system')

def get_cluster(model_use, cluster_method, ref_choice, n_speaker, snr, t60, **kwargs):
    if t60 == 666:
        rooms = [
            {'size': [9.3, 6.9, 4.9], 't60': 0.8}
        ]
    else:
        rooms = [
            {'size': [6.7, 4.9, 3.5], 't60': t60}
        ]
    model = None
    count_method = None
    if model_use == 'ECAPAModel':
        model = ECAPAModel(**vars(ECAPA_opt))
        model.load_parameters('exps/mytrain_ECAPA.model')
        get_embedding = model.compute_ECAPAembedding
    elif model_use == "ECAPA_CNNModel":
        model = ECAPA_CNNModel(**vars(ECAPA_opt))
        model.load_parameters('exps/mytrain_ECAPA_CNN.model')
        get_embedding = model.compute_ECAPAembedding
    elif model_use == "CRNN_featureModel":
        model = CRNN_featureModel(**vars(CRNN_feature_opt))
        model.load_parameters('exps/mytrain_CRNN_feature.model')
        get_embedding = model.compute_CRNNembedding
    elif model_use == 'ModMFCC':
        model = Mod_MFCC()
        get_embedding = model.get_Mod_MFCC
    if cluster_method == "SpkCount":
        count_model = CRNNModel(**vars(CRNN_opt))
        count_model.load_parameters('exps/CRNN.model')
        count_method = count_model.count

    if 'model' in locals():
        model.eval()
    if 'count_model' in locals():
        count_model.eval()
    
    if model == None:
        print("No SpkEm Model!")
    print(n_speaker)
    directory = f'./wavs/simulation_{n_speaker}_{snr}_{t60}'
    location_results = pd.read_csv(f'csv_files/location_results_{n_speaker}_{snr}_{t60}.csv')

    unique_scenes = location_results['room_scene'].unique()
    all_clustering_results = {}
    k_different = 0
    room_params = rooms[0]
    V = room_params['size'][0]*room_params['size'][1]*room_params['size'][2]
    T = room_params['t60']
    critical_distance = ((0.01/np.pi)*(V/T))**0.5
    for k_scene, scene in enumerate(unique_scenes, start=1):
        scene_data = location_results.loc[location_results['room_scene'] == scene].copy()
        scene_data['distance'] = scene_data['distance'].apply(ast.literal_eval)

        distance = np.array(scene_data['distance'].tolist())
        mic_num, source_num = distance.shape

        files = [f for f in os.listdir(directory) if f.startswith(scene) and f.endswith('.wav')]
        files = sorted(files, key=sort_wav)

        embeddings_list = []
        count_score = []
        signal_list = []
        n_clusters_target = n_speaker + 1
        score_array = np.zeros((mic_num, 5))

        for file in files:
            file_path = os.path.join(directory, file)
            audio, fs = sf.read(file_path)
            signal_list.append(audio)

        if(model!=None):
            for file in files:
                file_path = os.path.join(directory, file)
                embedding = get_embedding(file_path)
                embeddings_list.append(embedding)
                if count_method != None:
                    count_score.append(count_method(file_path))

            if count_method != None:
                score_array = np.array(count_score)
                k_array = np.argmax(score_array, axis=1)
                counts = np.bincount(k_array)
                count_max = 0
                for index, count in enumerate(counts):
                    if count > 0:
                        count_max = count
                        k = index
                if k != n_speaker:
                    k_different += 1
                    print(scene, k_array)
                n_clusters_target = k + 1
            embeddings_matrix = np.array(embeddings_list)
        memberships = np.zeros((mic_num, n_clusters_target))
        # clustering
        if cluster_method == "FCM" or cluster_method == "CountNet":
            fcm = FCM(
                n_clusters=n_clusters_target, 
                m=2,                
                max_iter=150,       
                error=1e-5,         
                random_state=42,   
                metric="euclidean", 
                init="kmeans++",    
                n_init=10           
            )
            fcm.fit(embeddings_matrix)
            labels = fcm.u.argmax(axis=1)
            memberships = fcm.u
        if model_use == "NMF":
            labels, B= NMF_clustering(directory, files, n_clusters_target-1)
            memberships = B
        elif model_use == "Distance":
            cluster_labels = np.full(mic_num, -1, dtype=int)
            current_cluster_label = 0
            for source_idx in range(source_num):
                distance_to_source = distance[:, source_idx]
                within_cri_dist = np.where(distance_to_source <= critical_distance)[0]
                if len(within_cri_dist) > 0:
                    cluster_labels[within_cri_dist] = current_cluster_label
                    current_cluster_label += 1
            remaining_mics = np.where(cluster_labels == -1)[0]
            if len(remaining_mics) > 0:
                cluster_labels[remaining_mics] = current_cluster_label
            labels = cluster_labels

        if np.min(labels) < 0:
            labels += 1

        all_clustering_results[scene] = {
            'labels': labels,
            'memberships': memberships,
            'scores':score_array
        }

    results_list = []
    print(f'ClusteringAcc: {(k_scene-k_different)/k_scene*100}')
    for key, result in all_clustering_results.items():
        room_scene = key
        labels = result['labels']
        memberships_ordered = result['memberships']
        scores = result['scores']
        for i, (label, membership_ordered, score) in enumerate(zip(labels, memberships_ordered, scores)):
            mic_idx = i 
            results_list.append({
                'room_scene': room_scene,
                'mic_idx': mic_idx,
                'cluster_label': label,
                'membership': membership_ordered.tolist(),
                'score':score.tolist()
            })

    results_df = pd.DataFrame(results_list)
    # save
    results_df.to_csv(f'csv_files/{model_use}_{cluster_method}_{ref_choice}_{n_speaker}_{snr}_{t60}_clustering_results.csv', index=False)

