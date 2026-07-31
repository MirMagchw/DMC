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
from matplotlib import pyplot as plt

import ast, os
from tools import sort_wav

torch.multiprocessing.set_sharing_strategy('file_system')

def reorder_labels_and_memberships(labels, memberships):
    labels = np.asarray(labels)
    memberships = np.asarray(memberships)

    label_order = []
    label_map = {}
    for label in labels:
        if label not in label_map:
            label_map[label] = len(label_order)
            label_order.append(label)

    reordered_labels = np.array([label_map[label] for label in labels], dtype=int)

    if memberships.ndim == 2 and memberships.shape[1] > 0:
        ordered_columns = [int(label) for label in label_order if int(label) < memberships.shape[1]]
        remaining_columns = [column for column in range(memberships.shape[1]) if column not in ordered_columns]
        column_order = ordered_columns + remaining_columns
        reordered_memberships = memberships[:, column_order]
    else:
        reordered_memberships = memberships

    return reordered_labels, reordered_memberships

def get_cluster(model_use, cluster_method, ref_choice, n_speaker, snr, t60, **kwargs):
    model = None
    count_method = None
    if model_use == 'ECAPAModel':
        model = ECAPAModel(**vars(ECAPA_opt))
        model.load_parameters('path_to_model/ECAPA.model')
        get_embedding = model.compute_ECAPAembedding_batch
    elif model_use == "ECAPA_CNNModel":
        model = ECAPA_CNNModel(**vars(ECAPA_opt))
        model.load_parameters('path_to_model/ECAPA_CNN.model')
        get_embedding = model.compute_ECAPAembedding_batch
    elif model_use == "CRNN_featureModel":
        model = CRNN_featureModel(**vars(CRNN_feature_opt))
        model.load_parameters('path_to_model/CRNN_feature.model')
        get_embedding = model.compute_CRNNembedding_batch

    if cluster_method == "SpkCount":
        count_model = CRNNModel(**vars(CRNN_opt))
        count_model.load_parameters('path_to_model/CRNN.model')
        count_method = count_model.count_batch
    
    if model == None:
        print("No SpkEm Model!")
    print(n_speaker)
    directory = f'./wavs/simulation_{n_speaker}_{snr}_{t60}'
    location_results = pd.read_csv(f'csv_files/location_results_{n_speaker}_{snr}_{t60}.csv')

    unique_scenes = location_results['room_scene'].unique()
    all_clustering_results = {}

    for k_scene, scene in enumerate(unique_scenes, start=1):
        scene_data = location_results.loc[location_results['room_scene'] == scene].copy()
        scene_data['distance'] = scene_data['distance'].apply(ast.literal_eval)

        distance = np.array(scene_data['distance'].tolist())
        critical_distance = np.array(scene_data['critical_distance'].tolist())[0]
        
        mic_num, source_num = distance.shape

        files = [f for f in os.listdir(directory) if f.startswith(scene) and f.endswith('.wav')]
        files = sorted(files, key=sort_wav)
        file_paths = [os.path.join(directory, file) for file in files]
        
        embeddings_list = []
        count_score = []
        signal_list = []
        n_clusters_target = n_speaker + 1
        score_array = np.zeros((mic_num, 5))

        if(model!=None):
            embeddings_list = get_embedding(file_paths)
            if count_method != None:
                count_score = count_method(file_paths)

            if count_method != None:
                score_array = np.array(count_score)
                k_array = np.argmax(score_array, axis=1)
                counts = np.bincount(k_array)
                count_max = 0
                for index, count in enumerate(counts):
                    if count > 0:
                        count_max = count
                        k = index

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

        labels, memberships = reorder_labels_and_memberships(labels, memberships)

        all_clustering_results[scene] = {
            'labels': labels,
            'memberships': memberships,
            'scores':score_array
        }

    results_list = []

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

