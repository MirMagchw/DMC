import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast, os
def plot_clustered(model_use, cluster_method, ref_choice, n_speaker, snr, t60, **kwargs):

    directory = f'pictures/clustered_picture_{n_speaker}_{snr}_{t60}/{model_use}_{cluster_method}_{ref_choice}'
    #directory = f"pictures/quality_violinplot_png"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    clustering_results = pd.read_csv(f'./csv_files/{model_use}_{cluster_method}_{ref_choice}_{n_speaker}_{snr}_{t60}_clustering_results.csv')
    #map_results = pd.read_csv(f'./csv_files/map_results_{model_use}_{cluster_method}_{ref_choice}_{n_speaker}_{snr}_{t60}.csv')
    location_results = pd.read_csv(f'./csv_files/location_results_{n_speaker}_{snr}_{t60}.csv')

    merged_data = pd.merge(clustering_results, location_results, on=['room_scene', 'mic_idx'])

    unique_scenes = merged_data['room_scene'].unique()
    #unique_scenes = ['room05_scene04', 'room05_scene07']

    colors = ['blue', 'orange', 'brown', 'red', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    markers = ['o', 'o', 'o', 'o', 'o']
    faces = ['blue', 'orange', 'brown', 'red', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']

    for scene in unique_scenes:
        scene_data = merged_data[merged_data['room_scene'] == scene]
        # map_result = map_results.loc[map_results['room_scene'] == scene].copy()
        # map_result['cluster2source'] = map_result['cluster2source'].apply(ast.literal_eval)
        # map_dict = map_result['cluster2source'].iloc[0]

        source_positions_str = scene_data['source_positions'].iloc[0]
        source_positions = ast.literal_eval(source_positions_str)
        source_positions = np.array(source_positions)
        total_mics = len(scene_data)
        labels = np.sort(scene_data['cluster_label'].unique())

        plt.rcParams.update({'font.size': 18}) 

        fig, ax = plt.subplots()

        x_min = min(scene_data['x'].min(), source_positions[:,0].min())
        x_max = max(scene_data['x'].max(), source_positions[:,0].max())
        y_min = min(scene_data['y'].min(), source_positions[:,1].min())
        y_max = max(scene_data['y'].max(), source_positions[:,1].max())

        x_padding = (x_max - 0) * 0.1
        y_padding = (y_max - 0) * 0.1

        ax.set_xlim(max(0 - x_padding, 0), x_max + x_padding) 
        ax.set_ylim(max(0 - y_padding, 0), y_max + y_padding) 

        x_ticks = np.arange(0, x_max + x_padding + 2, 2)
        y_ticks = np.arange(0, y_max + y_padding + 2, 2)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])

        # mapped_values = set(map_dict.values())
        # remaining_values = set(range(len(labels))) - mapped_values
        # remaining_values = sorted(list(remaining_values))

        # full_map = map_dict.copy()
        # unmapped_labels = set(labels) - set(map_dict.keys())

        # for label in unmapped_labels:
        #     if remaining_values:
        #         full_map[label] = remaining_values.pop(0)
        #     else:
        #         full_map[label] = max(mapped_values) + 1
        #         mapped_values.add(full_map[label])

        # color_dict = {mapped_val: colors[i % len(colors)] 
        #             for i, mapped_val in enumerate(sorted(set(full_map.values())))}
        # marker_dict = {mapped_val: markers[i % len(markers)] 
        #             for i, mapped_val in enumerate(sorted(set(full_map.values())))}
        # face_dict = {mapped_val: faces[i % len(faces)] 
        #             for i, mapped_val in enumerate(sorted(set(full_map.values())))}

        for label in labels:
            cluster_data = scene_data[scene_data['cluster_label'] == label]
            #mapped_label = full_map[label]
            plt.scatter(cluster_data['x'], cluster_data['y'], 
                    edgecolors=colors[label],
                    facecolors=faces[label],
                    marker = markers[label],
                    label=f'Cluster {label}',
                    s = 85)

        for source in source_positions:
            plt.scatter(source[0], source[1], color='red', marker='*', s=120)
            

        # for i in range(total_mics):
        #     ax.annotate(i, (scene_data['x'].iloc[i], scene_data['y'].iloc[i]))

        plt.title(f'Microphone Clustering in {scene}')

        plt.xlabel('X (m)', fontsize=24)
        plt.ylabel('Y (m)', fontsize=24)

        handles, labels = ax.get_legend_handles_labels()

        # sorted_indices = np.argsort([int(label.split()[-1]) for label in labels])
        # sorted_handles = [handles[i] for i in sorted_indices]
        # sorted_labels = [labels[i] for i in sorted_indices]

        #plt.legend(handles, labels, loc='best')

        #plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{directory}/{scene}_{model_use}.png')

        #quit()
    #plt.show()