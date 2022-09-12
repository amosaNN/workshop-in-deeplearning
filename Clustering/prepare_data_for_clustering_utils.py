import pandas as pd
import numpy as np
import re
import os
from glob import glob
import config

def using_clump(a):
    return [a[s] for s in np.ma.clump_unmasked(np.ma.masked_where(a.astype(str)==str(np.nan),a))]
def centroids_zero_center(tracks_arr):
    for centroids_arr in tracks_arr:
        centroids_arr-=centroids_arr[0]
    return tracks_arr
def str_array_to_float(arr_of_arr_of_str):
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    final_mat = []
    for arr_of_arr in arr_of_arr_of_str:
        float_ts = []
        for str in arr_of_arr:
            float_arr = [float(i) for i in rx.findall(str)]
            if(len(float_arr)>=2):
                float_ts.append(float_arr)
        if(len(float_ts)>=1):
            final_mat.append(float_ts)
    return np.array(final_mat,dtype=object)
def get_lens(tracks):
    return pd.Series([len(trk) for trk in tracks]).value_counts()
def get_feature_index(feature_type,features):
    for idx,fet in enumerate(features):
        if fet in feature_type:
            return idx
    return 0
def cut_feture_vecs_and_preprocess(tracks,feature_type,ts_len,cut_longer_ts=False):
    if(cut_longer_ts):
        track_cut = np.array([trk[:ts_len] for trk in tracks if len(trk)>=ts_len])
    else:
        track_cut = np.array([trk for trk in tracks if len(trk)==ts_len])
    if 'centroids' in feature_type:
        track_cut = centroids_zero_center(track_cut)
    return track_cut
def save(tracks_final,well_name):
    np.save('../npy_files/'+well_name+'.npy',tracks_final)

def from_results_folder_PATH_to_arrays(features=['centroids','morphologies','embeddings'],ts_len=10,cut_longer_ts=False,save=False,name_ext=""):
    all_tracks = []
    wells = []
    all_paths = [path for path, subdir, files in os.walk(config.res_path)]
    for path in all_paths:
        feature_vecs_cut = []
        all_files = [file for file in glob(os.path.join(path, config.csv_file_ext))]
        if(len(all_files)<1):
            continue
        for file in all_files:
            file_name = file.split('_')
            well_name = file_name[1]
            feature_type = file_name[-1]
            if(not any(fet in feature_type for fet in features)):
                continue
            df_str = pd.read_csv(file,index_col=[0])
            splitted = []
            for cell_id, series in df_str.iterrows():
                tracks = np.array(using_clump(np.array(series)),dtype=object)
                for tr in tracks:
                    splitted.append(tr)
            tracks_str = np.array(splitted,dtype=object)
            #print("tracks_str shape: ",tracks_str.shape)
            tracks = str_array_to_float(tracks_str)
            tracks_cut = cut_feture_vecs_and_preprocess(tracks,feature_type,ts_len,cut_longer_ts)
            feature_vecs_cut.append(tracks_cut)
        feature_vecs_cut = np.dstack(feature_vecs_cut)
        if(len(feature_vecs_cut[0])>0):
            print(feature_vecs_cut.shape)
            all_tracks.append(feature_vecs_cut)
            wells.append(well_name)
    #return all_tracks,wells
    labels = []
    for well_name,tracks_vec in zip(wells,all_tracks):
        labels.append(np.repeat(well_name,len(tracks_vec)))

    results_tracks = np.vstack(all_tracks)
    results_labels = np.concatenate(labels)
    cell_types = np.array([config.wells_to_genetype_dict[well] for well in results_labels])
    if(save):
        np.save(config.npy_save_path+'/features'+name_ext+'.npy',results_tracks)
        np.save(config.npy_save_path+'/labels'+name_ext+'.npy',results_labels)
        np.save(config.npy_save_path+'/celltypes'+name_ext+'.npy',cell_types)
    return results_tracks,results_labels
