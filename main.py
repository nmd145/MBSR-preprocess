import configparser
from eeg_v2 import EEG
from utils import chunks, baseline_calc, butter_bandpass_filter
import numpy as np
from mne.preprocessing import ICA
import mne
from matplotlib import pyplot as plt
import json
import sys, os
import pandas as pd

# https://github.com/mne-tools/mne-python/issues/2404

config = configparser.ConfigParser()
config.read("config.ini")

_WINDOW = int(config["CHUNK"]["WINDOW"])


def _get_ica_map(ica, components=None):
    """Get ICA topomap for components"""
    fast_dot = np.dot
    if components is None:
        components = list(range(ica.n_components_))
    maps = fast_dot(ica.mixing_matrix_[:, components].T,
                    ica.pca_components_[:ica.n_components_])
    return maps

def main():

    path_edf="edf/1489/1489_alice/edf/A0001489.edf"
    path_stage="edf/1489/1489_alice/csv/STAGE.csv"
    
    eeg = EEG(path_edf=path_edf, path_stage=path_stage)

    name = input("Subject no: ")

   
    _components = []

    N_SAMPLE = int(config["DEFAULT"]["N_SAMPLE_PER_SUBJECT"])
    counter = 0 

    if not os.path.exists("csvs"):
        os.mkdir("csvs")

    for task in eeg.tasks:
        if counter > N_SAMPLE:
            break

        for idx, chunk in enumerate(chunks(task, _WINDOW)):

            if counter > N_SAMPLE:
                break

            _min, _max = np.min(chunk), np.max(chunk)
            
            # Getting sample data
            sample_raw = eeg.raw.copy().crop(_min, _max, include_tmax=False)

            # Apply baseline correction
            sample_raw_baseline = sample_raw.copy()
            sample_raw_baseline.apply_function(baseline_calc)

            # sample_raw.copy().pick('Fp1').plot()
            # sample_raw_baseline.copy().pick('Fp1').plot()

            # break
            
            # Apply bandpass filter
            sample_raw_bandpass = sample_raw_baseline.copy()
            sample_raw_bandpass.apply_function(butter_bandpass_filter)
            
            # Apply ICA
            sample_raw_train = sample_raw_bandpass.copy()
            sample_raw_corrected = sample_raw_bandpass.copy()
            
            # Train
            ica = ICA(method="extended-infomax", random_state=1)
            ica.fit(sample_raw_corrected)
            
            # Plot ICA component
            ica.plot_sources(inst=sample_raw_train)
            ica.plot_components(inst=sample_raw_train)
            
            while True:
                try:
                    list_of_eog = input("List of EOG components seperated by space: ")
                    list_of_eog = list(map(int, list_of_eog.split()))

                    list_of_ecg = input("List of ECG components seperated by space: ")
                    list_of_ecg = list(map(int, list_of_ecg.split()))

                    list_of_emg = input("List of EMG components seperated by space: ")
                    list_of_emg = list(map(int, list_of_emg.split()))


                    break
                except ValueError:
                    print("Try again..")
            
            if list_of_eog or list_of_ecg or list_of_emg:

                ica.exclude = [*list_of_eog, *list_of_emg, *list_of_ecg]
                # raw_ = eeg.raw.copy()
                # sample_raw_corrected = sample_raw_bandpass.copy()

                ica.apply(sample_raw_corrected)

                sample_raw.plot(title="RAW")
                sample_raw_corrected.plot(title="RAW_CORRECTED")

                to_save = input("Save this trunk? [y/n]: ")
                # print("=================================================================================")

                if to_save == 'y':  
                    maps = _get_ica_map(ica).T
                    scalings = np.linalg.norm(maps, axis=0)
                    maps /= scalings[None, :]
                    
                    # maps[:,0] = 0

                    components = ica.get_sources(inst=sample_raw_train).get_data()
                    components_name = ica.ch_names
                    
                    def f(idx):
                        if idx in list_of_eog:
                            return "EOG"
                        
                        if idx in list_of_ecg:
                            return "ECG"
                        
                        if idx in list_of_emg:
                            return "EMG"

                        return "EEG"


                    for idx2, comp_name in enumerate(components_name):
                        tmp = {
                            "name" : "{comp_name}_{idx}".format(comp_name=comp_name, idx=idx),
                            # "component" : list(components[idx2, :]),
                            "map" : list(maps[:, idx2]),
                            "label" : f(idx2)
                        }
                        # Save file to csv
                        temp = pd.DataFrame(components[idx2,:].reshape(-1, len(components[idx2,: ])), columns = [f"comp_{i}" for i in range(len(components[idx2,:]))])
                            
                        temp['name'] = "{comp_name}_{idx}".format(comp_name=comp_name, idx=idx)
                        temp['label'] = "EOG" if idx2 in list_of_eog else "Non-EOG"
                        # print(temp)
                        comp_map = list(maps[:, idx2])
                        # print(comp_map)
                        for i in range(len(comp_map)):
                            temp[f"map_component_{i}"] = comp_map[i]
                        temp.to_csv(f'./csvs/{name}_{idx}_{comp_name}.csv', index=False, header=True)
                        # print(maps[:, idx2].shape)
                        # sys.exit()
                        try:
                            _components.append(tmp)
                            counter+=1
                        
                        except Exception as e:
                            break

    j = {
        "name" : name,
        "data" : _components
    }

    with open("data.json", "w") as f:
        json.dump(j, f)


if __name__ == "__main__":
    main()