import configparser
from eeg_v2 import EEG
from utils import chunks, baseline_calc, butter_bandpass_filter
import numpy as np
from mne.preprocessing import ICA
import mne
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import json
import sys
import os
import pandas as pd
from datetime import datetime
from sklearn.cross_decomposition import CCA

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


def plot_components(data, figsize=(25, 20), title="CCA components"):
    """
    Plot components
    """
    fig, axes = plt.subplots(10, 1, figsize=figsize)
    for i in range(10):
        ax = axes[i]
        ax.plot(data[:, i], linewidth=0.8)
        ax.axis("off")
        ax.set_title(f"comp {i}", loc="left")
    fig.suptitle(title, fontsize=20)
    plt.show()


def plot_raw(data, eeg, figsize=(25, 20), title="Raw"):
    fig, axes = plt.subplots(data.shape[1], 1, figsize=figsize)
    for i, ch in enumerate(eeg.ch_names):
        ax = axes[i]
        ax.plot(data[:, i], linewidth=0.8, color="black")
        ax.axis("off")
        ax.set_title(f"{ch}", loc="left")
    fig.suptitle(title, fontsize=20)
    plt.show(block=False)


def main():

    path_edf = "edf/1489/1489_alice/edf/A0001489.edf"
    path_stage = "edf/1489/1489_alice/csv/STAGE.csv"

    eeg = EEG(path_edf=path_edf, path_stage=path_stage)

    name = input("Subject no: ")

    _components = []

    N_SAMPLE = int(config["DEFAULT"]["N_SAMPLE_PER_SUBJECT"])
    counter = 0

    if not os.path.exists("csvs2"):
        os.mkdir("csvs2")

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

            # Get raw numpy data
            raw = sample_raw_bandpass.get_data().T

            # CCA
            cca = CCA(10)
            c1, c2 = cca.fit_transform(raw, shift_data(raw))

            # Plot CCA components
            plot_components(c1)

            while True:
                try:
                    list_of_emg = input(
                        "List of EMG components seperated by space: ")
                    list_of_emg = list(map(int, list_of_emg.split()))

                    break
                except ValueError:
                    print("Try again..")
            # Back up the components
            components = c1.copy().T
            if list_of_emg is not None and len(list_of_emg) > 0:
                # Zeroing out the chosen components
                c1[:, list_of_emg] = 0
                corrected = cca.inverse_transform(c1)

                # Plot the results
                # plt.figure()
                plot_raw(raw, eeg)
                # plt.figure()
                plot_raw(corrected, eeg, title='EMG-removed')

                to_save = input("Save this chunk? [y/n]: ")
                if to_save == 'y':
                    # Output the components and labels to csv
                    labels = [
                        'EMG' if i in list_of_emg else 'Non-EMG' for i in range(10)]
                    temp = pd.DataFrame(components, columns=[f"comp_{i}" for i in range(components.shape[1])])
                    temp['label'] = labels
                    now = datetime.now()
                    now_st = now.strftime("%Y%m%d_%H%M%S")
                    temp.to_csv(f"./csvs2/{name}_{idx}_{now_st}.csv", index=False, header=True)

                    continuer = input("Continue? [y/n]: ")
                    if continuer == "n":
                        sys.exit()
                else:
                    continuer = input("Continue? [y/n]: ")
                    if continuer == "n":
                        sys.exit()
            else:
                print("No EMG component to remove")
                to_save = input("Save this chunk? [y/n]: ")
                if to_save == 'y':
                    # Output the components and labels to csv
                    labels = ['Non-EMG' for i in range(10)]
                    temp = pd.DataFrame(components, columns=[f"comp_{i}" for i in range(components.shape[1])])
                    temp['label'] = labels
                    now = datetime.now()
                    now_st = now.strftime("%Y%m%d_%H%M%S")
                    temp.to_csv(f"./csvs2/{name}_{idx}_{now_st}.csv", index=False, header=True)

                    continuer = input("Continue? [y/n]: ")
                    if continuer == "n":
                        sys.exit()
                else:
                    continuer = input("Continue? [y/n]: ")
                    if continuer == "n":
                        sys.exit()

if __name__ == "__main__":
    main()
