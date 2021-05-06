import configparser
from eeg_v2 import EEG
from utils import chunks, baseline_calc, butter_bandpass_filter, create_features, calc_psd
import numpy as np
from mne.preprocessing import ICA
import mne
import matplotlib
# matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import json
import sys
import os
import pandas as pd
from datetime import datetime
import joblib
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA
from coroica import UwedgeICA
import argparse

# https://github.com/mne-tools/mne-python/issues/2404
config = configparser.ConfigParser()
config.read("config.ini")

_WINDOW = int(config["CHUNK"]["WINDOW"])
N_COMP = 10
# THRESHOLD = 0.263506042367534
# THRESHOLD = 0.32729012499342763
# THRESHOLD = 0.75

def _get_ica_map(ica, components=None):
    """Get ICA topomap for components"""
    fast_dot = np.dot
    if components is None:
        components = list(range(ica.n_components_))
    maps = fast_dot(ica.mixing_matrix_[:, components].T,
                    ica.pca_components_[:ica.n_components_])
    return maps


def shift_data(data, k=1):
    """
    Shift data by k time points
    """
    shifted = np.roll(data, k, axis=0)
    shifted[:k, :] = 0
    return shifted


def plot_components(data, n_components, figsize=(25, 20), title="CCA components",):
    """
    Plot components
    """
    fig, axes = plt.subplots(n_components, 1, figsize=figsize)
    print(n_components)
    for i in range(n_components):
        ax = axes[i]
        ax.plot(data[:, i], linewidth=0.8)
        ax.axis("off")
        ax.set_title(f"comp {i}", loc="left")
    fig.suptitle(title, fontsize=20)
    plt.show(block=False)


def plot_raw(data, eeg, figsize=(25, 20), title="Raw"):
    fig, axes = plt.subplots(data.shape[1], 1, figsize=figsize)
    for i, ch in enumerate(eeg.ch_names):
        ax = axes[i]
        ax.plot(data[:, i], linewidth=0.8, color="black")
        ax.axis("off")
        ax.set_title(f"{ch}", loc="left")
    fig.suptitle(title, fontsize=20)
    plt.show(block=False)

def plot_psd(data1, data2, eeg, figsize=(25, 20), title="PSD Plots"):
    fig, axes = plt.subplots(data1.shape[1] // 2, 2, figsize=figsize, sharex='col')
    for i, ax in enumerate(axes.flat):
        x1, y1 = calc_psd(data1[:, i])
        x2, y2 = calc_psd(data2[:, i])
        ax.plot(x1, y1, label="Raw")
        ax.plot(x2, y2, label="Corrected")
        ax.set_xlim([0, 50])
        ax.legend()
        # ax.axis("off")
        ax.text(0.5, 1.05, eeg.ch_names[i], horizontalalignment='center', transform=ax.transAxes)
        # ax.set_title(f"{eeg.ch_names[i]}", loc="right")
    fig.suptitle(title, fontsize=20)
    plt.show(block=False)

def detect_emg(data, clf, n_components=10, threshold=0.5):
    """
    Returns a list of post-CCA EMG components from raw EMG data if any
    """
    # CCA
    cca = CCA(n_components, max_iter=1000)
    c1, c2 = cca.fit_transform(data, shift_data(data))

        # sys.exit()

    # EMG detection
    components = c1.copy().T
    # Create features
    X = create_features(components)
    # Predict
    # emg_preds = clf.predict(X)
    emg_probs = clf.predict_proba(X)[:, 1]
    # print(emg_preds)
    emg_preds = (emg_probs >= threshold).astype('int')
    list_of_emg = np.where(emg_preds == 1)[0]
    list_of_probs = emg_probs[list_of_emg]
    return cca, c1, list_of_emg, list_of_probs

def detect_emg2(data, clf, n_components=10, threshold=0.5):
    """
    Returns a list of post-ICA EMG components from raw EMG data if any
    """
    uica = UwedgeICA(n_components=n_components, timelags=[1,5,10])
    components = uica.fit_transform(data)

    comp = components.copy().T
    X = create_features(comp)
    # Predict
    # emg_preds = clf.predict(X)
    emg_probs = clf.predict_proba(X)[:, 1]
    # print(emg_preds)
    emg_preds = (emg_probs >= threshold).astype('int')
    list_of_emg = np.where(emg_preds == 1)[0]
    list_of_probs = emg_probs[list_of_emg]

    return uica, components, list_of_emg, list_of_probs

def detect_emg3(data, clf, n_components=10, threshold=0.5):
    """
    Returns a list of post-CCA EMG components from raw EMG data if any
    """
    # CCA
    fica = FastICA(n_components, max_iter=1000)
    c1 = fica.fit_transform(data,)

    # EMG detection
    components = c1.copy().T
    # Create features
    X = create_features(components)
    # Predict
    # emg_preds = clf.predict(X)
    emg_probs = clf.predict_proba(X)[:, 1]
    print(emg_probs)
    emg_preds = (emg_probs >= threshold).astype('int')
    list_of_emg = np.where(emg_preds == 1)[0]
    list_of_probs = emg_probs[list_of_emg]
    return fica, c1, list_of_emg, list_of_probs


def remove_emg(data, clf, plot=False):
    """
    Remove EMG artifacts from EEG data
    """
    cca, c1, list_of_emg = detect_emg(data, clf, plot)
    if list_of_emg is not None and len(list_of_emg) > 0:
        # Zeroing out the chosen components
        c1[:, list_of_emg] = 0
        corrected = cca.inverse_transform(c1)
        return corrected
    else:
        return data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method', help='Method for blind source separation. Default is CCA', default='cca')
    parser.add_argument(
        '--threshold', type=float, help='EMG prediction threshold. Default is 0.5', default=0.5)

    path_edf = "edf/1750/1750_alice/edf/A0001750.edf"
    path_stage = "edf/1750/1750_alice/csv/STAGE.csv"

    eeg = EEG(path_edf=path_edf, path_stage=path_stage)

    name = input("Subject no: ")

    _components = []

    # Load classification model
    clf = joblib.load("./models/emg_classifier_20200819.joblib")
    # print(clf)
    # sys.exit()

    args = parser.parse_args()
    method = args.method
    THRESHOLD = args.threshold

    comp_title = ''
    if method == 'cca':
        comp_title = 'CCA components'
    elif method == 'uica' or method == 'fastica':
        comp_title = 'ICA components'
    else:
        comp_title = 'Components'

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
            # Detect EMG
            print("Detecting EMG...")
            if method == 'cca':
                bss, c1, list_of_emg, list_of_probs = detect_emg(raw, clf, N_COMP, threshold=THRESHOLD)
            elif method == 'uica': # ICA
                bss, c1, list_of_emg, list_of_probs = detect_emg2(raw, clf, N_COMP, threshold=THRESHOLD)
            elif method =='fastica': # method = fastica
                bss, c1, list_of_emg, list_of_probs = detect_emg3(raw, clf, N_COMP, threshold=THRESHOLD)
            if list_of_emg is not None and len(list_of_emg) > 0:
                # Plot CCA components
                plot_components(c1, N_COMP, title=comp_title)
                # Zeroing out the chosen components
                print("Found EMG in the following components: {}".format(str(list_of_emg).strip('[]')))
                print("Corresponding probabilities: {}".format(str(list_of_probs).strip('[]')))
                # print(list_of_probs)
                if method == 'cca' or method == 'fastica':
                    c1[:, list_of_emg] = 0
                    corrected = bss.inverse_transform(c1)
                else: # UwedgeICA
                    c1[:, list_of_emg] = 0
                    corrected = np.linalg.lstsq(bss.V_, c1.T)[0].T

                # Plot the results
                # plt.figure()
                plot_raw(raw, eeg)
                plot_psd(raw, corrected, eeg, figsize=(25, 60), title="PSD plots")
                # plt.figure()
                plot_raw(corrected, eeg, title='EMG-removed')
                # plot_psd(corrected, eeg, figsize=(25, 60), title="PSD - EMG-removed")

                continuer = input("Continue? [y/n]: ")
                if continuer == "n":
                    sys.exit()
            else:
                # Plot CCA components
                plot_components(c1, N_COMP, title=comp_title)
                print("No EMG component to remove")
                continuer = input("Continue? [y/n]: ")
                if continuer == "n":
                    sys.exit()


if __name__ == "__main__":
    main()
