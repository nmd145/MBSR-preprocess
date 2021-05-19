import mne
import pandas as pd
from mne.io import read_raw_edf
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
from scipy.signal import welch
from scipy import stats
from scipy.integrate import simps
from scipy.signal import savgol_filter
from statsmodels.tsa.ar_model import AutoReg
import pywt
import numpy as np
from matplotlib import pyplot as plt
from mne.preprocessing import ICA

# Sample rate and desired cutoff frequencies (in Hz).
FS = 500.0
LOWCUT = 0.5
HIGHCUT = 45.0

# https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

# Calculate Representative Rest1 mean:
def baseline_calc(a):
    e = np.mean(a)
    return a - e

# Banpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut=.5, highcut=45., fs=500, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def _get_ica_map(ica, components=None):
    """Get ICA topomap for components"""
    fast_dot = np.dot
    if components is None:
        components = list(range(ica.n_components_))
    maps = fast_dot(ica.mixing_matrix_[:, components].T,
                    ica.pca_components_[:ica.n_components_])
    return maps

def ica_pipe(data_after_bandpass_baseline, clf_eog):
    """ Use ICA algorithm to remove eog signals from eeg data which was applied bandpass & baseline correction
    Args:
        data_after_bandpass_baseline: eeg data applied bandpass & baseline correction
        clf_eog: eog removal classification model (trained)
    Returns:
        ica: the ica object (defined by mne.preprocessing.ICA) after removing eog components
        sample_raw_removed_eog: eeg data after removing eog signals
    """
    sample_raw_removed_eog = data_after_bandpass_baseline.copy()

    # Fitting ICA
    ica = ICA(method="extended-infomax", random_state=1)
    ica.fit(sample_raw_removed_eog)

    maps = _get_ica_map(ica).T
    scalings = np.linalg.norm(maps, axis=0)
    maps /= scalings[None, :]
    X = maps.T

    # Predict EOG
    eog_preds = clf_eog.predict(X)
    list_of_eog = np.where(eog_preds == 1)[0]

    # ica.plot_sources(inst=sample_raw_train)
    # ica.plot_components(inst=sample_raw_train)

    ica.exclude = list_of_eog
    ica.apply(sample_raw_removed_eog)

    return ica, sample_raw_removed_eog

def calc_psd(s, _fs = FS, _avg='median', fmax = 100):
    _nperseg = 4*_fs
    x, y = welch(s, fs=_fs, average=_avg, nperseg=_nperseg)
    x, y = x[np.where(x<fmax)], y.T[np.where(x<fmax)]

    return x, y

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n + 1]

def bandpower(freqs, psd, low, high, relative=True):
    """
    Calculate band power
    """
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def get_spectral_features(row,):
    """
    Create spectral features
    """
    x, y = calc_psd(row)
    return np.array([bandpower(x, y, i, i + 5) for i in range(0, 46, 5)])

def get_dwt_features(row, wavelet="db1"):
    """
    Create discrete wavelet transform features
    """
    cA, cD = pywt.dwt(row, wavelet=wavelet)
    return np.array(
        [
            cA.mean(),
            cD.mean(),
            np.median(cA),
            np.median(cD),
            stats.skew(cA),
            stats.skew(cD),
            cA.std(),
            cD.std(),
            cA.max(),
            cD.max(),
            cA.min(),
            cD.min(),
        ]
    )

def get_autoreg_features(row):
    mod = AutoReg(row, 6)
    res = mod.fit()
    return np.array(res.params)

def get_sg_features(row):
    sg = savgol_filter(row, 255, 2)
    return np.array([sg.mean(), np.median(sg), sg.max(), sg.min(), np.std(sg)])

def create_features(data):
    # Create dwt features
    data_dwt = np.apply_along_axis(get_dwt_features, 1, data)
    # Create spectral features
    data_spectral = np.apply_along_axis(get_spectral_features, 1, data)
    # Create autoregressive features
    data_autoreg = np.apply_along_axis(get_autoreg_features, 1, data)
    # Create smoothed features
    data_sg = np.apply_along_axis(get_sg_features, 1, data)
    # Return merged features
    return np.concatenate((data_spectral, data_dwt, data_autoreg, data_sg), axis=1)

def shift_data(data, k=1):
    """
    Shift data by k time points
    """
    shifted = np.roll(data, k, axis=0)
    shifted[:k, :] = 0
    return shifted

def plot_raw(data, 
             eeg, 
             figsize=(25, 20), 
             y_unit='V',
             fs=500,
             title='Raw'):
    fig, axes = plt.subplots(data.shape[1], 1, figsize=figsize)
    dim_x, dim_y = data.shape
    time_total = dim_x / fs
    time = np.linspace(0,time_total,num=dim_x)
    if y_unit == 'mV':
        data = data * 10e3
    elif y_unit == 'uV':
        data = data * 10e6
    for i, ch in enumerate(eeg.ch_names):
        ax = axes[i]
        ax.plot(time, data[:,i], linewidth=0.8, color="black")
        ax.set_title(f"{ch}", loc="left")
        ax.set_ylabel(y_unit)
        ax.set_xlabel('time (s)')
    fig.suptitle(title, fontsize=20)
    plt.show(block=False)
    
def plot_channel_imfs(data, 
             eeg, 
             imf_idx=0,
             figsize=(25, 20), 
             y_unit='V',
             fs=500,
             title='Raw'):
    channel_data = data[:,imf_idx,:]
    fig, axes = plt.subplots(channel_data.shape[1], 1, figsize=figsize)
    dim_x, dim_y = channel_data.shape
    time_total = dim_x / fs
    time = np.linspace(0,time_total,num=dim_x)
    if y_unit == 'mV':
        channel_data = channel_data * 10e3
    elif y_unit == 'uV':
        channel_data = channel_data * 10e6
    for i, ch in enumerate(eeg.ch_names):
        ax = axes[i]
        ax.plot(time, channel_data[:, i], linewidth=0.8)
        ax.set_title(f"{ch}", loc="left")
        ax.set_ylabel(y_unit)
        ax.set_xlabel('time (s)')
    fig.suptitle(title, fontsize=20)
    plt.show(block=False)
    
def plot_imfs(data, 
              n_imfs=10, 
              figsize=(25, 20), 
              title="IMF", 
              fs=500,
              y_unit = 'V'):
    """
    Plot components
    """
    dim_x, dim_y = data.shape
    time_total = dim_x / fs
    time = np.linspace(0,time_total,num=dim_x)
    fig, axes = plt.subplots(n_imfs, 1, figsize=figsize)
    print(n_imfs)
    if y_unit == 'mV':
        data = data * 10e3
    if y_unit == 'uV':
        data = data * 10e6
    for i in range(n_imfs):
        ax = axes[i]
        ax.plot(time, data[:,i], linewidth=0.8)
        #plt.xticks(#np.arange(0,time,100))
        #ax.axis("off")
        ax.set_title(f"imf {i+1}", loc='left')
        ax.set_ylabel(y_unit)
        ax.set_xlabel("time (s)")
    fig.suptitle(title, fontsize=20)
    plt.show(block=False)
    
#def plot_raw(data, eeg, figsize=(25, 20), title="Raw"):
#    fig, axes = plt.subplots(data.shape[1], 1, figsize=figsize)
#    for i, ch in enumerate(eeg.ch_names):
#        ax = axes[i]
#        ax.plot(data[:, i], linewidth=0.8, color="black")
#        ax.axis("off")
#        ax.set_title(f"{ch}", loc="left")
#    fig.suptitle(title, fontsize=20)
#    plt.show(block=False)

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

#def plot_components(data, n_components, figsize=(25, 20), title="CCA components"):
    """
    Plot components
    """
    #fig, axes = plt.subplots(n_components, 1, figsize=figsize)
    #print(n_components)
    #for i in range(n_components):
    #    ax = axes[i]
    #    ax.plot(data[:, i], linewidth=0.8)
    #    ax.axis("off")
   #     ax.set_title(f"comp {i}", loc="left")
   # fig.suptitle(title, fontsize=20)
   # plt.show(block=False)
    
def plot_components(data, 
              n_component=10, 
              figsize=(25, 20), 
              title="Components", 
              fs=500,
              y_unit = 'V'):
    """
    Plot components
    """
    dim_x, dim_y = data.shape
    time_total = dim_x / fs
    time = np.linspace(0,time_total,num=dim_x)
    fig, axes = plt.subplots(n_component, 1, figsize=figsize)
    print(n_component)
    if y_unit == 'mV':
        data = data * 10e3
    if y_unit == 'uV':
        data = data * 10e6
    for i in range(n_imfs):
        ax = axes[i]
        ax.plot(time, data[:,i], linewidth=0.8)
        #plt.xticks(#np.arange(0,time,100))
        #ax.axis("off")
        ax.set_title(f"component {i}", loc='left')
        ax.set_ylabel(y_unit)
        ax.set_xlabel("time (s)")
    fig.suptitle(title, fontsize=20)
    plt.show(block=False)

def plot_fft(data_1, data_2, eeg, FS , figsize = (20,15)): # data with all channel 
    fig, axes = plt.subplots(data_1.shape[1], 1, figsize=figsize)
    for i, ax in enumerate(axes.flat):

        eeg_fft = fft(data_1[:,i])
        eeg_fft_2 = fft(data_2[:,i]) 
        N = len(eeg_fft)
        n = np.arange(N)
        T = N/ FS # data points 
        freq = n/T 
        ax.stem(freq, np.abs(eeg_fft), 'b', \
                markerfmt=" ", basefmt="-b", label = 'Before', alpha=0.7)
        ax.stem(freq, np.abs(eeg_fft_2), 'r', \
                markerfmt=" ", basefmt="-r", label = 'After')
        ax.set_xlim(0,50)
        ax.set_ylim(0,4)
        ax.legend()

        ax.set_title(f"{eeg.ch_names[i]}", loc="left")

        plt.tight_layout()
        plt.show()
