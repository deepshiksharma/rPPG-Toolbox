import os, re, math

import cv2
import pickle
import torch
import numpy as np

import scipy
from scipy import linalg
from scipy import signal
from scipy import sparse
from scipy.sparse import spdiags
from scipy.signal import butter

import matplotlib.pyplot as plt

# HELPER FUNCTIONS
def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data

def _process_signal(signal, fs=30, diff_flag=True):
    # Detrend and filter
    use_bandpass = True
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        gt_bvp = _detrend(np.cumsum(signal), 100)
    else:
        gt_bvp = _detrend(signal, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz
        # equals [45, 150] beats per min
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        signal = scipy.signal.filtfilt(b, a, np.double(signal))
    return signal

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def visualize_results(filepath='sample_output.pickle', trial_idx=1, chunk_size=180, chunk_num=0):
    """
    Args:
        - filepath (str): path to pickle file with stored results
        - trial_idx (int): index of test subject result to visualize
        - chunk_size (int): size of test chunk to visualize (-1 to plot the entire test result signal)
        - chunk_num (int): index of test chunk to visualize
    """
    # Read in data and list subjects
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # List of all video trials
    trial_list = list(data['predictions'].keys())
    print(f'Num Trials: {len(trial_list)}\n')

    # Reform label and prediction vectors from multiple trial chunks
    prediction = np.array(_reform_data_from_dict(data['predictions'][trial_list[trial_idx]]))
    label = np.array(_reform_data_from_dict(data['labels'][trial_list[trial_idx]]))
    
    # Read in meta-data from pickle file
    fs = data['fs'] # Video Frame Rate
    label_type = data['label_type'] # PPG Signal Transformation: `DiffNormalized` or `Standardized`
    diff_flag = (label_type == 'DiffNormalized')
    
    if chunk_size == -1:
        chunk_size = len(prediction)
        chunk_num = 0
    
    # Process label and prediction signals
    prediction = _process_signal(prediction, fs, diff_flag=diff_flag)
    label = _process_signal(label, fs, diff_flag=diff_flag)
    start = (chunk_num)*chunk_size
    stop = (chunk_num+1)*chunk_size
    samples = stop - start
    x_time = np.linspace(0, samples/fs, num=samples)
    
    plt.figure(figsize=(10,5))
    plt.plot(x_time, prediction[start:stop], color='r')
    plt.plot(x_time, label[start:stop], color='black')
    plt.title('Trial: ' + trial_list[trial_idx])
    plt.legend(['Predictions', 'Labels'])
    plt.xlabel('Time (s)')
    plt.show()
