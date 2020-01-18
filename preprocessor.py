import numpy as np
from utils import prepare_frames_features
from frame import Frame

train_file = "data/training_data.csv"

def process_file(file_path, k_mers, label):
    with open(file_path, 'r') as f:
        content = f.read()
        examples = content.split('>')
        frames = [Frame(k_mers, s.split('\n')[1], label) for s in examples if len(s) > 0]
    return prepare_frames_features(frames)

def preprocess(k_mers):
    randoms = process_file('data/randoms1500.txt', k_mers, 0)
    vista = process_file('data/vista1500.txt', k_mers, 1)
    data = np.concatenate((randoms, vista))
    np.savetxt(train_file, data)
    return data

def get_train_data():
    return np.genfromtxt(train_file)