import pickle
import numpy as np

with open('data/raw/RML2016.10a_dict.pkl', 'rb') as f:
    Xd = pickle.load(f, encoding='latin1')

mods = sorted(set([key[0] for key in Xd.keys()]))
snrs = sorted(set([key[1] for key in Xd.keys()]))

X = []
Y = []

for mod in mods:
    for snr in snrs:
        if (mod, snr) in Xd:
            X.append(Xd[(mod, snr)])
            Y.extend([(mod, snr)] * Xd[(mod, snr)].shape[0])

X = np.vstack(X)
