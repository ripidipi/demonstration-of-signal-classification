import os
import numpy as np
import h5py


def load_radioml2018(path="data/raw/radioml2018/versions/2/GOLD_XYZ_OSC.0001_1024.hdf5"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    print(f"Загружаем из HDF5: {path}...")
    with h5py.File(path, "r") as f:
        X = f['X'][:]          
        Y_onehot = f['Y'][:]   
        SNRs = f['Z'][:]       
    y_idx = np.argmax(Y_onehot, axis=1)
    num_classes = Y_onehot.shape[1]

    print(" Данные загружены:")
    print(f" - X.shape         = {X.shape}")
    print(f" - Y_onehot.shape  = {Y_onehot.shape}")
    print(f" - SNRs.shape      = {SNRs.shape}")
    print(f" - Число классов   = {num_classes}")

    return X, y_idx, SNRs, num_classes


if __name__ == '__main__':
    X, y_idx, snrs, num_classes = load_radioml2018()
    print("🔎 Пример:")
    print(" X[0].shape =", X[0].shape)
    print(" y_idx[0]   =", y_idx[0])
    print(" snrs[0]    =", snrs[0])
    print(" num_classes=", num_classes)