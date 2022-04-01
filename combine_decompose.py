import os
import h5py
import numpy as np
from tensorly.decomposition import tucker
from tqdm import tqdm


def combine_decompose(text_hdf5, img_hdf5, core_hdf5):
    print("\t-----Combining and Decomposing Feature Tensors-----")
    
    tensor_to_decompose = []
    ques_ids = []

    hdf5_file = h5py.File(core_hdf5, 'w')

    with h5py.File(img_hdf5, 'r') as fi, h5py.File(text_hdf5, 'r') as fq:
        ques_ids = list(fq.keys())
        for i in tqdm(range(len(ques_ids))):
            q_id = str(ques_ids[i])
            img_id = q_id[:len(q_id) - 3]
            tensor_dot = np.tensordot(fi[img_id], fq[q_id], 0)
            core, factors = tucker(np.array(tensor_dot), rank=[1, 384])
            hdf5_file[ques_ids[i]] = core

    hdf5_file.close()
