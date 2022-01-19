import os
import h5py
import numpy as np
from tensorly.decomposition import tucker
from tqdm import tqdm


def combine_decompose(text_hdf5, img_hdf5):
    tensor_to_decompose = []

    # image_ids = gen_img_ids()

    core_tensors = []
    ques_ids = []

    with h5py.File(img_hdf5, 'r') as fi, h5py.File(text_hdf5, 'r') as fq:
        ques_ids = list(fq.keys())
        for q_id in ques_ids:
            q_id = str(q_id)
            img_id = q_id[:len(q_id) - 3]
            tensor_dot = np.tensordot(fi[img_id], fq[q_id], 0)
            core, factors = tucker(np.array(tensor_dot), rank=[1, 16000])
            core_tensors.append(core)

    
    # for tensor in tensor_to_decompose:
    #     core, factors = tucker(np.array(tensor), rank=[1, 16000])
    #     core_tensors.append(core)

    hdf5_file = h5py.File("/data/core_tensors_train.hdf5", 'w')

    for i in tqdm(range(len(ques_ids))):
        hdf5_file[ques_ids[i]] = core_tensors[i]

    hdf5_file.close()
