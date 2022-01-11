import os
import h5py
import numpy as np
from tensorly.decomposition import tucker
from tqdm import tqdm


# def gen_img_ids():
#     ids = []
#     id = "000000000000"

#     for i in range(1, 82784):
#         ids.append(id[:12 - len(str(i))] + str(i))

#     return ids


def get_img_id(ques_id):
    ques_id = ques_id[:len(ques_id) - 3]
    for i in range(12 - len(ques_id)):
        ques_id = "0" + ques_id
    return ques_id[:12]


def combine_decompose(text_hdf5, img_hdf5):
    tensor_to_decompose = []

    # image_ids = gen_img_ids()

    core_tensors = []
    ques_ids = []

    hdf5_file = h5py.File("core_tensors_train.hdf5", 'w')

    with h5py.File(img_hdf5, 'r') as fi, h5py.File(text_hdf5, 'r') as fq:
        ques_ids = list(fq.keys())
        for q_id in ques_ids:
            q_id = str(q_id)
            img_id = get_img_id(q_id)
            tensor_dot = np.tensordot(fi[img_id], fq[q_id], 0)
            core, factors = tucker(np.array(tensor_dot), rank=[1, 16000])
            core_tensors.append(core)

    
    # for tensor in tensor_to_decompose:
    #     core, factors = tucker(np.array(tensor), rank=[1, 16000])
    #     core_tensors.append(core)

    hdf5_file = h5py.File("core_tensors_train.hdf5", 'w')

    for i in tqdm(range(len(ques_ids))):
        hdf5_file[ques_ids[i]] = core_tensors[i]

    hdf5_file.close()
