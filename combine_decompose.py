import os
import h5py
import numpy as np
from tensorly.decomposition import tucker
from tqdm import tqdm


def gen_img_ids():
    ids = []
    id = "000000000000"

    for i in range(1, 82784):
        ids.append(id[:12 - len(str(i))] + str(i))

    return ids


def combine_decompose(text_hdf5, img_hdf5):
    tensor_to_decompose = []

    image_ids = gen_img_ids()

    with h5py.File(img_hdf5, 'r') as f, h5py.File(text_hdf5, 'r') as fq:
        for i in image_ids:
            tensor_to_decompose.append(np.tensordot(f[i], fq[i], 0))

    core_tensors = []
    for tensor in tensor_to_decompose:
        core, factors = tucker(np.array(tensor), rank=[1, 1, 3000])
        core_tensors.append(core)

    hdf5_file = h5py.File("core_tensors.hdf5", 'w')

    for i in tqdm(range(len(image_ids))):
        hdf5_file[image_ids[i]] = core_tensors[i]

    hdf5_file.close()
