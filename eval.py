import os
import json
import numpy as np 
import h5py
import dataset
import extract_img_features as ex_img
import extract_text_features as ex_text
import combine_decompose as c_d

from numpy import vstack
import tqdm

import torch
from torchmetrics.classification import Accuracy
from sklearn.metrics import accuracy_score

repo_path = os.getcwd() 

model = torch.load('model.pt')

dataset.download_dataset(repo_path, "images", "val")
dataset.download_dataset(repo_path, "questions", "val")

ex_img.ImgExtractor().extract(repo_path + dataset.directories["img_val"], "val")
ex_text.TextFeatureExtractor().extract_features(repo_path + dataset.directories["ques_val"], "val")

c_d.combine_decompose(repo_path + dataset.HDF5_files["text_val"], repo_path + dataset.HDF5_files["img_val"], repo_path + dataset.HDF5_files["core_tensors"])

core_hdf5 = repo_path + dataset.HDF5_files["core_tensors_val"]

ans_file = open(repo_path + dataset.directories["ans_val"])
ans_data = json.load(ans_file)

freq_ans_file = open(repo_path + "/frequent_embeddings.json")
freq_ans_data = json.load(freq_ans_file)

inputs = []
outputs = []

with h5py.File(core_hdf5, 'r') as core_file:
    ques_ids = list(core_file.keys())
    for r in tqdm(range(len(ques_ids))):
        i = ques_ids[r]
        inputs.append(np.array(core_file[i], dtype = np.float32))
        
        for element in ans_data['annotations']:
            if element['question_id'] == int(i):
                if element['multiple_choice_answer'] in freq_ans_data:
                    ans_arr = [[int(x) for x in freq_ans_data[element['multiple_choice_answer']]]]
                    outputs.append(np.array(ans_arr, dtype = np.float32))
                else:
                    ans_arr = [[int(x) for x in freq_ans_data["yes"]]]
                    outputs.append(np.array(ans_arr, dtype = np.float32))

inputs = torch.from_numpy(np.array(inputs))
outputs = torch.from_numpy(np.array(outputs))
print(inputs,outputs)

def eval_model(model):
    predictions, actuals = [],[]
    for i in range(inputs):
        # evaluate the model on the test set
        yhat = model(inputs[i])
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = outputs[i].numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

eval_model(model)

