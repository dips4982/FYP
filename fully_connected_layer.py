import json
import numpy as np 
import h5py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

from tqdm import tqdm


def fit(num_of_epochs, model, loss_fn, opt, train_dl):
    train_accuracy = Accuracy()
    for epoch in range(num_of_epochs):

        # for dropout
        model.train()
        losses = list()

        for xb, yb in train_dl:

            # 1 forward
            preds = model(xb)

            # 2 compute the objective function
            loss = loss_fn(torch.squeeze(preds), torch.squeeze(yb))

            # 3 cleaning the gradients
            opt.zero_grad()

            # 4 accumulate the partial derivatives of loss wrt params
            loss.backward()

            # 5 step in the opposite direction of the gradient
            opt.step()

            # training step accuracy
            batch_acc = train_accuracy(output, yb.to(dtype=torch.int32))

            losses.append(loss.item())
        
        total_train_accuracy = train_accuracy.compute()
        print(f'Epoch: {epoch+1} \t Training Loss: {torch.tensor(losses).mean(): .2f} \t Accuracy: {total_train_accuracy.absolute()}')
        train_accuracy.reset()


def train_fc_layer(core_hdf5, embeddings_file, annotations_file):
    # instantiate the model
    model = nn.Linear(384, 3000)

    # print model architecture
    print(model.parameters())

    ans_file = open(annotations_file)
    ans_data = json.load(ans_file)

    freq_ans_file = open(embeddings_file)
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

    train_ds = TensorDataset(inputs, outputs)
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size, shuffle = True)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), 1e-5)

    fit(100, model, loss_fn, opt, train_dl)
    torch.save(model, "model.pt")