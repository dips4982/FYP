import json
import numpy as np 
import h5py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from tqdm import tqdm


def train_model(num_of_epochs, model, loss_fn, opt, train_dl):
    train_accuracy = Accuracy()
    for epoch in range(num_of_epochs):

        # for dropout
        model.train()
        losses = list()
        actual = list()
        predictions = list()
        
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
            
            # output = (torch.softmax(preds, dim=1)>0.5).int()

            # training step accuracy
            # batch_acc = train_accuracy(preds, yb.to(dtype=torch.int32))
            # print(preds, yb)
            actual.append(torch.squeeze(yb))
            predictions.append(torch.squeeze(preds))

            losses.append(loss.item())
        
        # total_validation_accuracy = train_accuracy.compute()

        total = 0
        correct = 0
        for i in range(len(train_dl)):
          a = actual[i].detach().numpy()
          p = predictions[i].detach().numpy()
          for j in range(len(a)):
            # print(a[j], p[j])
            act_label = np.argmax(a[j]) # act_label = 1 (index)
            pred_label = np.argmax(p[j]) # pred_label = 1 (index)
            if(act_label == pred_label):
                correct += 1
            total += 1
        accuracy = (correct/total)

        print(f'Epoch: {epoch+1} \t Training Loss: {torch.tensor(losses).mean(): .2f} \t Accuracy: {accuracy}')
        train_accuracy.reset()


def validate_model(num_of_epochs, model, loss_fn, opt, validation_dl):
    validation_accuracy = Accuracy()
    for epoch in range(num_of_epochs):

        # to turn off dropout
        model.eval()
        losses = list()
        actual = list()
        predictions = list()
        
        for xb, yb in validation_dl:

            # 1 forward
            preds = None
            with torch.no_grad():
                preds = model(xb)

            # 2 compute the objective function
            loss = loss_fn(torch.squeeze(preds), torch.squeeze(yb))

            # output = (torch.softmax(preds, dim=1)>0.5).int()

            # validationing step accuracy
            # batch_acc = validation_accuracy(preds, yb.to(dtype=torch.int32))
            actual.append(torch.squeeze(yb))
            predictions.append(torch.squeeze(preds))

            # add loss to loss_list
            losses.append(loss.item())
        
        total = 0
        correct = 0
        for i in range(len(validation_dl)):
          a = actual[i].detach().numpy()
          p = predictions[i].detach().numpy()
          for j in range(len(a)):
            # print(a[j], p[j])
            act_label = np.argmax(a[j]) # act_label = 1 (index)
            pred_label = np.argmax(p[j]) # pred_label = 1 (index)
            if(act_label == pred_label):
                correct += 1
            total += 1
        accuracy = (correct/total)
        # total_validation_accuracy = validation_accuracy.compute()
        print(f'Epoch: {epoch+1} \t Validation Loss: {torch.tensor(losses).mean(): .2f} \t Accuracy: {accuracy}')
        validation_accuracy.reset()


def train_and_validate_fc_layer(train_core_hdf5, validation_core_hdf5, embeddings_file, train_annotations_file, validation_annotations_file):
    # instantiate the model
    model = nn.Linear(384, 3000)

    # print model architecture
    print(model.parameters())

    freq_ans_file = open(embeddings_file)
    freq_ans_data = json.load(freq_ans_file)


    ## Training
    train_ans_file = open(train_annotations_file)
    train_ans_data = json.load(train_ans_file)

    train_inputs = []
    train_outputs = []

    with h5py.File(train_core_hdf5, 'r') as core_file:
        ques_ids = list(core_file.keys())
        for r in tqdm(range(len(ques_ids))):
            i = ques_ids[r]
            train_inputs.append(np.array(core_file[i], dtype = np.float32))
            
            for element in train_ans_data['annotations']:
                if element['question_id'] == int(i):
                    if element['multiple_choice_answer'] in freq_ans_data:
                        ans_arr = [[int(x) for x in freq_ans_data[element['multiple_choice_answer']]]]
                        train_outputs.append(np.array(ans_arr, dtype = np.float32))
                    else:
                        ans_arr = [[int(x) for x in freq_ans_data["yes"]]]
                        train_outputs.append(np.array(ans_arr, dtype = np.float32))

    # Convert to tensors
    train_inputs = torch.from_numpy(np.array(train_inputs))
    train_outputs = torch.from_numpy(np.array(train_outputs))

    # Training Dataset
    train_ds = TensorDataset(train_inputs, train_outputs)
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size, shuffle = True)


    ## Validation
    validation_ans_file = open(validation_annotations_file)
    validation_ans_data = json.load(validation_ans_file)

    validation_inputs = []
    validation_outputs = []

    with h5py.File(validation_core_hdf5, 'r') as core_file:
        ques_ids = list(core_file.keys())
        for r in tqdm(range(len(ques_ids))):
            i = ques_ids[r]
            validation_inputs.append(np.array(core_file[i], dtype = np.float32))
            
            for element in validation_ans_data['annotations']:
                if element['question_id'] == int(i):
                    if element['multiple_choice_answer'] in freq_ans_data:
                        ans_arr = [[int(x) for x in freq_ans_data[element['multiple_choice_answer']]]]
                        validation_outputs.append(np.array(ans_arr, dtype = np.float32))
                    else:
                        ans_arr = [[int(x) for x in freq_ans_data["yes"]]]
                        validation_outputs.append(np.array(ans_arr, dtype = np.float32))

    # Convert to tensors
    validation_inputs = torch.from_numpy(np.array(validation_inputs))
    validation_outputs = torch.from_numpy(np.array(validation_outputs))

    # Validation Dataset
    validation_ds = TensorDataset(validation_inputs, validation_outputs)
    batch_size = 5
    validation_dl = DataLoader(validation_ds, batch_size, shuffle = True)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_model(50, model, loss_fn, opt, train_dl)
    validate_model(50, model, loss_fn, opt, validation_dl)


    torch.save(model, "model.pt")