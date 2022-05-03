import json
import numpy as np 
import h5py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics import F1
from tqdm import tqdm
import pickle


def train_model(num_of_epochs, model, loss_fn, opt, train_dl):
    print("\t-----Training Phase Started-----")
    
    train_graph = {
        "accuracy": [],
        "loss": [],
        "F1": []
    }

    f1 = F1(num_classes = 3000, threshold = 0.5, top_k = 5)

    train_accuracy = Accuracy()

    for epoch in range(num_of_epochs):

        # for dropout
        model.train()

        losses = list()
        actual = list()
        predictions = list()

        actual_index = list()
        predictions_index = list()
        
        for xb, yb in train_dl:
            
            xb = xb.cuda()
            yb = yb.cuda()

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

            for i in range(len(preds)):
                predictions.append(np.array(preds[i][0].detach().cpu(), dtype = np.float64))
                actual.append(np.array(yb[i][0].cpu(), dtype = np.int32))

                predictions_index.append(torch.argmax(preds[i][0]))
                actual_index.append(torch.argmax(yb[i][0]))

             # training step accuracy
            batch_acc = train_accuracy(torch.tensor(predictions_index), torch.tensor(actual_index))

            losses.append(loss.item())

        f1_score = f1(torch.tensor(predictions), torch.tensor(actual))
        
        total_train_accuracy = train_accuracy.compute()

        mean_loss = torch.tensor(losses).mean()
        train_graph["loss"].append(mean_loss)

        train_graph["accuracy"].append(total_train_accuracy.absolute())

        train_graph["F1"].append(f1_score)

        print(f'Epoch: {epoch+1} \t Training Loss: {mean_loss: .2f} \t Accuracy: {total_train_accuracy.absolute()} \t F1 Score: {f1_score}')
        train_accuracy.reset()

    return train_graph


def train_fc_layer(train_core_hdf5, embeddings_file, train_annotations_file):
    
    # instantiate the model
    model = nn.Sequential(
        nn.Linear(384, 3000),
        nn.Softmax(dim=2)
    ).cuda()

    # print model architecture
    print(model.parameters)

    freq_ans_file = open(embeddings_file)
    freq_ans_data = json.load(freq_ans_file)
    freq_ans_file.close()

    mapping_file = open('non_frequent_mapping_qid', 'rb')
    mapping_data = pickle.load(mapping_file)

    ## Training
    train_ans_file = open(train_annotations_file)
    train_ans_data = json.load(train_ans_file)
    train_ans_file.close()

    train_inputs = []
    train_outputs = []
    
    print("\t-----Constructing Training Input Output Datasets-----")

    with h5py.File(train_core_hdf5, 'r') as core_file:
        ques_ids = list(core_file.keys())
        for r in tqdm(range(len(ques_ids))):
            i = ques_ids[r]
            train_inputs.append(np.array(core_file[i], dtype = np.float32))

            ans_arr = [[int(x) for x in freq_ans_data[mapping_data[int(i)]]]]
            train_outputs.append(np.array(ans_arr, dtype = np.float32))

    # Convert to tensors
    train_inputs = torch.from_numpy(np.array(train_inputs))
    train_outputs = torch.from_numpy(np.array(train_outputs))

    # Training Dataset
    train_ds = TensorDataset(train_inputs, train_outputs)
    batch_size = 512
    train_dl = DataLoader(train_ds, batch_size, shuffle = True)


    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_graph = train_model(10, model, loss_fn, opt, train_dl)
    
    torch.save(model, "model.pt")
    print("\t-----Model Saved Successfully-----")

    # save training graph in json file
    try:
        train_file = open('train_graph', 'wb')
        pickle.dump(train_graph, train_file)
        train_file.close()

    except:
        print("Unable to write training stats to file !!")
