import json
import numpy as np 
import h5py
import json
import torch
import torch.nn as nn
import pickle
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics import F1
from tqdm import tqdm


def validate_model(num_of_epochs, model, loss_fn, opt, validation_dl):
    print("/t-----Validation Phase Started-----")
    
    validation_graph = {
        "accuracy": [],
        "loss": [],
        "F1": []
    }

    f1 = F1(num_classes = 3000, threshold = 0.5, top_k = 5)

    validation_accuracy = Accuracy()

    for epoch in range(num_of_epochs):

        # to turn off dropout
        model.eval()

        losses = list()
        actual = list()
        predictions = list()

        actual_index = list()
        predictions_index = list()
        
        for xb, yb in validation_dl:

            xb = xb.cuda()
            yb = yb.cuda()

            # 1 forward
            preds = None
            with torch.no_grad():
                preds = model(xb)

            # 2 compute the objective function
            loss = loss_fn(torch.squeeze(preds), torch.squeeze(yb))

            for i in range(len(preds)):
                predictions.append(np.array(preds[i][0].detach().cpu(), dtype = np.float64))
                actual.append(np.array(yb[i][0].cpu(), dtype = np.int32))

                predictions_index.append(torch.argmax(preds[i][0]))
                actual_index.append(torch.argmax(yb[i][0]))

            # validationing step accuracy
            batch_acc = validation_accuracy(torch.tensor(predictions_index), torch.tensor(actual_index))

            # add loss to loss_list
            losses.append(loss.item())

        f1_score = f1(torch.tensor(predictions), torch.tensor(actual))
        
        total_validation_accuracy = validation_accuracy.compute()

        mean_loss = torch.tensor(losses).mean()
        validation_graph["loss"].append(mean_loss)

        validation_graph["accuracy"].append(total_validation_accuracy.absolute())

        validation_graph["F1"].append(f1_score)

        print(f'Epoch: {epoch+1} \t Validation Loss: {mean_loss: .2f} \t Accuracy: {total_validation_accuracy.absolute()} \t F1 Score: {f1_score}')
        validation_accuracy.reset()

    return validation_graph


def validate_fc_layer( validation_core_hdf5, embeddings_file, validation_annotations_file):
    
    # load model
    model = torch.load("model.pt")

    # print model architecture
    print(model.parameters)

    freq_ans_file = open(embeddings_file)
    freq_ans_data = json.load(freq_ans_file)
    freq_ans_file.close()

    ## Validation
    validation_ans_file = open(validation_annotations_file)
    validation_ans_data = json.load(validation_ans_file)
    validation_ans_file.close()

    print("/t-----Constructing Validation Input Output Datasets-----")

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
    batch_size = 512
    validation_dl = DataLoader(validation_ds, batch_size, shuffle = True)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    validation_graph = validate_model(10, model, loss_fn, opt, validation_dl)

    # save validation graph in json file
    try:
        val_file = open('validation_graph', 'wb')
        pickle.dump(validation_graph, val_file)
        val_file.close()

    except:
        print("Unable to write Validation stats to file !!")

    print("/t-----Validation Phase Ended-----")
