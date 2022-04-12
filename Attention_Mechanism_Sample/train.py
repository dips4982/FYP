import json
import numpy as np 
import h5py
import json
import torch
import pickle
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics import F1
from tqdm import tqdm
from torch.autograd import Variable


from BaseModel import build_baseline_model


def train_model(num_of_epochs, model, loss_fn, opt, train_dl):
    print("/t-----Training Phase Started-----")
    
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
        
        for q, v, yb in train_dl:
            v = Variable(v, requires_grad=True).cuda()
            q = Variable(q, requires_grad=True).cuda()
            yb = yb.cuda()

            # 1 forward

            preds = model(v, q)
            # print(preds)

            # 2 compute the objective function
            loss = loss_fn(torch.squeeze(preds), torch.squeeze(yb))

            # 3 cleaning the gradients
            opt.zero_grad()

            # 4 accumulate the partial derivatives of loss wrt params
            loss.backward()

            # 5 step in the opposite direction of the gradient
            opt.step()

            for i in range(len(preds)):
                predictions.append(np.array(preds[i].detach().cpu(), dtype = np.float64))
                actual.append(np.array(yb[i].cpu(), dtype = np.int32))

                predictions_index.append(torch.argmax(preds[i]))
                actual_index.append(torch.argmax(yb[i]))

             # training step accuracy
            batch_acc = train_accuracy(torch.tensor(predictions_index), torch.tensor(actual_index))

            losses.append(loss.item())

        # print(predictions, len(actual))
        f1_score = f1(torch.tensor(np.array(predictions)), torch.tensor(np.array(actual)))
        
        total_train_accuracy = train_accuracy.compute()

        mean_loss = torch.tensor(losses).mean()
        train_graph["loss"].append(mean_loss)

        train_graph["accuracy"].append(total_train_accuracy.absolute())

        train_graph["F1"].append(f1_score)

        print(f'Epoch: {epoch+1} \t Training Loss: {mean_loss: .2f} \t Accuracy: {total_train_accuracy.absolute()} \t F1 Score: {f1_score}')
        train_accuracy.reset()

    return train_graph


def train_fc_layer(image_hdf5, question_hdf5, embeddings_file, train_annotations_file):
    
    # instantiate the model
    model = build_baseline_model()

    # print model architecture
    print(model.parameters)

    # Frequent Embeddings
    freq_ans_file = open(embeddings_file)
    freq_ans_data = json.load(freq_ans_file)
    freq_ans_file.close()

    # Answer Maping
    json_file = open(train_annotations_file, 'rb')
    data = pickle.load(json_file)


    train_questions = []
    train_images = []
    train_labels = []
    
    print("/t-----Constructing Training Input Output Datasets-----")

    with h5py.File(question_hdf5, 'r') as ques_file, h5py.File(image_hdf5, 'r') as img_file:
        ques_ids = list(ques_file.keys())
        for r in tqdm(range(len(ques_ids))):
            # question
            q_id = ques_ids[r]
            train_questions.append(np.array(ques_file[q_id], dtype = np.float32))
            # image
            img_id = q_id[:len(q_id) - 3]
            train_images.append(np.array(img_file[img_id], dtype = np.float32))
            # label
            ans_arr = [[int(x) for x in freq_ans_data[data[int(q_id)]]]]
            train_labels.append(np.array(ans_arr, dtype = np.float32))

    # Convert to tensors
    train_questions = torch.from_numpy(np.array(train_questions))
    train_images = torch.from_numpy(np.array(train_images))
    train_labels = torch.from_numpy(np.array(train_labels))

    # Training Dataset
    train_ds = TensorDataset(train_questions, train_images, train_labels)
    batch_size = 256
    train_dl = DataLoader(train_ds, batch_size, shuffle = True)  #see


    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_graph = train_model(100, model, loss_fn, opt, train_dl)

    # save training graph in json file
    with open('train_graph.json', 'w') as f:
        json.dump(train_graph, f)
    
    torch.save(model, "model.pt")
    print("/t-----Model Saved Successfully-----")