import h5py
import torch
from torch import nn
from torch.autograd import Variable
import json
import numpy as np
from torchmetrics.classification import Accuracy
import torchmetrics

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1,384, 100, 3000

class FCLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define the layers
        self.layers = nn.Sequential(
            nn.Linear(D_in, D_out)
        )
    
    def forward(self, x):
        # forward pass
        x = torch.softmax(self.layers(x), dim=1)
        return x


def train_fc_layer(core_hdf5, embeddings_file, annotations_file):
    # instantiate the model
    model = FCLayer()

    # print model architecture
    print(model)

    ans_file = open(annotations_file)
    ans_data = json.load(ans_file)

    freq_ans_file = open(embeddings_file)
    freq_ans_data = json.load(freq_ans_file)

    input_tensor = []
    output_tensor = []

    with h5py.File(core_hdf5, 'r') as core_file:
        ques_ids = list(core_file.keys())
        for i in ques_ids:
            input_tensor.append(Variable(torch.Tensor(np.array(core_file[i])), requires_grad=True))
            
            for element in ans_data['annotations']:
                if element['question_id'] == int(i):
                    if element['multiple_choice_answer'] in freq_ans_data:
                        ans_arr = [int(x) for x in freq_ans_data[element['multiple_choice_answer']]]
                        output_tensor.append(Variable(torch.Tensor(ans_arr), requires_grad=True))
                    else:
                        ans_arr = [int(x) for x in freq_ans_data["yes"]]
                        output_tensor.append(Variable(torch.Tensor(ans_arr), requires_grad=True))

    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    learning_rate = 1e-4

    epochs = 100

    train_accuracy = Accuracy()
    valid_accuracy = Accuracy(compute_on_step=False)
    for e in range(epochs):
      train_loss = 0.0
      correct = 0
      for i in range(len(input_tensor)):
        x = input_tensor[i]

        # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
        # x = Variable(torch.randn(N, D_in))
        y = output_tensor[i]

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x)
        output = torch.flatten((y_pred>0.2).int())

        # Compute and print loss
        # print(torch.argmax(torch.flatten(y_pred)),torch.argmax(y))
        print(torch.argmax(torch.flatten(y_pred)),torch.argmax(y))
        loss = loss_fn(torch.flatten(y_pred), y.long())

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass:
        loss.backward()

        # Update Weights
        optimizer.step()

        # Calculat Loss
        train_loss += loss.item()
        # training step accuracy
        batch_acc = train_accuracy(output, y.to(dtype=torch.int32))
        # print(batch_acc)

      # total accuracy over all training batches
      total_train_accuracy = train_accuracy.compute()
      print('Epoch {} \t\t Training Loss: {} \t Accuracy: {}'.format(e+1, train_loss / len(input_tensor), total_train_accuracy.absolute()))

    torch.save(model, "model.pt")