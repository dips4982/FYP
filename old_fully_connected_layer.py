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
            input_tensor.append(Variable(torch.Tensor(np.array(core_file[i]))))
            
            for element in ans_data['annotations']:
                if element['question_id'] == int(i):
                    if element['multiple_choice_answer'] in freq_ans_data:
                        ans_arr = [[int(x) for x in freq_ans_data[element['multiple_choice_answer']]]]
                        output_tensor.append(Variable(torch.Tensor(ans_arr)))
                    else:
                        ans_arr = [[int(x) for x in freq_ans_data["yes"]]]
                        output_tensor.append(Variable(torch.Tensor(ans_arr)))

    loss_fn = torch.nn.MSELoss(size_average=False)
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

        # Use the nn package to define our model as a sequence of layers. nn.Sequential
        # is a Module which contains other Modules, and applies them in sequence to
        # produce its output. Each Linear Module computes output from input using a
        # linear function, and holds internal Variables for its weight and bias.
        # model = torch.nn.Sequential(
        #     torch.nn.Linear(D_in, H),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(H, D_out),
        # )

        # Clear the gradients
        optimizer.zero_grad()

        # The nn package also contains definitions of popular loss functions; in this
        # case we will use Mean Squared Error (MSE) as our loss function.
    
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Variable of input data to the Module and it produces
        # a Variable of output data.
        y_pred = model(x)

        # Compute and print loss. We pass Variables containing the predicted and true
        # values of y, and the loss function returns a Variable containing the
        # loss.
        loss = loss_fn(y_pred, y)
        # print(i, loss.data)

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Variables with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Variable, so
        # we can access its data and gradients like we did before.
        # for param in model.parameters():
        #     param.data -= learning_rate * param.grad.data

        # Update Weights
        optimizer.step()

        # Calculat Loss
        train_loss += loss.item()
        # training step accuracy
        output = (y_pred>0.5).int()
        batch_acc = train_accuracy(output, y.to(dtype=torch.int32))
        # print(batch_acc)

      # total accuracy over all training batches
      total_train_accuracy = train_accuracy.compute()
      print('Epoch {} \t\t Training Loss: {} \t Accuracy: {}'.format(e+1, train_loss / len(input_tensor), total_train_accuracy.absolute()))

    torch.save(model, "model.pt")