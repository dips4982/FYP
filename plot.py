import matplotlib.pyplot as plt
import pickle

train_graph = None
validation_graph = None

# load train_graph json file
train_file = open('train_graph', 'rb') 
train_graph = pickle.load(train_file)

# load validation_graph json file
val_file = open('validation_graph', 'rb')
validation_graph = pickle.load(val_file)

print("/t-----Plotting Loss and Accuracy Graphs of Training and Validation-----")

plt.figure(figsize=(10,5))
plt.title("Loss")
plt.plot(train_graph["loss"],label="train loss")
plt.plot(validation_graph["loss"],label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("Accuracy")
plt.plot(train_graph["accuracy"],label="train Accuracy")
plt.plot(validation_graph["accuracy"],label="validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("F1 score")
plt.plot(train_graph["F1"],label="train F1")
plt.plot(validation_graph["F1"],label="validation F1")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.legend()
plt.show()

