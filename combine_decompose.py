import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from tensorly.decomposition import tucker


# Load the pretrained model
model = models.resnet152(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    if len(img.size) == 2:
      img = Image.new("RGB", img.size)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(2048)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

hdf5_file = h5py.File("temp.hdf5", 'w')
# hdf5_att = hdf5_file.create_dataset('att', [len(a),2048], dtype='f')

count = 0

a = list(os.listdir("/content/train2014"))
image_ids = []
print(a[0])

for i in a:
  if count < 10:
    hdf5_file[i[15:27]] = get_vector("/content/train2014/" + i).detach().cpu().numpy()
    image_ids.append(i[15:27])
    print(count)
    count += 1



hdf5_file.close()


hdf5_file = h5py.File("tempq.hdf5", 'w')

question_arr = np.random.random(size=(10,768))
id=0

for i in question_arr:
  hdf5_file[image_ids[id]] = i
  id +=1


hdf5_file.close()

tensor_to_decompose = []

with h5py.File('temp.hdf5' , 'r') as f ,h5py.File('tempq.hdf5' , 'r') as fq:
    for i in image_ids:
      tensor_to_decompose.append(np.tensordot(f[i], fq[i],0))

tensor_to_decompose = np.array(tensor_to_decompose)
print(tensor_to_decompose.shape)


core, factors = tucker(tensor_to_decompose, rank=[1, 1, 3000])