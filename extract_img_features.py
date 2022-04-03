import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import h5py
from tqdm import tqdm
import os


class ImgExtractor():

    def __init__(self):
        # Load the pretrained model
        self.model = models.resnet152(pretrained=True).cuda()
        # Use the model object to select the desired layer
        self.layer = self.model._modules.get('avgpool')

        # Set model to evaluation mode
        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vector(self, image_name):
        # 1. Load the image with Pillow library
        img = Image.open(image_name)
        if len(img.size) == 2:
            img = Image.new("RGB", img.size)
        # 2. Create a PyTorch Variable with the transformed image
        t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0)).cuda()
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(2048)
        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data.reshape(o.data.size(1)))
        # 5. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        self.model(t_img)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector
        return my_embedding

    def extract(self, direc, dataset_part = "train"):
        print("\t-----Extracting Image Features-----")
        
        img_files = list(os.listdir(direc))

        hdf5_file = h5py.File("./data/image_features_" + dataset_part + ".hdf5", 'w')
        
        if dataset_part == "train":
            for i in tqdm(range(len(img_files))):
                hdf5_file[str(int(img_files[i][15:27]))] = self.get_vector(direc + "/" + img_files[i]).detach().cpu().numpy()
                
        else:
            for i in tqdm(range(len(img_files))):
                hdf5_file[str(int(img_files[i][13:25]))] = self.get_vector(direc + "/" + img_files[i]).detach().cpu().numpy()

        hdf5_file.close()
