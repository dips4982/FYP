import json
import pickle
from transformers import pipeline
import h5py
import numpy
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class TextFeatureExtractor():

    def __init__(self):
        self.classifier = SentenceTransformer('sentence-transformers/distilbert-base-nli-max-tokens')

    def extract_features(self, direc, dataset_part = "train"):
        print("\t-----Extracting Text Features-----")
        
        classifier = self.classifier
        with open(direc) as data_file:    
            h5file = h5py.File('./data/text_features_' + dataset_part + '.hdf5', 'w')

            data = json.load(data_file)
            for i in tqdm(range(len(data['questions']))):
                questions = data['questions'][i]
                h5file[str(questions['question_id'])] = classifier.encode(questions['question'])

            h5file.close()
                    
    def load_features(self):
        fileh5 = h5py.File('text_features_train.hdf5', 'r')
        item = fileh5['item']
        keys = item.keys()
        out_dict = {}
        for key in keys:
            out_dict[key] = item[key][()]
        fileh5.close()
        return out_dict
    
    
