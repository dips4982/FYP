import json
import pickle
from transformers import pipeline
import h5py
import numpy
from sentence_transformers import SentenceTransformer

class TextFeatureExtractor():

    def __init__(self):
        self.classifier = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    def extract_features(self, direc):
        classifier = self.classifier
        with open('questions.json') as data_file:    
            with h5py.File('text_features_train.hdf5', 'w') as h5file:
                data = json.load(data_file)
                for questions in data['questions']:
                    h5file[str(questions['question_id'])] = classifier.encode(questions['question'])
                    
                    
    def load_features(self):
            fileh5 = h5py.File('text_features_train.hdf5', 'r')
        item = fileh5['item']
        keys = item.keys()
        out_dict = {}
        for key in keys:
            out_dict[key] = item[key][()]
        fileh5.close()
        return out_dict
            
    