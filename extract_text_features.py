import json
import pickle
from transformers import pipeline
import h5py
import numpy
from sentence_transformers import SentenceTransformer

class textFeatureExtractor():

    def __init__(self):
        self.classifier = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    def extract_features(self, direc):
        path = 'item/'
        classifier = self.classifier
        with open('/content/drive/MyDrive/VQA_FinalYrProject/pretty.json') as data_file:    
            with h5py.File('/content/drive/MyDrive/VQA_FinalYrProject/text_features_data.h5', 'w') as h5file:
                data = json.load(data_file)
                for questions in data['questions']:
                    h5file[path+str(questions['question_id'])] = classifier.encode(questions['question'])

        
    