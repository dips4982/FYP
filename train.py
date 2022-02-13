import os
import dataset
import extract_img_features as ex_img
import extract_text_features as ex_text 
import combine_decompose as c_d
import fully_connected_layer as fcl

# store cwd path
repo_path = os.getcwd() 

dataset.download_dataset(repo_path, "images", "train")
dataset.download_dataset(repo_path, "questions", "train")

ex_img.ImgExtractor().extract(repo_path + dataset.directories["img_train"], "train")
ex_text.TextFeatureExtractor().extract_features(repo_path + dataset.directories["ques_train"], "train")

c_d.combine_decompose(repo_path + dataset.HDF5_files["text_train"], repo_path + dataset.HDF5_files["img_train"], repo_path + dataset.HDF5_files["core_tensors_train"])
fcl.train_fc_layer(repo_path + dataset.HDF5_files["core_tensors_train"], repo_path + "/frequent_embeddings.json", repo_path + dataset.directories["ans_train"])
