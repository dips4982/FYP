import os
import dataset
import extract_img_features as ex_img
import extract_text_features as ex_text 
import combine_decompose as c_d
import fully_connected_layer as fcl

# store cwd path
repo_path = os.getcwd()

directories = {
    "img_train" : "/data/train2014",
    "img_val" : "/data/val2014",
    "ques_train" : "/data/v2_Questions_Train_mscoco.json",
    "ques_val" : "/data/v2_Questions_Val_mscoco.json",
    "ans_train" : "/data/v2_Annotations_Train_mscoco.json",
    "ans_val" : "/data/v2_Annotations_Val_mscoco.json"
}

HDF5_files = {
    "text" : "/data/text_features_train.hdf5",
    "img" : "/data/image_features_train.hdf5",
    "core_tensors" : "/data/core_tensors_train.hdf5"
}

dataset.download_dataset(repo_path, "images", "train")
dataset.download_dataset(repo_path, "questions", "train")

ex_img.ImgExtractor().extract(repo_path + directories["img_train"])
ex_text.TextFeatureExtractor().extract_features(repo_path + directories["ques_train"])

c_d.combine_decompose(repo_path + HDF5_files["text"], repo_path + HDF5_files["img"])
# fcl