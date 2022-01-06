import os
import dataset
# import extract_img_features as ex_img
# import extract_text_features as ex_text 
# import combine_decompose as c_d
# import fully_connected_layer as fcl

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
    "text" : "text_features_train.hdf5",
    "img" : "image_features_train.hdf5",
    "core_tensors" : "core_tensors_train.hdf5"
}

# dataset.download_dataset("images", "train")
dataset.download_dataset(repo_path, "questions", "train")

# ex_img.ImgExtractor().extract(directories["img_train"])
# ex_text.TextFeatureExtractor().extract_features(directories["ques_train"])

# c_d.combine_decompose(HDF5_files["text"], HDF5_files["img"])
# fcl