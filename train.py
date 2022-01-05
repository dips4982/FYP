import dataset
import extract_img_features as ex_img
import extract_text_features as ex_text 
import combine_decompose as c_d
import fully_connected_layer as fcl

directories = {
    "img_train" : "train2014",
    "img_val" : "val2014",
    "ques_train" : "v2_Questions_Train_mscoco",
    "ques_val" : "v2_Questions_Val_mscoco"
}

HDF5_files = {
    "text" : "text_features_train.hdf5",
    "img" : "image_features_train.hdf5",
    "core_tensors" : "core_tensors_train.hdf5"
}

dataset.download_dataset("Images", "train")
dataset.download_dataset("Questions", "train")

ex_img.ImgExtractor().extract(directories["img_train"])
ex_text.TextFeatureExtractor().extract_features(directories["ques_train"])

c_d.combine_decompose(HDF5_files["text"], HDF5_files["img"])
fcl