
from train import train_fc_layer


directories = {
    "img_train" : "/data/train2014",
    "img_val" : "/data/val2014",
    "ques_train" : "/data/v2_OpenEnded_mscoco_train2014_questions.json",
    "ques_val" : "/data/v2_OpenEnded_mscoco_val2014_questions.json",
    "ans_train" : "/data/v2_mscoco_train2014_annotations.json",
    "ans_val" : "/data/v2_mscoco_val2014_annotations.json",
    "frequent_embeddings" : "/frequent_embeddings.json",
    "non_frequent_mappings" : "/non_frequent_mappings_qid",
}

HDF5_files = {
    "text_train" : "/data/text_features_train.hdf5",
    "img_train" : "/data/image_features_train.hdf5",
    "core_tensors_train" : "/data/core_tensors_train.hdf5",

    "text_val" : "/data/text_features_val.hdf5",
    "img_val" : "/data/image_features_val.hdf5",
    "core_tensors_val" : "/data/core_tensors_val.hdf5"
}



train_fc_layer(HDF5_files["img_train"], HDF5_files["text_train"], directories["frequent_embeddings"], directories["non_frequent_mappings"])