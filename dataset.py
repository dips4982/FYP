import os


directories = {
    "img_train" : "/data/train2014",
    "img_val" : "/data/val2014",
    "ques_train" : "/data/v2_OpenEnded_mscoco_train2014_questions.json",
    "ques_val" : "/data/v2_OpenEnded_mscoco_val2014_questions.json",
    "ans_train" : "/data/v2_mscoco_train2014_annotations.json",
    "ans_val" : "/data/v2_mscoco_val2014_annotations.json"
}

HDF5_files = {
    "text_train" : "/data/text_features_train.hdf5",
    "img_train" : "/data/image_features_train.hdf5",
    "core_tensors_train" : "/data/core_tensors_train.hdf5",

    "text_val" : "/data/text_features_val.hdf5",
    "img_val" : "/data/image_features_val.hdf5",
    "core_tensors_val" : "/data/core_tensors_val.hdf5"
}

def download_dataset(cwd, dataset_type = "images", dataset_part = "train"):
    if not os.path.isdir(cwd + "/data"):
        os.system('mkdir ' + cwd + '/data')

    if dataset_type == "images":
        if dataset_part == "train":
            if os.path.isdir(cwd + "/data/train2014"):
                print("Train dataset for Images already present")
            else:
                print("\t-----Downloading Training Images-----")
                os.chdir(cwd + "/data")
                os.system('wget http://images.cocodataset.org/zips/train2014.zip')
                os.system('unzip train2014.zip')
                os.system('rm train2014.zip')
        else:
            if os.path.isdir(cwd + "/data/val2014"):
                print("Val dataset for Images already present")
            else:
                print("\t-----Downloading Validation Images-----")
                os.chdir(cwd + "/data")
                os.system('wget http://images.cocodataset.org/zips/val2014.zip')
                os.system('unzip val2014.zip')
                os.system('rm val2014.zip')

    elif dataset_type == "questions":
        if dataset_part == "train":
            if os.path.isfile(cwd + "/data/v2_OpenEnded_mscoco_train2014_questions.json"):
                print("Train dataset for Questions already present")
            else:
                print("\t-----Downloading Training Questions-----")
                os.chdir(cwd + "/data")
                os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip')
                os.system('unzip v2_Questions_Train_mscoco.zip')
                os.system('rm v2_Questions_Train_mscoco.zip')

            if os.path.isfile(cwd + "/data/v2_mscoco_train2014_annotations.json"):
                print("Train dataset for Annotations already present")
            else:
                print("\t-----Downloading Training Annotations-----")
                os.chdir(cwd + "/data")
                os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip')
                os.system('unzip v2_Annotations_Train_mscoco.zip')
                os.system('rm v2_Annotations_Train_mscoco.zip')
        else:
            if os.path.isfile(cwd + "/data/v2_OpenEnded_mscoco_val2014_questions.json"):
                print("Val dataset for Questions already present")
            else:
                print("\t-----Downloading Validation Questions-----")
                os.chdir(cwd + "/data")
                os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip')
                os.system('unzip v2_Questions_Val_mscoco.zip')
                os.system('rm v2_Questions_Val_mscoco.zip')

            if os.path.isfile(cwd + "/data/v2_mscoco_val2014_annotations.json"):
                print("Val dataset for Annotations already present")
            else:
                print("\t-----Downloading Validation Annotations-----")
                os.chdir(cwd + "/data")
                os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip')
                os.system('unzip v2_Annotations_Val_mscoco.zip')
                os.system('rm v2_Annotations_Val_mscoco.zip')

                

        
