import os

def download_dataset(cwd, dataset_type = "images", dataset_part = "train"):
    if not os.path.isdir("data"):
        os.system('mkdir ' + cwd + '/data')

    os.system('cd data')

    if dataset_type == "images":
        if dataset_part == "train":
            if os.path.isdir("train2014"):
                print("Train dataset for Images already present")
            else:
                os.system('wget http://images.cocodataset.org/zips/train2014.zip')
                os.system('unzip train2014.zip')
                os.system('rm train2014.zip')
        else:
            if os.path.isdir("val2014"):
                print("Val dataset for Images already present")
            else:
                os.system('wget http://images.cocodataset.org/zips/val2014.zip')
                os.system('unzip val2014.zip')
                os.system('rm val2014.zip')

    elif dataset_type == "questions":
        if dataset_part == "train":
            if os.path.isfile("v2_Questions_Train_mscoco.json"):
                print("Train dataset for Questions already present")
            else:
                os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip')
                os.system('unzip v2_Questions_Train_mscoco.zip')
                os.system('rm v2_Questions_Train_mscoco.zip')

            if os.path.isfile("v2_Annotations_Train_mscoco.json"):
                print("Train dataset for Annotations already present")
            else:
                os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip')
                os.system('unzip v2_Annotations_Train_mscoco.zip')
                os.system('rm v2_Annotations_Train_mscoco.zip')
        else:
            if os.path.isfile("v2_Questions_Val_mscoco.json"):
                print("Val dataset for Questions already present")
            else:
                os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip')
                os.system('unzip v2_Questions_Val_mscoco.zip')
                os.system('rm v2_Questions_Val_mscoco.zip')

            if os.path.isfile("v2_Annotations_Val_mscoco.json"):
                print("Val dataset for Annotations already present")
            else:
                os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip')
                os.system('unzip v2_Annotations_Val_mscoco.zip')
                os.system('rm v2_Annotations_Val_mscoco.zip')

                

        