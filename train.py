import dataset
import extract_img_features as ex_img

directories = {
    "img_train" : "train2014",
    "img_val" : "val2014",
    "ques_train" : "v2_Questions_Train_mscoco",
    "ques_val" : "v2_Questions_Val_mscoco"
}

dataset.download_dataset("Images", "train")
dataset.download_dataset("Questions", "train")

ex_img.ImgExtractor().extract(directories["img_train"])
ex_text.TextFeatureExtractor().extract_features(directories["ques_train"])