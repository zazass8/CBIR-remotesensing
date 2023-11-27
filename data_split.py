import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# Split the dataset into training, validation, test
def train_val_test_split(path):
    _datasets = ["FAIR1M_partial", "RESISC45_partial", "Sentinel2_partial"]
    for dataset in _datasets:
        preprocess_dir = os.path.join(path + "data_preprocessed", dataset + "/")
        data_dir = os.path.join(path + "interview_datasets", dataset + "/")
        labels = [label for label in os.listdir(data_dir)[1:] if not label.endswith("labels")]
        sub_directories = [data_dir + label + "/" for label in labels]

        train = []
        lab_train = []
        val = []
        lab_val = []
        test = []
        lab_test = []

        for i in range(len(labels)):
            obj = os.listdir(sub_directories[i])
            obj = [sub_directories[i] + ob for ob in obj]

            train.extend(obj[:round(0.6*len(obj))])
            val.extend(obj[round(0.6*len(obj)):round(0.8*len(obj))])
            test.extend(obj[round(0.8*len(obj)):])

            lab_train.extend([labels[i]]*round(len(obj)*0.6))
            lab_val.extend([labels[i]]*round(len(obj)*0.2))
            lab_test.extend([labels[i]]*round(len(obj)*0.2))

        data_train = {"Images": train, "Labels": lab_train}
        data_val = {"Images": val, "Labels": lab_val}
        data_test = {"Images": test, "Labels": lab_test}

        df_train = pd.DataFrame(data = data_train)
        df_val = pd.DataFrame(data = data_val)
        df_test = pd.DataFrame(data = data_test)

        df_train.to_csv(os.path.join(preprocess_dir, "train.csv"))
        df_val.to_csv(os.path.join(preprocess_dir, "val.csv"))
        df_test.to_csv(os.path.join(preprocess_dir, "test.csv"))

