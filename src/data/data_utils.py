import numpy as np
import os
import pandas as pd

class Data():
    def __init__(self, path):
        pass

    def get_numpy_data():
        pass

    def get_pytorch_dataset():
        pass



## TODO
class GoEmotionsData(Data):

    def __init__(self, path):
        
        csv1_file_path = os.path.join(path, "goemotions_1.csv")
        csv2_file_path = os.path.join(path, "goemotions_2.csv")
        csv3_file_path = os.path.join(path, "goemotions_3.csv")

        data1 = pd.read_csv(csv1_file_path)
        data2 = pd.read_csv(csv2_file_path)
        data3 = pd.read_csv(csv3_file_path)

        data = pd.concat([data1, data2, data3])

        print(data)

    def get_numpy_data():
        pass

    def get_pytorch_dataset():
        pass


## TODO
class EmotionsData(Data):

    def __init__(self, path):
        pass

    def get_numpy_data():
        pass

    def get_pytorch_dataset():
        pass



## TODO
class AppReviewsData(Data):

    def __init__(self, path):
        pass

    def get_numpy_data():
        pass

    def get_pytorch_dataset():
        pass

if __name__ == "__main__":

    data = GoEmotionsData(path="/home/florian/Dokumente/Programmierung/Python/NLP/NLP_Project/data/goEmotions/data/full_dataset")

