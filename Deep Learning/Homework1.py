import os
import pandas as pandas

Dataset_File = "WeatherData.csv"
if os.path.isfile(Dataset_File):
    os.system("wget https://raw.githubusercontent.com/cnchi/datasets/master/"+ Dataset_File) 

dfCSV = pd.read_csv("WeatherData.csv")