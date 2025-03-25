import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#file ="PyTorch_results/commits_info_pairs.csv"
#data = pd.read_csv(file)
#data = data.where(data["issue_severity"] == "HIGH").dropna(inplace=False)
#print(data.values)

files = [
    "PyTorch_results/commits_info.csv",
    "OpenCV_results/commits_info.csv",
    "NumPy_results/commits_info.csv"
    ]

for file in files:
    data = pd.read_csv(file)
    data = data.where(data["issue_severity"] == "HIGH").dropna(inplace=False)
    data = data.values
    print(file.split("_")[0],":",len(data))
