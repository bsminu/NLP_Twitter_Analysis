import pandas as pd

data = pd.read_csv("./data/data.csv")
print("Columns available:", data.columns)
print("No of tweets available: ", data.shape[0])