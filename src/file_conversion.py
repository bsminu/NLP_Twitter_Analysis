
# import pandas as pd
# from itertools import groupby 
# from collections import OrderedDict
# import json  

# data = pd.read_csv("./data/Huawei.csv")
# print("Columns available:", data.columns)
# print("No of tweets available: ", data.shape[0])

# # results = []

# for (company,channel), bag in data.groupby(["company","channel"]):
#     contents_df = bag.drop(["company","channel"], axis=1)
#     subset = [OrderedDict(row) for i,row in contents_df.iterrows()]
#     results.append(OrderedDict([("company", company),
#                                 ("channel", channel),
#                                 ("subset", subset)]))
#     print(json.dumps(results[0]))
#     break



import csv
import json
from collections import OrderedDict

def make_record(row):
    return {
                "tweet_created": row["tweet_created"],
                "tweets":[
                    {
                        "tweet_id": row["tweet_id"],
                        "type": row["type"],
                        "tweet": row["tweet"],
                        "metrics": [
                            {
                                "favorite_count": row["favorite_count"],
                                "replies_count": row["replies_count"],
                                "retweet_count": row["retweet_count"],
                                "unmetric_engagement": row["unmetric_engagement_score"]
                            }
                        ],
                        "url": row["url"]
                    }
                ]
            }

results = []
with open("./data/Huawei.csv", 'r', newline='') as csvfile, \
     open('./data/Huawei.json', 'w') as jsonfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    sub_records = [make_record(row) for row in reader]
    
    results.append(OrderedDict([("Huawei", "Huawei"),
                                ("Twitter", sub_records)]))
    
    out = json.dumps(results[0], indent=4)
    jsonfile.write(out)
