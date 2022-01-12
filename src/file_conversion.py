### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
###     Filename : file_conversion.py                       ### 
###     Description: This python file convert the csv       ###  
###     into a specfic json format.                         ### 
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

# Loading python libraries
import csv
import json
from collections import OrderedDict


def make_record(row):
    """ Function to create sub records for json record
    Args:
        row : each row in csv file
    Returns:
        dict: dict contains sub record
    """
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
results1 = []
sub_records = []
with open("./data/Huawei.csv", 'r', newline='') as csvfile, \
     open('./data/Huawei.json', 'w') as jsonfile:
    # Reads the csv file
    reader = csv.DictReader(csvfile, delimiter=',')
    # Makes sub record from each row in csv file and append to a list
    sub_records = [make_record(row) for row in reader]
    
    results.append(OrderedDict([("Twitter", sub_records)]))
    
    results1.append(OrderedDict([("Huawei", results[0])]))
    out = json.dumps(results1[0], indent=4)
    
    #Write the json data to the file
    jsonfile.write(out)
