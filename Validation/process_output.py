"""
This file averages results together from muliple runs of a file. There
are several possible ways this can happen.
1. The runs are stored in different files with the same name
2. The runs are all stored in the same file
Either way, this takes the start of the output file name, and will average
the results with that name + _averaged.csv as the output file.
"""


import sys
import pandas as pd
import copy
from os import listdir
import numpy as np


def process_files(file_starter):
    """
    This function takes in all the files with a particular starting name
    and averages the four scores per token and overall
    """
    if "\\" in file_starter:
        list_dir = file_starter[:file_starter.rfind("\\")]
        file_starter = file_starter[file_starter.rfind("\\")+1:]
    else:
        list_dir = "."
    token_scores_dict = {}
    overal_scores = {"Accuracy":[],"F1-Score":[],"Precision":[],"Recall":[]}
    columns_to_fill = ["Accuracy","Precision","Recall","F1-Score", "Num Positive Examples", "Num Negative Examples"]
    for file in listdir(list_dir):
        if file.startswith(file_starter):
            print("looking at file",file)
            with open(list_dir+"\\"+file,"r",encoding="utf-8") as result_file:
                for line in result_file:
                    #If the line starts with one of the main scores, it is
                    #a total score
                    for key in overal_scores:
                        if line.startswith(key):
                            score = float(line.replace(key+":","").replace("\n",""))
                            overal_scores[key].append(score)
                    #If the line has a , this is most likely the score
                    #for a particular token
                    if "," in line and len(line.split(",")) == 7:
                        results= line.split(",")
                        token = results[0]

                        #TODO: what is this for again?
                        if results[1] == "accuracy":
                            continue

                        #add the token if it is not already in there
                        if token not in token_scores_dict.keys():

                            token_scores_dict[token] = {}
                            for col in columns_to_fill:
                                token_scores_dict[token][col] = []

                        #the rest of the line should fill the dictionary in order
                        for i in range(len(columns_to_fill)):
                            token_scores_dict[token][columns_to_fill[i]].append(float(results[i+1]))
    #average the results
    for token in token_scores_dict.keys():
        for col in token_scores_dict[token].keys():
            token_scores_dict[token][col] = np.mean(token_scores_dict[token][col])

    overal_var = {}
    for score in overal_scores.keys():
        overal_var[score] = np.var(overal_scores[score])
        overal_scores[score] = np.mean(overal_scores[score])

    overal_scores["Num Positive Examples"] = -1
    overal_scores["Num Negative Examples"] = -1
    overal_var["Num Positive Examples"] = -1
    overal_var["Num Negative Examples"] = -1
    token_scores_dict["ALL_TOKENS"] = overal_scores
    print("Overal scores:")
    for sType,score in overal_scores.items():
        print(sType,":",score)
        print(sType,"Variance:",overal_var[sType])
    token_scores_frame = pd.DataFrame(token_scores_dict).transpose()
    #print(token_scores_frame)
    token_scores_frame.to_csv(list_dir+"\\"+file_starter+"_averaged.csv",encoding="utf-8")

if __name__ == "__main__":
        file_starter = sys.argv[1]
        process_files(file_starter)
