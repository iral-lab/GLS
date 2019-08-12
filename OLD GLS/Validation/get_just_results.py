import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file',help='file_location',required=True)

args = parser.parse_args()

file_name = args.file

file = open(file_name)

avoid = ["Threshold", "Folder Number", "Accuracy", "F1", "Precision", "Recall", "input"]

for line in file.readlines():
    flag = False
    for item in avoid:
        if item in line:
            flag = True
        
    if flag:
        continue


    print(line.strip())
