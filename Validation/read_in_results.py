import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file',help='file_location',required=True)
parser.add_argument('--metric', help='metric')

args = parser.parse_args()

file_name = args.file
metric = args.metric

file = open(file_name)

sum = 0

#print(metric)
for line in file.readlines():
    if metric in line:
        #printLine = line[10:].strip()
        print(line.strip())
        #print(printLine)
        #sum += float(printLine)

print(str(sum / 25))
