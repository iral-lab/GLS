import sys
import os
import numpy as np

command = "for i in {1..25}; do python2 macro-pos5DescrNegDocVecdistractorTest.py ../../Results_75_2min/NoOfDataPoints/ object object ../conf_files/UW_english/UW_AMT_description_documents_per_image_nopreproc_stop_raw.conf <<< {} | tee -a UW_raw_75_object_{}.csv; done"

metric = "F1"

for num in np.arange(0.0, 1.0, 0.05):
    command = "for i in {1..25}; do python2 macro-pos5DescrNegDocVecdistractorTest.py ../../Results_75_2min/NoOfDataPoints/ object object ../conf_files/UW_english/UW_AMT_description_documents_per_image_nopreproc_stop_raw.conf <<< " + str(num) + " | tee -a UW_raw_75_object_"+  str(num) +".csv; done"

    command = "python read_in_results.py --file UW_raw_75_object_"+ str(num) + ".csv --metric " + metric

    line = str(num) + ","

    print(line, end="")

    os.system(command)
