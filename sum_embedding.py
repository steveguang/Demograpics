from os import listdir
from os.path import isfile, join
import numpy as np

def string2float(vals):
    return [float(val) for val in vals]
#root = "transdb_bert-uncased-tweet/"
root = "transdb_bert-uncased-nli/"
subroot = "sorted_individual/"
mypath = root+subroot
"""
count_limits = [200]
ws = []
for i in range(len(count_limits)):
    ws.append(open(root+"sumtransData"+str(count_limits[i])+".csv", 'w'))
"""
w = open(root+"tweet_embedding_year.csv", 'w')
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for filename in onlyfiles:
    count = 1
    f = open(root+subroot+filename)
    info= f.readline().strip().split(",")
    latest_year, latest_emd = info[0].split(" ")[0].split("-")[0], info[1:]
    latest_emd = string2float(latest_emd)
    for line in f:
        info = line.strip().split(",")
        year, emd = info[0].split(" ")[0].split("-")[0], info[1:]
        emd = string2float(emd)
        if int(year) >= int(latest_year)-1 or count < 20:
            count += 1
            for i in range(len(latest_emd)):
                latest_emd[i] += emd[i]
        else:
            break
    for i in range(1):
        w.write(",".join([filename[:-4].lower()]+[str(emd) for emd in latest_emd])+"\n")
    f.close()
w.close()
"""
for i in range(len(count_limits)):
    ws[i].close()
"""
