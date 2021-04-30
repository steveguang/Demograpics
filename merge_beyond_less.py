root = "transdb_bert-uncased-nli/"
fless = open(root+"mergedAll.csv")
fmore = open(root+"mergedAllMore.csv")
w = open(root+"merge_everything.py", 'w')
w.write(fless.readline())
fmore.readline()
for line in fless:
    w.write(line)
for line in fmore:
    w.write(line)
fmore.close()
fless.close()
w.close()
