def get_handle2emd(filename):
    handle2emd = {}
    f = open(filename)
    for line in f:
        info = line.strip().split(",")
        handle, emd = info[0], info[1:]
        handle2emd[handle] = emd
    f.close()
    return handle2emd

root = "transdb_bert-uncased-nli/"
handle2emd = get_handle2emd(root+"tweet_embedding_year.csv")
w = open(root+"merged_all_year.csv", 'w')
f = open(root+"mergedAll.csv")
header = f.readline()
w.write(header)
for line in f:
    info = line.strip().split(",")
    handle = info[0]
    bio = info[1:1+768]
    num = info[1+768:1+768+48]
    tweet = info[1+768+48:1+768+48+768]
    label = info[-2:]
    rep = handle2emd[handle]
    new_info = [handle]+bio+num+rep+label
    w.write(",".join(new_info)+"\n")
w.close()
