from os import listdir
from os.path import isfile, join

handle2year = {}
mypath = "wiki_data/wiki_sort_emoji_hashtag/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for filename in onlyfiles:
    f = open(mypath+filename)
    info = f.readline().lower().split("\x1b")
    if len(info) <= 1:
        continue
    handle, year = info[0], info[2].split(" ")[0].split("-")[0]
    handle2year[handle] = year
    f.close()

w = open("handle2year.csv", "w")
for handle in handle2year:
    w.write(",".join([handle, handle2year[handle]])+"\n")
w.close()
