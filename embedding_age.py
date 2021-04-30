import torch
import numpy as np
from os import listdir
from os.path import isfile, join
from sentence_transformers import SentenceTransformer
import regex
from sentence_transformers import models, losses

#print(torch.cuda.current_device())
torch.cuda.set_device(3)
#print ("---------------")
def getRemainHandles(path):
    f = open(path)
    f.readline()
    handles = set()
    for line in f:
        line = line.strip()
        handle = line.split("\x1b")[0]
        handles.add(handle)
    return handles

def filterText(text):
    """
    Remove Twitter username handles from text.
    """
    pattern = regex.compile(
        r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)"
    )
    # Substitute handles with ' ' to ensure that text on either side of removed handles are tokenized correctly
    text = pattern.sub(" ", text)
    if text.isspace():
        return ""
    return text

def get_genders(filename):
    f = open(filename)
    f.readline()
    genders = {}
    for line in f:
        line = line.strip()
        name, age, gender, handle, verified = line.split("\x1b")
        if gender == "female":
            genders[handle.lower()] = "1"
        else:
            genders[handle.lower()] = "0"
    return genders

def get_model(path):
    word_embedding_model = models.Transformer(path)

# Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

s = 0
e = 21000
isTest = False
modelname = "bert-uncased-nli"
#model = get_model("/home/yaguang/dl/fine_tune_models/Bertconvid")
#modelname = "roberta-uncased"
model =  SentenceTransformer("bert-base-nli-mean-tokens")
handles = getRemainHandles("/home/yaguang/dl/new_nonstop_onefeaturesword1.csv")
genders = get_genders("/home/yaguang/feature_gen/wiki_ground_truth.csv")
w = open("transdb_"+modelname +"/transData"+str(e)+".csv", "w")
mypath = "/home/yaguang/pattern/db/wiki_en/" if not isTest else "../test/a55afcohen.csv"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()
count = 0
#model = SentenceTransformer("bert-base-nli-mean-tokens")
temp = []
for filename in onlyfiles[s:e]:
    handle = filename.lower().split(".")[0]
    if handle.startswith("hidden_"):
        handle = handle[6:]
    if handle not in handles:
        continue
    if handle not in genders:
        continue
    f = open(mypath+filename)
    temp = []
    dates = []
    for line in f:
        info = line.strip().split("\x1b")
        handle,  tweet_date, tweet_text, is_retweet, lang = info[0], info[2], info[3], info[8], info[9]
        if int(tweet_date[0:4]) < 2015:
            continue
        if is_retweet=="t":
            continue
        if lang != "en" and lang != "und":
            continue
        filter_tweet_text = filterText(tweet_text.lower())
        #print (filter_tweet_text)
        if filter_tweet_text:
            temp.append(filter_tweet_text)
            dates.append(tweet_date)
    #print (temp)
    if not temp:
        continue
    print (handle)
    handle = handle.lower()
    gender = genders[handle.lower()]
    w2 = open("transdb_"+modelname+"/individual/"+handle+".csv", "w")
    arr = model.encode(temp)
    for i in range(len(arr)):
        w2.write(",".join([dates[i]]+[str(num) for num in arr[i]])+"\n")
    w2.close()
    sentencesVec = np.sum(arr, axis=0)
    w.write(",".join([handle]+[str(num) for num in sentencesVec]+[gender])+"\n")
    f.close()
w.close()

