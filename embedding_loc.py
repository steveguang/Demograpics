import torch
import numpy as np
from os import listdir
from os.path import isfile, join
from sentence_transformers import SentenceTransformer
import regex
from sentence_transformers import models, losses
import random

#print(torch.cuda.current_device())
torch.cuda.set_device(0)
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

def get_model(path):
    word_embedding_model = models.Transformer(path)

# Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

def get_handles(path):
    handles = set()
    f = open(path)
    for line in f:
        handle = line.strip().split(",")[0]
        handles.add(handle)
    f.close()
    return handles

s = 0
e = 429695
handles = get_handles("location/world/bert-uncased-nli/emb_sum_user_info.train.csv")
print (len(handles), random.sample(handles, 1))
isTest = False
modelname = "bert-uncased-nli"
model =  SentenceTransformer("bert-base-nli-mean-tokens")
print("Max Sequence Length:", model.max_seq_length)

#Change the length to 200
model.max_seq_length = 300

print("Max Sequence Length:", model.max_seq_length)
path = "location/world/"
for filename in ["user_info.train","user_info.dev", "user_info.test"]:
    total1 = 0
    total2 = 0
    count = 0
    w_sum = open(path+modelname +"/emb_sum_"+filename+"_"+str(e)+".csv", "w")
    w_mean = open(path+modelname +"/emb_mean_"+filename+"_"+str(e)+".csv", "w")
    f = open(path+filename)
    for line in f:
        temp = []
        info = line.strip().split("\t")
        try:
            handle,  lan, lon, tweets = info
        except Exception as e:
            print(e, " ", handle)
            break
            continue
        if handle in handles:
            continue
        total1 += min(200, len(tweets.split("|||")))
        total2 += len(tweets.split("|||"))
        filter_tweet_texts = [filterText(tweet_text.lower()) for tweet_text in tweets.split("|||")[:200]]
        for filter_tweet_text in filter_tweet_texts:
            if filter_tweet_text:
                temp.append(filter_tweet_text)
        if not temp:
            continue
        handle = handle.lower()
        arr = model.encode(temp)
        """
        w2 = open(path+modelname+"/individual/"+handle+".csv", "w")
        for i in range(len(arr)):
            w2.write(",".join([str(num) for num in arr[i]])+"\n")
        w2.close()
        """
        sentencesSumVec = np.sum(arr, axis=0)
        sentencesMeanVec = np.mean(arr, axis=0)
        w_sum.write(",".join([handle]+[str(num) for num in sentencesSumVec]+[lan, lon])+"\n")
        w_mean.write(",".join([handle]+[str(num) for num in sentencesMeanVec]+[lan, lon])+"\n")
        count += 1
        print (count)
        #if count == e:
         #   break
    print (total1, total2)
    f.close()
    w_sum.close()
    w_mean.close()

