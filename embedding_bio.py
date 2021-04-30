import torch
import numpy as np
from os import listdir
from os.path import isfile, join
from sentence_transformers import SentenceTransformer
import regex
from sentence_transformers import models, losses

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

path = "location/world/"
isTest = False
#modelname = "bert-base-nli-mean-tokens"
modelname = "bert-uncased-nli"
model =  SentenceTransformer("bert-base-nli-mean-tokens")
w = open(path+modelname+"/emb_bio.csv", "w")
#model =  SentenceTransformer(modelname)
texts = []
handles = []
f = open("/home/yaguang/location_world_user.csv")
f.readline()
for line in f:
    info = line.strip().split("\x1f")
    handle, bio_text = info[0].lower(), info[1].lower()
    dates = []
    filter_text = filterText(bio_text)
    if filter_text:
        texts.append(filter_text)
        handles.append(handle)

size = 1
idx = 0
print (len(handles))
for i in range(0, len(texts), size):
    arr = model.encode(texts[i:i+size])
    for i in range(len(arr)):
        w.write(",".join([handles[idx]]+[str(num) for num in arr[i]])+"\n")
        idx += 1
f.close()
w.close()

