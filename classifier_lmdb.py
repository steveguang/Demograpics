import random
import lmdb
import time
import pickle
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,  f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from os import listdir
from os.path import isfile, join
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from helper_functions import SeqAttention
from helper_functions import to_cuda, to_float_cuda, to_self_cuda, under_sample, get_remain_handles, map_handle_gt

dev = "cuda:0"
device = torch.device(dev)

class InferenceType:
    age = "age"
    race = "race"
    gender = "gender"

class BinType:
    two = "two"
    three = "three"
    four = "four"

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names):
        'Initialization'
        self.names = names

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

  def __getitem__(self, index):
        handle = self.names[index][:-4]
        pick_emd = txn.get(handle.encode())
        temp = pickle.loads(pick_emd)
        tweet_emb = to_float_cuda(temp[:fix_seq_len])
        print (handle)
        #print(tweet_emb[0])
        return tweet_emb, gt[handle][0]


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    print (xx[0])
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    print (xx_pad[0])
    return xx_pad, yy, x_lens

def train_model2(model, loss_fn, optimizer, train_names, batch_size):
    params = {'batch_size': 64, 'shuffle': True, 'num_workers': 8, 'collate_fn':pad_collate}
    training_set = Dataset(train_names)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    for epoch in range(1000):
        for X_batch_train, Y_batch_train, seq_lens in training_generator:
            X_batch_train, Y_batch_train = to_float_cuda(Y_batch_train), to_float_cuda(Y_batch_train).reshape(-1, 1)
            print (X_batch_train[0:3])
            print (Y_batch_train[0:3])
            y_pred = model(X_batch_train, seq_lens)
            loss = loss_fn(y_pred, Y_batch_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch%1 >= 0:
                print ("epoch "+str(epoch)+" with batch " + str(idx/batch_size) + " is "+str(loss.item()))
    #return loss.item()

"""
def padSequence(batch):
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    lengths = torch.LongTensor([len(x) for x in sequences])
    labels = torch.LongTensor(map(lambda x: x[1], sorted_batch))
    return sequences_padded, lengths, labels
"""

def map_race_to_label(handle):
    if handle not in races:
        return -1
    val = 1 if races[handle] == "black" else 0
    return val

def map_gender_to_label(handle):
    if handle not in genders:
        return -1
    if genders[handle] == "female":
        val = 1
    elif genders[handle] == "male":
        val = 0
    else:
        val = -1
    return val

def map_num_to_label(num, category):
    for i in range(len(category)-1):
        if category[i]<=num<category[i+1]:
            return i
    return len(category)-1

def map_age_to_label(handle):
    if handle not in ages:
        return -1
    val = map_num_to_label(ages[handle], bins)
    return val


def map_index_to_file_label(onlyfiles):
    index_file = {}
    count = 0
    for filename in onlyfiles:
        handle = filename[:-4].lower()
        """
        if handle not in gt:
            print (handle)
            continue
        """
        if handle not in remainHandles:
            continue
        val = map_attribute(handle)
        if val == -1:
            continue
        index_file[count] = (filename, val)
        count += 1
    return index_file

def get_index_label(index_to_file_label):
    index = []
    labels = []
    for idx in index_to_file_label:
        index.append(idx)
        labels.append(index_to_file_label[idx][1])
    index, labels = np.array(index), np.array(labels)
    return index, labels

def train_model(model, train_names, test_names):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss() if inference_type == InferenceType.age else nn.BCELoss()
    #model.train()
    X_test = []
    y_test = []
    seq_lens_test = []
    counter = [0, 0, 0 ,0 ,0]
    for i in range(len(test_names)):
        handle = test_names[i][:-4].lower()
        pick_emd = txn.get(handle.encode())
        temp = pickle.loads(pick_emd)
        dates = [val[0] for val in temp[:fix_seq_len]][::-1]
        tweet_emb = [val[1:] for val in temp[:fix_seq_len]]
        seq_lens_test.append(len(tweet_emb))
        while len(tweet_emb) < fix_seq_len:
            tweet_emb.append([0 for i in range(768)])
        X_test.append(tweet_emb)
        y_test.append(map_attribute(handle))
        counter[map_attribute(handle)] += 1
    print (counter)
    X_test = to_float_cuda(X_test)
    X_test = X_test.permute(1, 0, 2)
    y_test = to_float_cuda(y_test).reshape(-1, 1) if inference_type != InferenceType.age else to_self_cuda(y_test)
    for epoch in range(epochs):
        idx = 0
        counter = [0, 0, 0 ,0 ,0]
        model.train()
        train_names = random.sample(train_names, len(train_names))
        while idx < len(train_names):
            batch_train_names = train_names[idx:idx+batch_size]
            X_batch_train = []
            y_batch_train = []
            seq_lens = []
            end_time = time.time()
            for i in range(len(batch_train_names)):
                handle = batch_train_names[i][:-4].lower()
                pick_emd = txn.get(handle.encode())
                temp = pickle.loads(pick_emd)[:fix_seq_len]
                tweet_emb = [val[1:] for val in temp]
                #tweet_emb = temp[:fix_seq_len]
                seq_lens.append(len(tweet_emb))
                while len(tweet_emb) < fix_seq_len:
                    tweet_emb.append([0 for i in range(768)])
                X_batch_train.append(tweet_emb)
                y_batch_train.append(map_attribute(handle))
                counter[map_attribute(handle)] += 1
            X_batch_train = to_float_cuda(X_batch_train)
            X_batch_train = X_batch_train.permute(1, 0, 2)
            #X_batch_train = pack_padded_sequence(X_batch_train)
            y_batch_train = to_float_cuda(y_batch_train).reshape(-1, 1) if inference_type != InferenceType.age else to_self_cuda(y_batch_train)
            #y_batch_train = to_float_cuda(y_batch_train).reshape(-1, 1)
            y_pred = model(None, X_batch_train, seq_lens)
            loss = loss_fn(y_pred, y_batch_train)
            #print (y_pred, y_batch_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            idx += batch_size
            print ("epoch "+str(epoch)+" with batch " + str(int(idx/batch_size)) + " is "+str(loss.item()))
            #print ("train f1")
            #print (eval_mem_model(model,X_batch_train, y_batch_train, seq_lens))
            #print ("test f1")
        print (eval_mem_model(model,X_test, y_test, seq_lens_test))
        #break
        auc = eval_model(model, X_test_names)
        print (auc)
    return loss.item()

def eval_model(model, test_names):
    idx = 0
    y_hat_test_class = []
    y_test = []
    model.eval()
    while idx < len(test_names):
        batch_test_names = test_names[idx:idx+batch_size]
        X_batch_test = []
        seq_lens = []
        for i in range(len(batch_test_names)):
            handle = batch_test_names[i][:-4].lower()
            temp =  pickle.loads(txn.get(handle.encode()))[:fix_seq_len]
            #tweet_emb = pickle.loads(txn.get(handle.encode()))[:fix_seq_len]
            tweet_emb = [val[1:] for val in temp]
            seq_lens.append(len(tweet_emb))
            while len(tweet_emb) < fix_seq_len:
                tweet_emb.append([0 for i in range(768)])
            X_batch_test.append(tweet_emb)
            y_test.append(map_attribute(handle))
        X_batch_test = to_float_cuda(X_batch_test)
        X_batch_test = X_batch_test.permute(1, 0, 2)
        y_hat_test = model(None, X_batch_test, seq_lens)
        #print (y_hat_test)
        #print (y_hat_test.cpu().detach().numpy())
        if inference_type == InferenceType.age:
            target = np.argmax(y_hat_test.cpu().detach().numpy(), axis=1)
            for i in range(len(target)):
                y_hat_test_class.append(target[i])
        else:
            target = np.where(y_hat_test.cpu().detach().numpy()<0.5, 0, 1)
            for i in range(len(target)):
                y_hat_test_class.append(target[i][0])
        #print ("epoch "+str(idx))
        idx += batch_size
    #print (y_test, y_hat_test_class)
    f1 = f1_score(y_test, y_hat_test_class, average='macro')
    #auc = roc_auc_score(y_test, y_hat_test_class )
    return f1

def eval_mem_model(model, X_batch_test, y_batch_test, seq_lens):
    idx = 0
    y_hat_test_class = []
    y_test = y_batch_test.cpu().detach().numpy()
    model.eval()
    y_hat_test = model(None, X_batch_test, seq_lens)
    if inference_type == InferenceType.age:
        target = np.argmax(y_hat_test.cpu().detach().numpy(), axis=1)
        for i in range(len(target)):
            y_hat_test_class.append(target[i])
    else:
        target = np.where(y_hat_test.cpu().detach().numpy()<0.5, 0, 1)
        for i in range(len(target)):
            y_hat_test_class.append(target[i][0])
    #print (y_test, y_hat_test_class)
    f1 = f1_score(y_test, y_hat_test_class, average='macro')
    model.train()
    return f1

# Create random Tensors to hold inputs and outputs
#r = np.arange(1,775)
#r = np.arange(0,769)

env = lmdb.open('/home/yaguang/lmdb/dir/')
txn = env.begin(write=False)

inference_type = InferenceType.age
if inference_type == InferenceType.race:
    mypath = "/home/yaguang/transdb_bert-uncased-nli/sorted_individual_race/"
    map_attribute = map_race_to_label
elif inference_type == InferenceType.gender:
    mypath = "/home/yaguang/transdb_bert-uncased-nli/sorted_individual"
    map_attribute = map_gender_to_label
elif inference_type == InferenceType.age:
    mypath = "/home/yaguang/transdb_bert-uncased-nli/sorted_individual"
    print ("hello!")
    map_attribute = map_age_to_label
else:
    print (1/0)

bin_type = BinType.three
if bin_type == BinType.two:
    bins = [0, 45]
elif bin_type == BinType.three:
    bins = [0, 35, 55]
elif bin_type == BinType.four:
    bins = [0, 30, 40, 50]
else:
    print (1/0)

D_out = len(bins) if inference_type == InferenceType.age else 1

races, genders, ages = map_handle_gt("query_race_attributes.csv")
epochs = 20
remainHandles = get_remain_handles("/home/yaguang/new_nonstop_onefeaturesword1.csv")
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
fix_seq_len = 200
batch_size = 32
rocs = []

index_to_file_label = map_index_to_file_label(onlyfiles)
index, labels = get_index_label(index_to_file_label)
usecols = list(np.arange(1,769))
counter = [0, 0, 0, 0]
index, labels = under_sample(index.reshape(-1, 1), labels)
index = index.ravel()
for label in labels:
    counter[label] += 1
print (counter)
w = open("metric_result.txt", 'w')
skf = StratifiedKFold(n_splits=10, shuffle=True)
learning_rate = 0.0001

sample_files = open("sample_files.txt", 'w')
counter = [0, 0, 0, 0]
for idx in index:
    counter[index_to_file_label[idx][1]] += 1
    sample_files.write(index_to_file_label[idx][0]+"\n")
sample_files.close()
print (counter)

for train_index, test_index in skf.split(index, labels):
    X_train_index, X_test_index = index[train_index], index[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    #print (y_train.tolist())
    y_train = to_float_cuda(y_train)
    y_test = to_float_cuda(y_test)
    X_train_names = []
    X_test_names = []
    for idx in X_train_index:
        X_train_names.append(index_to_file_label[idx][0])
    for idx in X_test_index:
        X_test_names.append(index_to_file_label[idx][0])

    model = SeqAttention(768, int(768/2/2), D_out, int(768/2), False, True)
    to_cuda(model)
    train_model(model, X_train_names, X_test_names)

    auc = eval_model(model,X_test_names)
    print (auc)
    w.write(str(auc)+"\n")
    rocs.append(auc)
    print ("another epoch")
print (rocs)
print (sum(rocs)/len(rocs))
w.close()
