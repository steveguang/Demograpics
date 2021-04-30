import string
import unicodedata
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from helper_functions import to_float_cuda, to_self_cuda, divide_data, to_cuda, under_sample, over_sample, MyMLP, LstmAttention, LstmAttentionEnsemble, MulLstmAttentionEnsemble
from name_seg import NameLstmAttention, divide_name, get_handle2names
#from emoji_hashtag_seg import get_handle2idx_embeddings, divide_emojis, get_train_emojis, CNN_NLP
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


class ProcessType:
    mlp = "mlp"
    name_c_tbn = "name_c_tbn"
    tbn_att = "tbn_att"
    name = "name"
    tbn_c_name_att = "tbn_c_name_att"
    tbnn_att = "tbnn_att"
    tbnn_e_att = "tbnn_e_att"

class DataType:
    t = "t"
    tb = "tb"
    tbn = "tbn"

class SampleType:
    under = "under"
    over = "over"
    n = "none"

class SourceType:
    age = "age"
    gender = "gender"
    race = "race"

def handle2names(filename):
    handles2names = {}
    f = open(filename)
    for line in f:
        info = line.strip().lower().split(",")
        handle, num_tweets, name = info[0], info[1], info[2]
        handles2names[handle] = (num_tweets, name)
    f.close()
    return handles2names

def cat_embeddings(numerical, embedding_bio, embedding_tweet):
    if data_type == DataType.tbn:
        embedding_cat = torch.cat((numerical, embedding_bio, embedding_tweet), axis=1)
    elif data_type == DataType.tb:
        embedding_cat = torch.cat((embedding_bio, embedding_tweet), axis=1)
    elif data_type == DataType.t:
        embedding_cat = embedding_tweet
    else:
        embedding_cat = 1/0
    return embedding_cat

#def process_tbn(X_train, y_train, X_test, y_test):
def process_tbn(X_test):
    # divide into train and corresponding handles
    X_test, handles_test = X_test[:, 1:].astype(np.float), X_test[:, :1].flatten()
    embedding_tweet_test = to_float_cuda(X_test)
    bin_label = False if sourceType == SourceType.age else True
    model = torch.load(model_name)
    numerical_test, embedding_bio_test = None, None
    print (X_test.shape)
    if processType == ProcessType.mlp:
        embedding_test = cat_embeddings(numerical_test, embedding_bio_test, embedding_tweet_test)
        #model = MyMLP(embedding_test.shape[1], 20, D_out)
        #to_cuda(model)
        auc = eval_model(model, embedding_test)
    elif processType == ProcessType.name_c_tbn:
        l_out = 8
        #embedding_train = cat_embeddings(numerical_train, embedding_bio_train, embedding_tweet_train)
        embedding_test = cat_embeddings(numerical_test, embedding_bio_test, embedding_tweet_test)
        #train_names_idx, train_names_len = divide_name(handles_train, handles2names)
        #lstm_model = NameLstmAttention(batch_size, hidden_size, embedding_length, l_out)
        #model = LstmAttentionEnsemble(embedding_train.shape[1]+hidden_size, int(embedding_train.shape[1]+hidden_size/2), D_out, lstm_model, bin_label)
        #to_cuda(model)
        #train_model(model, embedding_train, train_names_idx, train_names_len, y_train)
        handles2name = {}
        for handle in handles2names:
            handles2name[handle] = handles2names[handle][1]
        test_names_idx, test_names_len = divide_name(handles_test, handles2name)
        auc = eval_model(model, embedding_test, test_names_idx, test_names_len)
    elif processType == ProcessType.tbn_att:
        embedding_train = torch.stack((embedding_bio_train, embedding_tweet_train), axis=1)
        embedding_test = torch.stack((embedding_bio_test, embedding_tweet_test), axis=1)
        l_out = 8
        lstm_model = LstmAttention(batch_size, hidden_size, embedding_length, l_out)
        model = LstmAttentionEnsemble(numerical_train.shape[1]+l_out, int(numerical_train.shape[1]+l_out/2), D_out, lstm_model, bin_label)
        to_cuda(model)
        train_model(model, numerical_train, embedding_train, y_train)
        auc = eval_model(model, numerical_test, embedding_test, y_test)
    elif processType == ProcessType.name:
        l_out = 8
        train_names_idx, train_names_len = divide_name(handles_train, handles2names)
        lstm_model = NameLstmAttention(batch_size, hidden_size, embedding_length, l_out)
        model = LstmAttentionEnsemble(hidden_size, int(hidden_size/2), D_out, lstm_model, bin_label)
        to_cuda(model)
        train_model(model, train_names_idx, train_names_len, y_train)
        test_names_idx, test_names_len = divide_name(handles_test, handles2names)
        auc = eval_model(model, test_names_idx, test_names_len, y_test)
    elif processType == ProcessType.tbn_c_name_att:
        l_out = 8
        embedding_train = torch.stack((embedding_bio_train, embedding_tweet_train), axis=1)
        embedding_test = torch.stack((embedding_bio_test, embedding_tweet_test), axis=1)
        train_names_idx, train_names_len = divide_name(handles_train, handles2names)
        lstm_sub_model = NameLstmAttention(batch_size, 768, 1000, l_out)
        lstm_model = LstmAttention(batch_size, hidden_size, embedding_length, l_out)
        model = MulLstmAttentionEnsemble(numerical_train.shape[1]+l_out, int(embedding_train.shape[1]+hidden_size/2), D_out, [lstm_sub_model], lstm_model, bin_label)
        to_cuda(lstm_sub_model)
        to_cuda(lstm_model)
        to_cuda(model)
        train_model(model, numerical_train, train_names_idx, embedding_train, train_names_len, y_train)
        test_names_idx, test_names_len = divide_name(handles_test, handles2names)
        auc = eval_model(model, numerical_test, test_names_idx, embedding_test, test_names_len, y_test)
    elif processType == ProcessType.tbnn_att:
        l_out = 8
        embedding_train = torch.stack((embedding_bio_train, embedding_tweet_train, embedding_network_train), axis=1)
        embedding_test = torch.stack((embedding_bio_test, embedding_tweet_test, embedding_network_test), axis=1)
        lstm_model = LstmAttention(batch_size, hidden_size, embedding_length, l_out)
        model = LstmAttentionEnsemble(numerical_train.shape[1]+l_out, int(numerical_train.shape[1]+l_out/2), D_out, lstm_model, bin_label)
        to_cuda(model)
        train_model(model, numerical_train, embedding_train, y_train)
        auc = eval_model(model, numerical_test, embedding_test, y_test)
    elif processType == ProcessType.tbnn_e_att:
        emoji_embeddings, emoji_input_ids, dim = get_handle2idx_embeddings("/home/yaguang/pattern/db/wiki_sort_emoji_hashtag/")
        l_out = 8
        embedding_train = torch.stack((embedding_bio_train, embedding_tweet_train, embedding_network_train), axis=1)
        embedding_test = torch.stack((embedding_bio_test, embedding_tweet_test, embedding_network_test), axis=1)
        #emoji
        train_emoji_idx = divide_emojis(handles_train, emoji_input_ids)
        test_emoji_idx = divide_emojis(handles_test, emoji_input_ids)
        #emoji_cnn_model = CNN_NLP(pretrained_embedding=emoji_embeddings, dropout=0.5)
        emoji_cnn_model = CNN_NLP(vocab_size=dim)

        #lstm_sub_model = NameLstmAttention(batch_size, 768, 1000, l_out)
        lstm_model = LstmAttention(batch_size, hidden_size, embedding_length, l_out)
        model = MulLstmAttentionEnsemble(numerical_train.shape[1]+l_out, int(embedding_train.shape[1]+hidden_size/2), D_out, [emoji_cnn_model], lstm_model, bin_label)

        to_cuda(emoji_cnn_model)
        to_cuda(lstm_model)
        to_cuda(model)
        train_model(model, numerical_train, train_emoji_idx, embedding_train, y_train)
        test_names_idx, test_names_len = divide_name(handles_test, handles2names)
        auc = eval_model(model, numerical_test, test_emoji_idx, embedding_test, test_names_len, y_test)
    return auc

def eval_model(*args):
    model = args[0]
    model.eval()
    print (len(all_handles))
    if processType == ProcessType.mlp:
        #model, embedding_test, y_test = args
        model, embedding_test = args
        y_hat_test = model(embedding_test)
    elif processType == ProcessType.name_c_tbn:
        #model, embedding_test, test_names_idx, test_names_len, y_test = args
        model, embedding_test, test_names_idx, test_names_len= args
        y_hat_test = model(embedding_test, test_names_idx, test_names_len)
    elif processType == ProcessType.tbn_att or processType == ProcessType.tbnn_att:
        model, numerical_test, embedding_test, y_test = args
        y_hat_test = model(numerical_test, embedding_test)
    elif processType == ProcessType.name:
        model, test_names_idx, test_names_len, y_test = args
        y_hat_test = model(test_names_idx, test_names_len)
    else:
        model, numerical_test, test_names_idx, embedding_test, test_names_len, y_test = args
        y_hat_test = model(numerical_test, embedding_test, [(test_names_idx, test_names_len)])
    y_hat_test_class = np.argmax(y_hat_test.cpu().detach().numpy(), axis=1) if sourceType == SourceType.age else np.where(y_hat_test.cpu().detach().numpy()<0.5, 0, 1)
    vals = [str(val[0]) for val in y_hat_test.tolist()]
    print (y_hat_test.shape, len(all_handles))
    w = open("vov/"+model_name.split("/")[1]+".csv", 'w')
    w.write(",".join(["handle", "prob", "num_of_tweets"])+"\n")
    for i in range(len(all_handles)):
        w.write(",".join([all_handles[i], vals[i], handles2names[all_handles[i]][0]])+"\n")
    w.close()
    #f1 = f1_score(Y_test.reshape(-1,1).cpu(), y_hat_test_class )
    #print (torch.squeeze(y_test).cpu().detach().numpy())
    #print (y_test)
    y_test = torch.squeeze(y_test).cpu().detach().numpy()
    #print (y_test)
    #print (y_hat_test_class)
    #auc = roc_auc_score(y_test, y_hat_test_class, multi_class="ovr" )
    print (classification_report(y_test, y_hat_test_class))
    f1 = f1_score(y_test, y_hat_test_class, average='macro')
    return f1

# split into train and label
def split_train_test(dataSet):
    cut = -1 if sourceType == SourceType.race else -2
    X, y = (dataSet[:, :cut], dataSet[:, cut+1]) if sourceType == SourceType.age else (dataSet[:, :cut], dataSet[:, cut])
    #if processType == ProcessType.name_c_tbn or processType == ProcessType.name or  processType == ProcessType.tbn_c_name_att:
    #idx = np.array([[num for num in range(len(X))]])
    #X = np.concatenate((X, idx.T), axis=1)
    #X = X[:, 1:]
    y = y.astype('int')
    #y = np.array([int(val) for val in y])
    if sourceType == SourceType.age:
        y = LabelEncoder().fit_transform(pd.cut(y, bins, labels=range(len(bins)-1)))
    if sample_type == SampleType.under:
        X, y = under_sample(X, y)
    elif sample_type == SampleType.over:
        X, y = over_sample(X, y)
    else:
        if sourceType != SourceType.age:
            y = y.reshape(-1, 1)
    return X, y

def read_data():
    tb = pd.read_csv("transdb_bert-uncased/mergedAll.csv", sep=",")
    network = pd.read_csv("/home/yaguang/dl/network/network_embedding.csv", sep=" ", names=["handle"]+["network_"+str(i) for i in range(0, 768)], index_col=False)
    merged_features = pd.merge(tb, network, on='handle', how='left')
    merged_features = merged_features.fillna(0)
    merged_features = merged_features[tb.columns.tolist()[:-2]+network.columns.tolist()[1:]+tb.columns.tolist()[-2:]]
    dataSet = merged_features.values
    #handles, dataSet = merged_features.iloc[:,0:1].values, merged_features.iloc[:,1:].values
    #print (dataSet.shape)
    #return handles, dataSet
    return dataSet

def count_label(labels):
    import collections
    dic = collections.defaultdict(int)
    for v in labels:
        dic[v] += 1
    print (dic)

# choose what to input
sample_type = SampleType.under
valid = False
data_type = DataType.t
batch_size = 64
sourceType = SourceType.gender
processType = ProcessType.name_c_tbn
processType = ProcessType.mlp
epochs = 1
learning_rate = 0.0001
bins = [0, 35,  50, 120]
#bins = [0, 40, 120]
#bins = [0, 30, 40, 50, 120]
model_name = "saved_model/model_text_name"
model_name = "saved_model/model_mlp"
D_out = len(bins)-1 if sourceType == SourceType.age else 1
if sourceType == SourceType.race:
    filename = "transdb_bert-uncased/transDataRace.csv"
    filename = "transdb_bert-uncased-nli/transDataRace.csv"
else:
    filename = "vov/transdb_bert-uncased-nli/tweet_embedding.csv"

# set paras
all_handles = []
f = open(filename)
count = 0
for line in f:
    info = line.strip().split(",")
    count += 1
    length = len(info)
    all_handles.append(info[0])
f.close()
print (len(all_handles), count)
#dataSet = np.loadtxt(filename,  delimiter=",", usecols=np.arange(1, length))
dataSet = pd.read_csv(filename, sep=",", header=None)

X = dataSet.values
print (X.shape)
#handles2names = get_handle2names("/home/yaguang/feature_gen/wiki_ground_truth.csv")
handles2names = handle2names("/home/yaguang/vov/data_for_transfer.csv")
#embeddings, input_ids = get_handle2idx_embeddings("/home/yaguang/pattern/db/wiki_sort_emoji_hashtag/")

rocs = []
auc = process_tbn(X) #, names_idx_train, names_idx_test)
"""
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #count_label(y_test)
    #names_idx_train, names_idx_test = name_map[train_index], name_map[test_index]
    p = np.random.permutation(len(X_train))
    #X_train, y_train = X_train[p], y_train[p]
    #print ([v for v in y_test])
    auc = process_tbn(X_train, y_train, X_test, y_test) #, names_idx_train, names_idx_test)
    rocs.append(auc)
    print (auc)
    count += 1
    break
"""
"""
rocs_holdout = []
for i in range (5):
    embedding_training, Y_training = divide_data(X, y)
    model = MyEnsemble(embedding_training.shape[1], 20, 1)
    model.to(device)
    train_model(model, embedding_training, Y_training)

    embedding_holdout, Y_test = divide_data(X_holdout, Y_holdout)
    auc = eval_model(model, embedding_holdout, Y_test)
    rocs_holdout.append(auc)
"""
#print (sum(rocs)/len(rocs))
#print (rocs)
#print (rocs_holdout)
#print (rocs_holdout[rocs.index(max(rocs))])

print (sum(rocs)/len(rocs))
#print (rocs_holdout)
print (rocs)
