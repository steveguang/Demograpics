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

def process_tbn(X_train, y_train, X_test, y_test, X_test_mic, y_test_mic):
    # divide into train and corresponding handles
    X_train, handles_train = X_train[:, 1:].astype(np.float), X_train[:, :1].flatten()
    X_test, handles_test = X_test[:, 1:].astype(np.float), X_test[:, :1].flatten()

    numerical_train, embedding_bio_train, embedding_tweet_train, embedding_network_train = divide_data(X_train, bioLen, numLen, tweetLen)
    #print (embedding_bio_train.shape, embedding_tweet_train.shape)
    y_train = to_float_cuda(y_train.reshape(-1, 1)) if sourceType != SourceType.age else to_self_cuda(y_train)
    #embedding_train = cat_embeddings(numerical, embedding_bio, embedding_tweet)

    numerical_test, embedding_bio_test, embedding_tweet_test, embedding_network_test = divide_data(X_test, bioLen, numLen, tweetLen)
    y_test = to_float_cuda(y_test.reshape(-1, 1)) if sourceType == SourceType.age else to_self_cuda(y_test)
    #embedding_test = cat_embeddings(numerical, embedding_bio, embedding_tweet)

    bin_label = False if sourceType == SourceType.age else True

    if processType == ProcessType.mlp:
        embedding_train = cat_embeddings(numerical_train, embedding_bio_train, embedding_tweet_train)
        embedding_test = cat_embeddings(numerical_test, embedding_bio_test, embedding_tweet_test)
        model = MyMLP(embedding_train.shape[1], 20, D_out)
        to_cuda(model)
        train_model(model, embedding_train, y_train)
        auc = eval_model(model, embedding_test, y_test)
    elif processType == ProcessType.name_c_tbn:
        l_out = 8
        embedding_train = cat_embeddings(numerical_train, embedding_bio_train, embedding_tweet_train)
        embedding_test = cat_embeddings(numerical_test, embedding_bio_test, embedding_tweet_test)
        print (handles_train)
        train_names_idx, train_names_len = divide_name(handles_train, handles2names)
        lstm_model = NameLstmAttention(batch_size, hidden_size, embedding_length, l_out)
        model = LstmAttentionEnsemble(embedding_train.shape[1]+hidden_size, int(embedding_train.shape[1]+hidden_size/2), D_out, lstm_model, bin_label)
        to_cuda(model)
        train_model(model, embedding_train, train_names_idx, train_names_len, y_train)
        test_names_idx, test_names_len = divide_name(handles_test, handles2names)
        auc = eval_model(model, embedding_test, test_names_idx, test_names_len, y_test)
    elif processType == ProcessType.tbn_att:
        embedding_train = torch.stack((embedding_bio_train, embedding_tweet_train), axis=1)
        embedding_test = torch.stack((embedding_bio_test, embedding_tweet_test), axis=1)
        #embedding_test_mic = torch.stack((embedding_bio_test_mic, embedding_tweet_test_mic), axis=1)
        l_out = 8
        lstm_model = LstmAttention(batch_size, hidden_size, embedding_length, l_out)
        model = LstmAttentionEnsemble(l_out, int(l_out/2), D_out, lstm_model, bin_label)
        to_cuda(model)
        train_model(model, numerical_train, embedding_train, y_train)
        auc = eval_model(model, numerical_test, embedding_test, y_test)
        print (auc)
        auc = eval_model(model, numerical_test_mic, embedding_test_mic, y_test_mic)
        print (auc)
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

#def train_model(model, mlp_train, y_train, lstm_data=None, lstm_len=None, lstm_embedding_data=None):
def train_model(*args):
    model, y_train = args[0], args[-1]
    loss_fn = nn.CrossEntropyLoss() if sourceType == SourceType.age else nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for t in range(epochs):
        for i in range(0, len(y_train), batch_size):
            if processType == ProcessType.mlp:
                model, embedding_train, y_train = args
                y_pred = model(embedding_train[i:i+batch_size])
            elif processType == ProcessType.name_c_tbn:
                model, embedding_train, train_names_idx, train_names_len, y_train = args
                y_pred = model(embedding_train[i:i+batch_size], train_names_idx[i:i+batch_size], train_names_len[i:i+batch_size])
            elif processType == ProcessType.tbn_att or processType == ProcessType.tbnn_att:
                model, numerical_train, embedding_train, y_train = args
                y_pred = model(None, embedding_train[i:i+batch_size])
            elif processType == ProcessType.name:
                model, train_names_idx, train_names_len, y_train = args
                y_pred = model(None, train_names_idx[i:i+batch_size], train_names_len[i:i+batch_size])
            elif processType == ProcessType.tbn_c_name_att:
                model, numerical_train, train_names_idx, embedding_train, train_names_len, y_train = args
                y_pred = model(numerical_train[i:i+batch_size], embedding_train[i:i+batch_size], [(train_names_idx[i:i+batch_size], train_names_len[i:i+batch_size]) ])
            else:
                model, numerical_train, train_emoji_idx, embedding_train, y_train = args
                batch_train_emoji_idx = get_train_emojis(batch_size, train_emoji_idx, i)
                y_pred = model(numerical_train[i:i+batch_size], embedding_train[i:i+batch_size], [(batch_train_emoji_idx, None) ])

            loss = loss_fn(y_pred, y_train[i:i+batch_size])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if t % 100 == 99:
            print(t, loss.item())

#def eval_model(model, mlp_test, y_test, lstm_data=None, lstm_len=None):
def eval_model(*args):
    model = args[0]
    model.eval()
    if processType == ProcessType.mlp:
        model, embedding_test, y_test = args
        y_hat_test = model(embedding_test)
    elif processType == ProcessType.name_c_tbn:
        model, embedding_test, test_names_idx, test_names_len, y_test = args
        y_hat_test = model(embedding_test, test_names_idx, test_names_len)
    elif processType == ProcessType.tbn_att or processType == ProcessType.tbnn_att:
        model, numerical_test, embedding_test, y_test = args
        y_hat_test = model(None, embedding_test)
    elif processType == ProcessType.name:
        model, test_names_idx, test_names_len, y_test = args
        y_hat_test = model(test_names_idx, test_names_len)
    else:
        model, numerical_test, test_names_idx, embedding_test, test_names_len, y_test = args
        y_hat_test = model(numerical_test, embedding_test, [(test_names_idx, test_names_len)])
    y_hat_test_class = np.argmax(y_hat_test.cpu().detach().numpy(), axis=1) if sourceType == SourceType.age else np.where(y_hat_test.cpu().detach().numpy()<0.5, 0, 1)
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
    return X, y

def sample_data(X, y):
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

def read_data(filename):
    #tb = pd.read_csv("transdb_bert-uncased-nli/mergedAll.csv", sep=",")
    tb = pd.read_csv(filename, sep=",")
    network = pd.read_csv("/home/yaguang/network/network_embedding.csv", sep=" ", names=["handle"]+["network_"+str(i) for i in range(0, 768)], index_col=False)
    merged_features = pd.merge(tb, network, on='handle', how='left')
    merged_features = merged_features.fillna(0)
    merged_features = merged_features[tb.columns.tolist()[:-2]+network.columns.tolist()[1:]+tb.columns.tolist()[-2:]]
    merged_features = merged_features.sort_values('gender')
    print (merged_features)
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
data_type = DataType.tb
batch_size = 14315
sourceType = SourceType.gender
processType = ProcessType.name_c_tbn
processType = ProcessType.tbn_att
epochs = 1000
learning_rate = 0.0001
bins = [0, 35,  50, 120]
#bins = [0, 40, 120]
#bins = [0, 30, 40, 50, 120]
D_out = len(bins)-1 if sourceType == SourceType.age else 1
filename = "transdb_bert-uncased-nli/mergedAll.csv"

# set paras
"""
f = open(filename)
info = f.readline().strip().split(",")
length = len(info)
f.close()
"""
bioLen = 768
tweetLen = 768
numLen = 48
hidden_size, embedding_length =  512, 768
#fix_len = 32

# read data
dataSetWiki = read_data(filename)
X, y = split_train_test(dataSetWiki)
X, y = sample_data(X, y)

handles2names = get_handle2names("/home/yaguang/wiki_ground_truth.csv")
#embeddings, input_ids = get_handle2idx_embeddings("/home/yaguang/pattern/db/wiki_sort_emoji_hashtag/")

rocs = []
rocs_holdout = []
skf = StratifiedKFold(n_splits=10, shuffle=True)
count = 0
#X, name_map = X[:, :-1], X[:, -1]
print ("-----")
#print (name_map)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #X_train = np.concatenate((X_train, X_train_mic), axis=0)
    #y_train = np.concatenate((y_train, y_train_mic), axis=0)
    #count_label(y_test)
    #names_idx_train, names_idx_test = name_map[train_index], name_map[test_index]
    p = np.random.permutation(len(X_train))
    #X_train, y_train = X_train[p], y_train[p]
    #print ([v for v in y_test])
    auc = process_tbn(X_train, y_train, X_test, y_test, X_test_mic=None, y_test_mic=None)
    rocs.append(auc)
    print (auc)
    count += 1

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
