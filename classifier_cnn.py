
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import f1_score
import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import torch
from helper_functions import SeqAttention
from helper_functions import map_handle_gt, under_sample
import regex
from nltk import word_tokenize
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import StratifiedKFold

bins = [0, 45]
races, genders, ages = map_handle_gt("query_race_attributes.csv")
#from ../sentence_embedding/helper_functions import LstmAttention
dev = "cuda:0"
device = torch.device(dev)

batch_size = 1000
max_tweets = 200
longest_len = 100
tk = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

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
    pattern = regex.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
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
        if int(age) <= 37:
            genders[handle.lower()] = 0
        else:
            genders[handle.lower()] = 1
    return genders

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

def map_index_to_file_label(onlyfiles):
    index_file = {}
    count = 0
    for filename in onlyfiles:
        handle = filename[:-4].lower()
        if handle not in remain_handles:
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

def load_text(mypath, filenames):
    """Load text data, lowercase text and save to a list."""
    #handles = getRemainHandles("/home/yaguang/new_nonstop_onefeaturesword1.csv")
    #genders = get_genders("/home/yaguang/feature_gen/wiki_ground_truth.csv")

    max_len = 0
    tokenized_texts = []
    word2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1
    idx = 2

    #onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    counter = []
    lens = []
    texts = []
    labels = []

    for filename in filenames:
        tweets = []
        handle = filename.lower().split(".")[0]
        if handle.startswith("hidden_"):
            handle = handle[6:]
        if handle not in remain_handles:
            continue
        if handle not in genders:
            continue
        labels.append(map_age_to_label(handle))
        f = open(mypath+filename)
        for line in f:
            info = line.strip().split("\x1b")
            handle, tweet_id, tweet_date, tweet_text, mention_handles, emojis, emojis_texts, full_name, hashtags, hashtags_words, is_retweet, lang = info
            if int(tweet_date[0:4]) < 2015:
                break
            if is_retweet=="t":
                continue
            if lang != "en": # and lang != "und":
                continue
            filter_tweet_text = filterText(tweet_text.lower())
            if filter_tweet_text:
                tokenized_sent =[]
                for token in tk.tokenize(filter_tweet_text):
                    if token.startswith("#"):
                        continue
                    tokenized_sent.append(token)
                lens.append(len(tokenized_sent))
                tweets.append(tokenized_sent[:longest_len])
            # Add new token to `word2idx`
                for token in tokenized_sent:
                    if token not in word2idx:
                        word2idx[token] = idx
                        idx += 1
                max_len = max(max_len, len(tokenized_sent))
                #if len(tokenized_sent) > 250:
                 #   print (filter_tweet_text)
            if len(tweets) == max_tweets:
                break
        counter.append(len(tweets))
        while len(tweets) < max_tweets:
            tweets.append(['<pad>'])
        f.close()
        texts.append(tweets)
    max_len = min(max_len, longest_len)
    lens.sort()
    #print (len(lens),lens.index(60),lens.index(70),lens.index(80),lens.index(90), lens.index(100), lens[0], lens[-1], lens[int(len(lens)/2)], lens[int(len(lens)*2/3)])
    #print (handle)
    #print (texts)
    #print ("-----")
    return texts, labels, word2idx, max_len, counter

# Load files
#mypath = "/home/yaguang/pattern/db/wiki_en_new/"

"""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
"""
from collections import defaultdict

def encode(tokenized_texts, word2idx, max_len):
    """Pad each sentence to the maximum sentence length and encode tokens to
    their index in the vocabulary.

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
            shape (N, max_len). It will the input of our CNN model.
    """

    input_ids = []
    for tokenized_sent in tokenized_texts:
        # Pad sentences to max_len
        tweet_ids = []
        for tweet in tokenized_sent:
            tweet += ['<pad>'] * (max_len - len(tweet))

            # Encode tokens to input_ids
            input_id = []
            for token in tweet:
                idx = word2idx.get(token) if token in word2idx else 1
                input_id.append(idx)
            #input_id = [word2idx.get(token) for token in tweet]
            tweet_ids.append(input_id)
        input_ids.append(tweet_ids)
    return np.array(input_ids)

from tqdm import tqdm_notebook

def load_pretrained_vectors(word2idx, fname):
    #w = open("missing_words.txt", 'w')
    print("Loading pretrained vectors...")
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #n, d = map(int, fin.readline().split())
    d = 50
    # Initilize random embeddings
    print (len(word2idx.values()), len(word2idx))
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx.values()), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    # Load pretrained vectors
    count = 0
    s = set()
    for line in tqdm_notebook(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        s.add(word)
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")
    word2idx_keys = word2idx.keys()
    """
    for word in word2idx_keys-s:
        w.write(word+"\n")
    w.close()
    """
    return embeddings


def data_loader(train_inputs, val_inputs, train_labels, val_labels,
                batch_size=50):
    """Convert train and validation sets to torch.Tensors and load them to
    DataLoader.
    """

    # Convert data type to torch.Tensor
    train_inputs, val_inputs, train_labels, val_labels =\
    tuple(torch.tensor(data) for data in
          [train_inputs, val_inputs, train_labels, val_labels])

    # Specify batch_size
    #batch_size = 200

    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs)
    #train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,  batch_size=batch_size)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_inputs)
    #val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    return train_dataloader, val_dataloader


class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=1,
                 dropout=0.5):
        super(CNN_NLP, self).__init__()
        num_classes = 1
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = torch.nn.Sequential(nn.Linear(np.sum(num_filters), num_classes), torch.nn.Sigmoid())
        self.dropout = nn.Dropout(p=dropout)
        self.seq_attention = SeqAttention(300, int(300/2/2), 1, int(300/2), True, True)

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        print (input_ids.shape)
        x_embed = self.embedding(input_ids).float()
        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        print ("input shape")
        print (x_embed.shape)
        x_reshaped = x_embed.permute(0, 2, 1)
        print (x_reshaped.shape)
        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        print (x_conv_list[0].shape)
        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        #print (x_pool_list[0].shape)
        #print (1/0)
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        # Compute logits. Output shape: (b, n_classes)
        #logits = self.fc(self.dropout(x_fc))
        #print (x_fc.shape)
        cov_out = torch.reshape(x_fc, (int(x_fc.shape[0]/max_tweets), -1, x_fc.shape[-1]))
        print (cov_out.shape)
        #cov_out = cov_out.permute(1,0,2)
        #print (cov_out.shape)
        #fc_out = self.seq_attention(None, cov_out)
        sum_out = torch.sum(cov_out, dim=1)
        print (sum_out.shape)
        print (1/0)
        fc_out = self.fc(self.dropout(sum_out))
        #print (sum_out.shape, fc_out.shape)
        logits = torch.squeeze(fc_out, 1)
        #print (logits.shape)
        #print (logits)
        #print ("-----")
        #print (1/0)
        return logits

def initilize_model(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3, 4, 5],
                    num_filters=[100, 100, 100],
                    num_classes=1,
                    dropout=0.5,
                    learning_rate=0.0005):
    """Instantiate a CNN model and an optimizer."""

    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = CNN_NLP(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=2,
                        dropout=0.5)
    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95)

    return cnn_model, optimizer

# Specify loss function
loss_fn = nn.BCELoss() #nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, optimizer, train_dataloader, train_label, val_dataloader=None, val_labels=None, epochs=10):
    """Train the CNN model."""
    # Tracking best validation accuracy
    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    #print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {\'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*60)

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        model.train()
        idx = 0
        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            #print (len(batch), batch[0].shape)
            #print (idx,idx+int(batch_size/max_tweets))
            b_input_ids = batch[0].to(device)  #tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            #conv_out = torch.reshape(model(b_input_ids), (max_tweets, -1))
            #print ("bacth input shape")
            #print (b_input_ids.shape)
            logits = model(b_input_ids)
            #print (conv_out.shape)
            b_labels = torch.FloatTensor(train_labels[idx:idx+int(batch_size/max_tweets)]).to(device)
            # Compute loss and accumulate the loss values
            #print ("*****")
            #print (logits.shape)
            #print (b_labels)
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()
            idx += int(batch_size/max_tweets)

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        # =======================================
        #               Evaluation
        # =======================================
        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader, val_labels)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

def evaluate(model, val_dataloader, val_labels):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    idx = 0
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        #b_input_ids, b_labels = tuple(t.to(device) for t in batch)
        b_input_ids = batch[0].to(device)
        b_labels = torch.FloatTensor(val_labels[idx:idx+int(batch_size/max_tweets)]).to(device)
        idx += int(batch_size/max_tweets)
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        #preds = torch.argmax(logits, dim=1).flatten()
        #print (logits.cpu().detach())
        #print (logits.cpu().detach().numpy())
        preds = np.where(logits.cpu().detach().numpy()<0.5, 0, 1)
        # Calculate the accuracy rate
        #accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        #print (preds == b_labels.cpu().detach().numpy())
        accuracy = np.array(preds == b_labels.cpu().detach().numpy()).mean() * 100
        accuracy = f1_score(b_labels.cpu().detach().numpy(), preds, average='macro')
        #print (accuracy)
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

map_attribute = map_gender_to_label
remain_handles = getRemainHandles("/home/yaguang/new_nonstop_onefeaturesword1.csv")
from sklearn.model_selection import train_test_split

mypath = "/home/yaguang/wiki_data/wiki_sort_emoji_hashtag/"

# Concatenate and label data
#tokenized_texts, labels, word2idx, max_len, counter = load_text(mypath, )

# Tokenize, build vocabulary, encode tokens
print("Tokenizing...\n")
#input_ids = encode(tokenized_texts, word2idx, max_len)
#print (input_ids.shape)
# Load pretrained vectors
#embeddings = load_pretrained_vectors(word2idx, "/home/yaguang/pretrained_models/glove.twitter.27B.50d.txt")
#embeddings = torch.tensor(embeddings)
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
index_to_file_label = map_index_to_file_label(onlyfiles)
index, labels = get_index_label(index_to_file_label)
index, labels = under_sample(index.reshape(-1, 1), labels)
index = index.ravel()
skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
for train_index, test_index in skf.split(index, labels):
    X_train_index, X_test_index = index[train_index], index[test_index]
    X_train_names = []
    X_test_names = []
    for idx in X_train_index:
        X_train_names.append(index_to_file_label[idx][0])
    for idx in X_test_index:
        X_test_names.append(index_to_file_label[idx][0])
    X_train_names = X_train_names[:100]
    X_test_names = X_test_names[:10]
    tokenized_texts, labels, word2idx, max_len, lengths = load_text(mypath, X_train_names+X_test_names)
    input_ids = encode(tokenized_texts, word2idx, max_len)
    embeddings = load_pretrained_vectors(word2idx, "/home/yaguang/pretrained_models/glove.twitter.27B.50d.txt")
    embeddings = torch.tensor(embeddings)
    length = len(X_train_names)
    train_inputs, val_inputs, train_labels, val_labels, train_counter, val_counter = input_ids[:length], input_ids[length:], labels[:length], labels[length:], lengths[:length], lengths[length:]
    #print (train_inputs.shape, val_inputs.shape)
    train_inputs = np.concatenate( train_inputs, axis=0 )
    val_inputs = np.concatenate( val_inputs, axis=0 )
    #print (train_inputs.shape, val_inputs.shape)
    train_dataloader, val_dataloader = \
    data_loader(train_inputs, val_inputs, train_labels, val_labels, batch_size=batch_size)
    cnn_static, optimizer = initilize_model(pretrained_embedding=embeddings,
                                        freeze_embedding=False,
                                        learning_rate=0.01,
                                        dropout=0.5)
    train(cnn_static, optimizer, train_dataloader, train_labels, val_dataloader, val_labels,epochs=100)
