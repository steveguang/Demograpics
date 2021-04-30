import unicodedata
from helper_functions import to_float_cuda, to_self_cuda, LstmAttention, process_emojis, process_hashtags
import string
import numpy as np
import torch.nn as nn
import torch
from os import listdir
from os.path import isfile, join
import regex
import emoji
import torch.nn.functional as F

class CNN(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[2, 3, 4],
                 num_filters=[256, 256, 256],
                 num_classes=1,
                 dropout=0.5):
        super(CNN, self).__init__()
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
        #self.fc = torch.nn.Sequential(nn.Linear(np.sum(num_filters), num_classes), torch.nn.Sigmoid())
        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        # Compute logits. Output shape: (b, n_classes)
        #logits = self.fc(self.dropout(x_fc))
        #print (x_fc.shape)
        #fc_out = self.fc(self.dropout(sum_out))
        return x_fc

def load_pretrained_vectors(word2idx, fname):
    w = open("missing_words.txt", 'w')
    print("Loading pretrained vectors...")
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #n, d = map(int, fin.readline().split())
    d = 50
    # Initilize random embeddings
    #embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx.values()), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    # Load pretrained vectors
    count = 0
    s = set()
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        s.add(word)
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)
    word2idx_keys = word2idx.keys()
    for word in word2idx_keys-s:
        w.write(word+"\n")
    w.close()
    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")
    return embeddings

def encode(tokenized_texts, word2idx, max_len, ordered_handles):
    input_ids = {}
    handle_idx = 0
    for tokenized_sent in tokenized_texts:
        tweet_ids = []
        for tweet in tokenized_sent:
            tweet += ['<pad>'] * (longest_len - len(tweet))
            # Encode tokens to input_ids
            input_id = []
            for token in tweet:
                idx = word2idx.get(token) if token in word2idx else 1
                input_id.append(idx)
            #input_id = [word2idx.get(token) for token in tweet]
            tweet_ids.append(input_id)
        input_ids[ordered_handles[handle_idx]] = np.array(tweet_ids)
        #print (ordered_handles[handle_idx], np.array(tweet_ids).shape)
        handle_idx += 1
    #return np.array(input_ids)
    return input_ids

def divide_emojis(handles, input_ids):
    handle_ids = [input_ids[handle] for handle in handles]
    #print (handles[:4])
    #print ([val.shape for val in handle_ids[:4]])
    return to_self_cuda(np.concatenate(handle_ids, axis=0))

def get_batch_emojis(batch_size, train_emoji_idx, idx):
    return train_emoji_idx[idx*max_tweets:(idx+batch_size)*max_tweets]

max_tweets = 200
longest_len = 20
mypath = "/home/yaguang/wiki_data/wiki_sort_emoji_hashtag/"

tokenized_texts, word2idx, max_len  = load_text(mypath)
#input_ids = encode(tokenized_texts, word2idx, max_len)
#embeddings = load_pretrained_vectors(word2idx, "../word_embedding/glove.6B.50d.txt")
#print (embeddings)
#embeddings = torch.tensor(embeddings)
