import unicodedata
from helper_functions import to_float_cuda, to_self_cuda, LstmAttention
import string
import numpy as np
import torch.nn as nn
import torch
from os import listdir
from os.path import isfile, join
import regex
import emoji
import torch.nn.functional as F

class CNN_NLP(nn.Module):
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
        super(CNN_NLP, self).__init__()
        num_classes = 3
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
        cov_out = torch.reshape(x_fc, (int(x_fc.shape[0]/max_tweets), -1, x_fc.shape[-1]))
        sum_out = torch.sum(cov_out, dim=1)
        #fc_out = self.fc(self.dropout(sum_out))
        return sum_out
        #logits = torch.squeeze(fc_out, 1)

def getRemainHandles(path):
    f = open(path)
    f.readline()
    handles = set()
    for line in f:
        line = line.strip()
        handle = line.split("\x1b")[0]
        handles.add(handle)
    return handles

def process_emojis(emojis):
    emojis_list = []
    if not emojis:
        return emojis_list
    for tweet_emoji in emojis.split(" "):
        emojis_words = tweet_emoji.lower().replace(":", "_").split("_")
        for word in emojis_words:
            if word:
                emojis_list.append(word)
    return emojis_list

def process_hashtags(hashtags):
    return hashtags.split(" ")

def load_text(mypath):
    """Load text data, lowercase text and save to a list."""
    handles = getRemainHandles("/home/yaguang/new_nonstop_onefeaturesword1.csv")
    ordered_handles = []
    max_len = 0
    tokenized_texts = []
    word2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1
    idx = 2
    count = 0
    import collections
    dic = collections.defaultdict(int)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #counter = []
    texts = []
    #labels = []
    lens = []
    for filename in onlyfiles:
        count += 1
        tweets = []
        handle = filename.lower().split(".")[0]
        if handle.startswith("hidden_"):
            handle = handle[6:]
        if handle not in handles:
            continue
        ordered_handles.append(handle)
        f = open(mypath+filename)
        for line in f:
            info = line.strip().split("\x1b")
            handle, tweet_id, tweet_date, tweet_text, mention_handles, emojis, emojis_texts, full_name, hashtags, hashtags_words, is_retweet, lang = info
            handle = handle.lower()
            #handle, tweet_date, tweet_text, emojis, hashtags, is_retweet, lang = info[0], info[2], info[3], info[4], info[5], info[-2], info[-1]
            if int(tweet_date[0:4]) < 2015:
                continue
            if is_retweet=="t":
                continue
            if lang != "en" and lang != "und":
                continue
            hashtags_words = process_hashtags(hashtags_words)
            emojis_texts = process_emojis(emojis_texts)
            if emojis_texts:
                dic[len(emojis_texts)] += 1
                tweets.append(emojis_texts[:longest_len])
            # Add new token to `word2idx`
                for token in emojis_texts[:longest_len]:
                    if token not in word2idx:
                        word2idx[token] = idx
                        idx += 1
                lens.append(len(emojis_texts))
                max_len = max(max_len, len(emojis_texts))
            if len(tweets) == max_tweets:
                break
        while len(tweets) < max_tweets:
            tweets.append(['<pad>'])
        f.close()
        texts.append(tweets)
    lens.sort()
    #print ("calculate the number of emojis")
    #print (dic)
    print ("calculate the avg number of emojis")
    print (sum(lens)/len(lens))
    print ("calculate the avg median, shortest, longest number of emojis")
    print (lens[int(len(lens)/2)], lens[0], lens[-1])
    max_len = min(max_len, longest_len)
    return texts, word2idx, max_len, ordered_handles

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
    #print (len(tokenized_texts), len(ordered_handles))
    for tokenized_sent in tokenized_texts:
        # Pad sentences to max_len
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

def get_handle2idx_embeddings(mypath):
    tokenized_texts, word2idx, max_len, ordered_handles = load_text(mypath)
    input_ids = encode(tokenized_texts, word2idx, max_len, ordered_handles)
    embeddings = load_pretrained_vectors(word2idx, "word_embedding/glove.6B.50d.txt")
    embeddings = torch.tensor(embeddings)
    #embeddings = None
    return embeddings, input_ids, len(word2idx)

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

#tokenized_texts, word2idx, max_len  = load_text(mypath)
#input_ids = encode(tokenized_texts, word2idx, max_len)
#embeddings = load_pretrained_vectors(word2idx, "../word_embedding/glove.6B.50d.txt")
#print (embeddings)
#embeddings = torch.tensor(embeddings)
