from helper_functions import process_emojis, to_self_cuda
import numpy as np
import torch

def load_pretrained_vectors(word2idx, fname="/home/yaguang/pretrained_models/glove.6B.50d.txt"):
    #w = open("missing_words.txt", 'w')
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
    """
    for word in word2idx_keys-s:
        w.write(word+"\n")
    w.close()
    """
    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")
    return torch.tensor(embeddings)

def read_word2idx(path="word2idx.csv"):
    f = open(path)
    word2idx = {}
    for line in f:
        word, idx = line.strip().split(",")
        word2idx[word] = int(idx)
    f.close()
    return word2idx

def process_tweet(handle, dates, longest_emoji_len, word2idx, fix_seq_len, filepath = "/home/yaguang/wiki_data/wiki_sort_emoji_hashtag/"):
    filename = filepath+handle+".csv"
    emoji_idx = []
    f = open(filename)
    #print ("-------")
    #print (handle)
    idx = 0
    for d in dates:
        info = f.readline().strip().split("\x1b")
        handle, tweet_id, tweet_date, tweet_text, mention_handles, emojis, emojis_texts, full_name, hashtags, hashtags_words, is_retweet, lang = info
        #print (tweet_date, d)
        while float("".join(tweet_date.split(" ")[1].split(":"))) != d:
            info = f.readline().strip().split("\x1b")
            handle, tweet_id, tweet_date, tweet_text, mention_handles, emojis, emojis_texts, full_name, hashtags, hashtags_words, is_retweet, lang = info
            continue
        temp = []
        words = process_emojis(emojis_texts, longest_emoji_len)
        for word in words:
            if word not in word2idx:
                word = "<unk>"
            temp.append(word2idx[word])
        emoji_idx.append(temp)
    if len(emoji_idx) != len(dates):
        print (emoji_idx)
        print (dates)
        print (len(emoji_idx), len(dates))
        print (filename)
        print (1/0)
    while len(emoji_idx) < fix_seq_len:
        emoji_idx.append([word2idx["<pad>"] for i in range(longest_emoji_len)])
    #print (np.array(emoji_idx).shape)
    return emoji_idx

"""
handle = "abcairns"
f = open("/home/yaguang/wiki_data/wiki_sort_emoji_hashtag/"+handle+".csv")
dates = []
for line in f:
    date = line.strip().split("\x1b")[2]
    dates.append(date)
    if len(dates) == 30:
        break
f.close()
process_tweet(handle, dates, 3)
"""
