"""
generate a txt with word and idx for emoji
"""

from helper_functions import get_remain_handles, process_emojis, process_hashtags, get_files_under_dir
import emoji

def load_text(mypath):
    """Load text data, lowercase text and save to a list."""
    handles = get_remain_handles("/home/yaguang/new_nonstop_onefeaturesword1.csv")
    word2idx = {}
    lens = []
    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1
    idx = 2
    onlyfiles = get_files_under_dir(mypath)
    for filename in onlyfiles:
        count = 0
        handle = filename.lower().split(".")[0]
        if handle.startswith("hidden_"):
            handle = handle[6:]
        if handle not in handles:
            continue
        f = open(mypath+filename)
        for line in f:
            info = line.strip().split("\x1b")
            handle, tweet_id, tweet_date, tweet_text, mention_handles, emojis, emojis_texts, full_name, hashtags, hashtags_words, is_retweet, lang = info
            handle = handle.lower()
            if int(tweet_date[0:4]) < 2015:
                continue
            if is_retweet=="t":
                continue
            if lang != "en" and lang != "und":
                continue
            hashtags_words = process_hashtags(hashtags_words)
            emojis_texts_list = process_emojis(emojis_texts)
            if emojis_texts_list:
                #print (emojis)
                #print (emojis_texts)
                for token in emojis_texts_list:
                    if token not in word2idx:
                        word2idx[token] = idx
                        idx += 1
                lens.append(len(emojis_texts.split(" ")))
            count += 1
        f.close()
    """
    lens.sort(reverse=True)
    print (len(lens))
    for i in range(1, 20):
        print (len(lens)-lens.index(i))
    """
    w = open("word2idx.csv", 'w')
    for word in word2idx:
        w.write(",".join([word, str(word2idx[word])])+"\n")
    w.close()
    return word2idx

#longest_len, max_tweets = 20, 250
load_text("/home/yaguang/wiki_data/wiki_sort_emoji_hashtag/")
