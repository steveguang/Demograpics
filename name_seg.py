import unicodedata
from helper_functions import to_float_cuda, to_self_cuda, LstmAttention
import string
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

vocab = ['<pad>'] + list(string.printable)

class NameLstmAttention(LstmAttention):
    def __init__(self, batch_size, hidden_size, embedding_length, D_out):
        LstmAttention.__init__(self, batch_size, hidden_size, embedding_length, D_out)
        self.char_embeddings = nn.Embedding(len(vocab), embedding_length)

    def forward(self, input, seq_lens):
        embeds = self.char_embeddings(input)
        input = pack_padded_sequence(embeds, seq_lens, enforce_sorted = False, batch_first=True)
        output, (final_hidden_state, final_cell_state) = self.lstm(input) #, (h_0, c_0))
        final_output = final_hidden_state[-1]
        return final_output

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in vocab)

def embedAndPack(seqs):
    vectorized_seqs = []
    seq_lens = []
    fix_len = 32
    for seq in seqs:
        temp = []
        seq = unicodeToAscii(seq)
        for tok in seq:
            if not tok:
                continue
            if len(temp) == fix_len:
                break
            temp.append(vocab.index(tok))
        seq_lens.append([len(temp)])
        while len(temp) < fix_len:
            temp.append(0)
        vectorized_seqs.append(temp)
    return np.array(vectorized_seqs), np.array(seq_lens)

"""
def divide_name(names_idx, filename):
    idx2name = get_idx2name(filename)
    names = [idx2name[int(idx)] for idx in names_idx]
    names_idx, names_len = embedAndPack(names)
    names_idx = to_self_cuda(names_idx)
    names_len = names_len.flatten()
    return names_idx, names_len
"""

def divide_name(handles, handles2names):
    names = [handles2names[handle] for handle in handles]
    names_idx, names_len = embedAndPack(names)
    print (names_idx)
    names_idx = to_self_cuda(names_idx)
    names_len = names_len.flatten()
    return names_idx, names_len

def get_handle2names(filename):
    f = open(filename)
    f.readline()
    names = {}
    for line in f:
        line = line.strip()
        name, age, gender, handle, verified = line.split("\x1b")
        handle = handle.lower()
        first_name = name.split(" ")[0]
        names[handle.lower()] = first_name
    return names

def get_idx2name(filename):
    handles2names = get_names("/home/yaguang/wiki_ground_truth.csv")
    f = open(filename)
    f.readline()
    idx2name = {}
    count = 0
    for line in f:
        handle = line.strip().split(",")[0].lower()
        idx2name[count] = handles2names[handle]
        count += 1
    return idx2name
