import torch
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import unicodedata
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from os import listdir
from os.path import isfile, join
import emoji

CUDA_LAUNCH_BLOCKING=1
dev = "cuda:0"
device = torch.device(dev)

ros = RandomOverSampler()
rus = RandomUnderSampler()


class MyMLP(nn.Module):
    def __init__(self, D_in, H, D_out, bin_label):
        super(MyMLP, self).__init__()
        layers = [
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    ]
        if bin_label:
            layers.append(torch.nn.Sigmoid())
        #else:
         #   layers.append(torch.nn.Softmax(dim=1))
        self.classifier = torch.nn.Sequential(*layers)
    def forward(self, x):
        x = self.classifier(x)
        return x

class LstmAttentionEnsemble(nn.Module):
    def __init__(self, D_in, H, D_out, lstm_model, bin_label):
        super(LstmAttentionEnsemble, self).__init__()
        layers = [
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    ]
        if bin_label:
            layers.append(torch.nn.Sigmoid())
        #else:
         #   layers.append(torch.nn.Softmax(dim=1))
        self.classifier = torch.nn.Sequential(*layers)
        self.lstm_model = lstm_model

    def forward(self, x=None, input=None, seq_lens=None):
        out = self.lstm_model(input, seq_lens)
        if x is not None:
            out = torch.cat((out, x), dim=1)
        out = self.classifier(out)
        return out

class MulLstmAttentionEnsemble(nn.Module):
    def __init__(self, D_in, H, D_out, lstm_sub_models, lstm_model, bin_label):
        super(MulLstmAttentionEnsemble, self).__init__()
        layers = [
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    ]
        if bin_label:
            layers.append(torch.nn.Sigmoid())
        else:
            layers.append(torch.nn.Softmax(dim=1))
        self.classifier = torch.nn.Sequential(*layers)
        self.lstm_model = lstm_model
        #self.lstm_sub_model = lstm_sub_models
        self.lstm_sub_models = lstm_sub_models

    def forward(self, x=None, file_input_data=None, model_input_data=None):
        outputs = []
        for i in range(len(self.lstm_sub_models)):
            lstm_sub_model = self.lstm_sub_models[i]
            input, seq_lens = model_input_data[i]
            if seq_lens is not None:
                lstm_out = lstm_sub_model(input, seq_lens)
            else:
                lstm_out = lstm_sub_model(input)
            outputs.append(lstm_out)
        out = torch.cat([out.reshape(out.shape[0], 1, -1) for out in outputs] + [file_input_data], 1)
        out = self.lstm_model(out)
        if x is not None:
            out = torch.cat((out, x), dim=1)
        out = self.classifier(out)
        return out

class LstmAttention(nn.Module):
    def __init__(self, batch_size, hidden_size, embedding_length, D_out, use_attention=False):
        super(LstmAttention, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.use_attention = use_attention
        self.classifier = torch.nn.Sequential(
        #nn.Linear(hidden_size, 64),
        #torch.nn.ReLU(),
        #torch.nn.Linear(64, D_out),
        torch.nn.Linear(hidden_size, D_out)
        )

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input, seq_lens=None):
        if seq_lens is not None:
            input = pack_padded_sequence(input_sentence, seq_lengths, enforce_sorted = False, batch_first=True)
        #h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
        #c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
        output, (final_hidden_state, final_cell_state) = self.lstm(input) #, (h_0, c_0))
        #print (final_hidden_state)
        if self.use_attention:
            attn_output = self.attention_net(output, final_hidden_state)
            final_output = self.classifier(attn_output)
        else:
            final_output = self.classifier(final_hidden_state[-1])
        return final_output

class Attention(nn.Module):
    def __init__(self, D_in, H, D_out, hidden_size, bin_label, use_lstm=False):
        super(Attention, self).__init__()
        layers = [
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    ]
        if bin_label:
            layers.append(torch.nn.Sigmoid())
        else:
            layers.append(torch.nn.Softmax(dim=1))
        self.use_lstm = use_lstm
        self.classifier = torch.nn.Sequential(*layers)
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(1, hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.bias.data.normal_(mean, std)

    def forward(self, numerical, embedding):
        #print (embedding.shape)
        output = matrix_mul(embedding.permute(1, 0, 2), self.weight, self.bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, 1)
        output = element_wise_mul(embedding.permute(1, 0, 2), output.permute(1, 0)).squeeze(0)
        output = self.classifier(output)
        return output

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

    def forward(self, input_ids):
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        #print (input_ids.shape)
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        x_embed = self.embedding(input_ids).float()
        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        return x_fc

class SeqAttention(nn.Module):
    def __init__(self, D_in, H, D_out, hidden_size, binary_classification, use_attention=True):
        super(SeqAttention, self).__init__()
        input_shape = D_in if use_attention else hidden_size
        media_shape = hidden_size if use_attention else int(hidden_size/2)
        layers = [
        torch.nn.Linear(input_shape, media_shape),
        torch.nn.ReLU(),
        torch.nn.Linear(media_shape, D_out),
    ]
        if binary_classification:
            layers.append(torch.nn.Sigmoid())
        #else:
         #   layers.append(torch.nn.Softmax(dim=1))
        self.lstm = nn.LSTM(D_in, hidden_size, bidirectional=True)
        self.gru = nn.GRU(D_in, hidden_size, bidirectional=True)
        self.classifier = torch.nn.Sequential(*layers)
        self.weight = nn.Parameter(torch.Tensor(2*hidden_size, 2*hidden_size))
        self.bias = nn.Parameter(torch.Tensor(1, 2*hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2*hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)
        self.use_attention = use_attention

    def _create_weights(self, mean=0.0, std=0.05):
        self.weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.bias.data.normal_(mean, std)

    def forward(self, numerical, embedding, seq_lens=None):
        #print (embedding.shape)
        #print (seq_lens)
        #sorted_idx = np.argsort(seq_lens)[::-1]
        #seq_lens = torch.tensor(seq_lens)
        #to_cuda(seq_lens)
        #lengths_sorted, sorted_idx = seq_lens.sort(descending=True)
        #embedding, seq_lens = embedding[sorted_idx], seq_lens[sorted_idx].tolist()
        #print (seq_lens, type(seq_lens))
        #print (sorted_idx)
        #embeddings = embedding[sorted_idx.copy()]
        input = pack_padded_sequence(embedding, seq_lens, enforce_sorted=False) if seq_lens!=None else embedding
        #packed_output, (final_hidden_state, final_cell_state) = self.lstm(input)
        packed_output, dummy = self.gru(input)
        if self.use_attention:
            output, input_sizes = pad_packed_sequence(packed_output)
            output = matrix_mul(output, self.weight, self.bias)
            output = matrix_mul(output, self.context_weight).permute(1, 0)
            output = F.softmax(output, 1)
            output = element_wise_mul(embedding, output.permute(1, 0)).squeeze(0)
            output = self.classifier(output)
        else:
            #print (final_hidden_state[-1].shape)
            output = self.classifier(final_hidden_state[-1])
        return output

class EnsembleModelsAttention(nn.Module):
    def __init__(self, D_in, H, D_out, hidden_size, models, bin_label):
        super(EnsembleModelsAttention, self).__init__()
        layers = [
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    ]
        if bin_label:
            layers.append(torch.nn.Sigmoid())
        self.classifier = torch.nn.Sequential(*layers)
        self.seq_attention = SeqAttention(D_in, H, D_out, hidden_size, bin_label, True)
        self.models = models

    def forward(self, fix_seq_len, numerical, pretrained_data, pretrained_seq_len, data):
        outputs = []
        for i in range(len(self.models)):
            model = self.models[i]
            input, seq_len = data[i]
            if seq_len is not None:
                sub_out = model(input, seq_len)
            else:
                sub_out = model(input)
            #print (sub_out.shape)
            sub_out = sub_out.view(fix_seq_len, -1, sub_out.shape[-1])
            outputs.append(sub_out)
        if len(outputs) > 1:
            out = torch.cat([sub_out.reshape(sub_out.shape[0], 1, -1) for sub_out in outputs], 1)
        else:
            out = outputs[0]
        #print (out.shape, pretrained_data.shape)
        out = torch.cat((out, pretrained_data), dim=-1)
        #print (out.shape)
        out = self.seq_attention(numerical, out, pretrained_seq_len)
        """
        if x is not None:
            out = torch.cat((out, x), dim=1)
        out = self.attention(out)
        """
        return out

def matrix_mul(input, weight, bias=False):
    #input = input.permute(1, 0, 2)
    feature_list = []
    for feature in input:
        feature_weight = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature_weight = feature_weight + bias.expand(feature_weight.size()[0], bias.size()[1])
        feature_weight = torch.tanh(feature_weight).unsqueeze(0)
        feature_list.append(feature_weight)
    output = torch.cat(feature_list, 0)
    return torch.squeeze(output, len(output.shape)-1)

def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def under_sample(X, y):
    X, y = rus.fit_resample(X, y)
    return X, y

def over_sample(X, y):
    X, y = ros.fit_resample(X, y)
    return X, y

def to_cuda(d):
    d.to(device)

def to_self_cuda(d):
    return torch.tensor(d).to(device)

def to_float_cuda(d):
    return torch.tensor(d).float().to(device)

def count_label(labels):
    import collections
    dic = collections.defaultdict(int)
    for v in labels:
        dic[v] += 1
    print (dic)

def divide_data(X_data, bioLen, numLen, tweetLen):
    bio_val = X_data[:,:bioLen]
    embedding_bio = to_float_cuda(X_data[:,:bioLen])
    embedding_tweet = to_float_cuda(X_data[:, bioLen+numLen:bioLen+numLen+tweetLen])
    #embedding = torch.stack((embedding_bio, embedding_tweet), axis=1)
    numerical = to_float_cuda(MaxAbsScaler().fit_transform(X_data[:,bioLen:bioLen+17]))
    embedding_network = to_float_cuda(X_data[:, bioLen+numLen+tweetLen:])
    #print (embedding_bio.shape, embedding_tweet.shape, embedding_network.shape)
    return numerical, embedding_bio, embedding_tweet, embedding_network

def map_handle_gt(filename):
    races = {}
    genders = {}
    ages = {}
    handle2year = get_handle2year("handle2year.csv")
    f = open(filename)
    whites = [race.lower() for race in "white,irish,Jewish,English,American,Armenians,Armenian,Albanians,Serbs,Greeks,Italian,Swedish-speaking,British,Scottish,Swedish,Poles,Pashtuns,Ukrainians,Kurds,German,Hungarians,Germans,Scotch-irish,Georgians,Russians,Swedes,Bulgarians,Italians,Ashkenazi,Ukrainian,Iranian,Austrians,Welsh,Czechs,Canadian,Albanian,Norwegians,Danes,Slovak,Polish,Transylvanian,Danish,Persian,Romanians,Icelanders,Australians,Croatian,Jews,Spaniards,French,Ossetians,Macedonians,Belarusians,Serbs|scottish,Romanian,Puerto,Australian,Cajun,Stateside,Arab,Swiss,Portuguese,Yazidis,Croats".split(",")]
    asians = [race.lower() for race in "Bengali,Japanese,Koreans,Indian,Korean,Punjabi,Tamil,han,Thai,Chinese,Malaysian,Indians,Dravida,Hoklo,Monguor,Filipino,Telugu,shan,Tibetan,Turkish,Malayali,Koreans|korean,Taiwanese,Pakistanis|pakistani,Sinhala,Vietnamese,Assamese,Pakistanis,Asian,Nepalis".split(",")]
    black = [race.lower() for race in "African, Afro-Americans,Yoruba,Fula,Black,Igbo,Baganda,Haitian,Somalis,Kiga,Zulu,Malians,Tutsi,Nyakyusa,Nigerian,Oromo,sotho".split(",")]
    for line in f:
        info = line.strip().split("\x1b")
        handle, gender, age, race = info[1].lower(), info[2].lower(), info[3].lower(), info[-1].lower()
        if gender:
            genders[handle] = gender
        if age:
            ages[handle] = 2020-int(age.split("-")[0])
            if handle in handle2year:
                ages[handle] = handle2year[handle]-int(age.split("-")[0])
        if not race:
            continue
        race = race.split(" ")[0]
        if race in whites:
            races[handle] = "white"
        elif race in black:
            races[handle] = "black"
        elif race in asians:
            races[handle] = "asian"
        else:
            races[handle] = "other"
    f.close()
    handle2year = get_handle2year("handle2year.csv")
    f = open("query_attributes.csv")
    for line in f:
        info = line.strip().split("\x1b")
        handle, gender, age, race = info[1].lower(), info[2].lower(), info[3].lower(), info[-1].lower()
        if age and handle not in ages:
            ages[handle] = 2020-int(age.split("-")[0])
            if handle in handle2year:
                ages[handle] = handle2year[handle]-int(age.split("-")[0])
        if gender and handle not in genders:
            genders[handle] = gender
    f.close()
    return races, genders, ages

def get_remain_handles(path):
    f = open(path)
    f.readline()
    handles = set()
    for line in f:
        line = line.strip()
        handle = line.split("\x1b")[0]
        handles.add(handle)
    return handles

def get_handle2year(filename):
    handle2year = {}
    f = open(filename)
    for line in f:
        handle, year = line.strip().split(",")
        handle2year[handle] = int(year)
    f.close()
    return handle2year

def process_emojis(emojis, longest_length=10):
    emojis_list = []
    if not emojis:
        return ["<pad>" for i in range(longest_length)]
    sep_emoji_texts = emojis.split(" ")
    for tweet_emoji in sep_emoji_texts:
        emojis_words = tweet_emoji.lower().replace(":", "_").split("_")
        for word in emojis_words:
            if len(emojis_list) == longest_length:
                break
            if word:
                emojis_list.append(word)
    for i in range(longest_length-len(emojis_list)):
        emojis_list.append("<pad>")
    return emojis_list

def process_hashtags(hashtags):
    return hashtags.split(" ")

def get_files_under_dir(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]
