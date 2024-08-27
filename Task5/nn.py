import torch.nn as nn
import torch


class Poem(nn.Module):
    def __init__(self, len_feature, len_words, len_hidden, num_to_word, word_to_num, strategy='lstm', pad_id=0,
                 start_id=1, end_id=2, drop_out=0.5):
        super(Poem, self).__init__()
        self.pad_id = pad_id
        self.start_id = start_id
        self.end_id = end_id
        self.num_to_word = num_to_word
        self.word_to_num = word_to_num
        self.len_feature = len_feature
        self.len_words = len_words
        self.len_hidden = len_hidden
        # 定义dropout层
        self.dropout = nn.Dropout(drop_out)
        # 使用Xavier初始化嵌入层
        _x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
        self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=_x)
        # 定义RNN层
        if strategy == 'lstm':
            self.gate = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, batch_first=True)
        elif strategy == 'gru':
            self.gate = nn.GRU(input_size=len_feature, hidden_size=len_hidden, batch_first=True)
        else:
            raise Exception("Unknown Strategy!")
        self.fc = nn.Linear(len_hidden, len_words)

    # 前向传播函数
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        self.gate.flatten_parameters()
        x, _ = self.gate(x)
        logits = self.fc(x)

        return logits

    #生成随机诗歌
    def generate_random_poem(self, max_len, num_sentence, random=False):
        if random:
            initialize = torch.randn
        else:
            initialize = torch.zeros
        hn = initialize((1, 1, self.len_hidden)).cuda()
        cn = initialize((1, 1, self.len_hidden)).cuda()
        x = torch.LongTensor([self.start_id]).cuda()
        poem = list()

        while len(poem) != num_sentence:
            word = x
            sentence = list()
            for j in range(max_len):
                word = torch.LongTensor([word]).cuda()
                word = self.embedding(word).view(1, 1, -1)
                output, (hn, cn) = self.gate(word, (hn, cn))
                output = self.fc(output)
                word = output.topk(1)[1][0].item()
                if word == self.end_id:
                    x = torch.LongTensor([self.start_id]).cuda()
                    break
                sentence.append(self.num_to_word[word])
                if self.word_to_num['。'] == word:
                    break
            else:
                x = self.word_to_num['。']
            if sentence:
                poem.append(sentence)
        return poem

    # 生成藏头诗

    def generate_hidden_head(self, heads, max_len=50, random=False):
        for head in heads:
            if head not in self.word_to_num:
                raise Exception("Word: " + head + " is not in the dictionary, please try another word")
        poem = list()
        if random:
            initialize = torch.randn
        else:
            initialize = torch.zeros
        for i in range(len(heads)):
            word = self.word_to_num[heads[i]]
            sentence = [heads[i]]
            hn = initialize((1, 1, self.len_hidden)).cuda()
            cn = initialize((1, 1, self.len_hidden)).cuda()
            for j in range(max_len - 1):
                word = torch.LongTensor([word]).cuda()
                word = self.embedding(word).view(1, 1, -1)
                output, (hn, cn) = self.gate(word, (hn, cn))
                output = self.fc(output)
                word = output.topk(1)[1][0].item()
                if j < max_len - 2 and word in [self.word_to_num['，'], self.word_to_num['。']]:
                    continue
                sentence.append(self.num_to_word[word])
            if sentence[-1] != self.num_to_word[self.word_to_num['。']]:
                sentence.append(self.num_to_word[self.word_to_num['。']])
            poem.append(sentence)
        return poem
