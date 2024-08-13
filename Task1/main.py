import numpy as np
import pandas as pd
import random
from tqdm import tqdm


train_set = 'F:/sentiment-analysis-on-movie-reviews/train.tsv/train.tsv'
test_set = 'F:/sentiment-analysis-on-movie-reviews/test.tsv/test.tsv'
df = pd.read_csv(train_set,sep='\t')
df_test = pd.read_csv(test_set,sep='\t')
# df.head(100)


# 词袋特征提取
def get_bow_token(data):
    feature_set = set()
    for row in data:
        token_list = row.split()
        for token in token_list:
            feature_set.add(token)
    return feature_set
# 获取训练集中的BOW特征，并返回一个特征矩阵


def get_bow_feature(data,feature_set):
    feature_size = len(feature_set)
    feature_list = np.zeros((data.shape[0], feature_size),dtype=np.int8)
    feature_map = dict(zip(feature_set,range(feature_size)))
    for index in range(data.shape[0]):
        token_list = str(data[index]).split()
        for token in token_list:
            if token in feature_set:
                feature_index = feature_map[token]
                feature_list[index, feature_index] += 1
    return feature_list


# 同上，对ngram进行相似的操作，获取ngram特征（考虑性能等问题，先采用2gram）
def get_ngram_token(data,ngram=2):
    feature_set=set()
    for row in data:
        token_list=row.split()
        for token in token_list:
            for i in range(len(token)-ngram):
                feature_set.add(token[i:i+ngram])
    return feature_set


def get_ngram_feature(data,feature_set,ngram=2):
    feature_size=len(feature_set)
    feature_list=np.zeros((data.shape[0],feature_size),dtype=np.int16)
    feature_map=dict(zip(feature_set,range(feature_size)))
    for index in range(data.shape[0]):
        token_list=str(data[index]).split()
        for token in token_list:
            for i in range(len(token)-ngram):
                gram=token[i:i+ngram]
                if gram in feature_set:
                    feature_index=feature_map[gram]
                    feature_list[index,feature_index]+=1
    return feature_list


# 根据参数的值来提取特征
def create_feature(ngram=2,analyzer='word'):
    if analyzer=='word':
        feature_set=get_bow_token(data=df['Phrase'].to_numpy())
        x=get_bow_feature(data=df['Phrase'].to_numpy(),feature_set=feature_set)
        test_x = get_bow_feature(data=df_test['Phrase'].to_numpy(),feature_set=feature_set)
    if analyzer=='char':
        feature_set = get_ngram_token(data=df['Phrase'].to_numpy(),ngram=ngram)
        x=get_ngram_feature(data=df['Phrase'].to_numpy(),feature_set=feature_set,ngram=ngram)
        test_x = get_ngram_feature(data=df_test['Phrase'].to_numpy(),feature_set=feature_set,ngram=ngram)
    return feature_set,x,test_x


# 分割数据集
def train_test_split(x,y,test_rate):
    x_size = x.shape[0]
    train_size = int(x_size*(1-test_rate))
    index = [i for i in range(x_size)]
    # 随机分割,获得训练集和验证集
    random.shuffle(index)
    train_x = x[index][0:train_size]
    train_y = y[index][0:train_size]
    val_x = x[index][train_size+1:-1]
    val_y = y[index][train_size+1:-1]
    return train_x,val_x,train_y,val_y


# 矩阵乘法加速
def train_faster(train_x, train_y, val_x, val_y, batchsize=32, lr=1e0, epoch_number=100):
    iter_number = train_x.shape[0] // batchsize
    iter_remain = train_x.shape[0] % batchsize
    weight = np.zeros((train_x.shape[1], class_number))
    # 不同初始值的影响
    # weight=np.random.normal(0,1,[train_X.shape[1],class_number])
    train_loss_list = []
    test_loss_list = []
    for i in range(epoch_number):
        train_loss = 0
        test_loss = 0
        for j in tqdm(range(iter_number)):
            train_data = train_x[j * batchsize:j * batchsize + batchsize]
            y_train = train_y[j * batchsize:j * batchsize + batchsize]
            y = np.exp(train_data.dot(weight))
            y_hat = np.divide(y.T, np.sum(y, axis=1)).T
            train_loss += (-1 / train_x.shape[0]) * np.sum(np.multiply(y_train, np.log10(y_hat)))
            # 每个batch权重更新一次
            weight += (lr / batchsize) * train_data.T.dot(y_train - y_hat)

        y = np.exp(val_x.dot(weight))
        y_hat = np.divide(y.T, np.sum(y, axis=1)).T
        test_loss = (-1 / val_x.shape[0]) * np.sum(np.multiply(val_y, np.log10(y_hat)))
        # print('train_loss:',train_loss," test_loss:",test_loss)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    return train_loss_list, test_loss_list, weight


class_number = 5
y = df['Sentiment'].to_numpy()
y_onehot = np.zeros((y.shape[0], class_number), dtype=np.int8)
for i in range(y.shape[0]):
    y_onehot[i, y[i]] += 1
parameter_list = [
    {'ngram': 2, 'analyzer': 'word', 'batchsize': 32, "lr": 1e0, 'epoch_number': 20, 'test_rate': 0.2},
    {'ngram': 2, 'analyzer': 'char', 'batchsize': 32, "lr": 1e0, 'epoch_number': 20, 'test_rate': 0.2},
    {'ngram': 3, 'analyzer': 'char', 'batchsize': 32, "lr": 1e0, 'epoch_number': 20, 'test_rate': 0.2},
    {'ngram': 2, 'analyzer': 'word', 'batchsize': 8, "lr": 1e0, 'epoch_number': 20, 'test_rate': 0.2},
    {'ngram': 2, 'analyzer': 'word', 'batchsize': 128, "lr": 1e0, 'epoch_number': 20, 'test_rate': 0.2},
    {'ngram': 2, 'analyzer': 'word', 'batchsize': 32, "lr": 1e1, 'epoch_number': 20, 'test_rate': 0.2},
    {'ngram': 2, 'analyzer': 'word', 'batchsize': 8, "lr": 1e0, 'epoch_number': 10, 'test_rate': 0.2},

]


def flow(parameter_list):
    for parameter_dict in parameter_list:
        ngram = parameter_dict['ngram']
        analyzer = parameter_dict['analyzer']
        batchsize = parameter_dict['batchsize']
        lr = parameter_dict['lr']
        epoch_number = parameter_dict['epoch_number']
        test_rate = parameter_dict['test_rate']

        feature_set, x, test_x = create_feature(ngram=ngram, analyzer=analyzer)

        train_x, val_x, train_y, val_y = train_test_split(x, y_onehot, test_rate=test_rate)
        train_loss_list, test_loss_list, weight = train_faster(train_x=train_x, train_y=train_y, val_x=val_x,
                                                               val_y=val_y, batchsize=batchsize, lr=lr,
                                                               epoch_number=epoch_number)

        y_temp = np.exp(val_x.dot(weight))
        y_temp = np.divide(y_temp.T, np.sum(y_temp, axis=1)).T
        y_predict = np.array([np.argmax(i) for i in y_temp])
        y_val = np.array([np.argmax(i) for i in val_y])

        acc = np.sum(y_predict.astype('int') == y_val.astype('int')) / y_val.shape[0]
        parameter_dict['acc'] = '%.4f' % acc
        print('acc:', acc)

        parameter_dict['best_train_loss'] = '%.4f' % np.min(train_loss_list)
        parameter_dict['best_test_loss'] = '%.4f' % np.min(test_loss_list)

        print('train_loss:', parameter_dict['best_train_loss'], " test_loss:", parameter_dict['best_test_loss'])

    return parameter_list


parameter_list = flow(parameter_list)