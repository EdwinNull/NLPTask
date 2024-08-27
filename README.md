# NLPTask
本项目是NLP-Beginner的任务五实现,实现了基于LSTM/GRU的语言模型，做到了输出随机诗歌或指定的藏头诗。

## 运行环境
- Python 3.12.4
- PyTorch 2.3.1

## 项目结构
feature.py 进行数据集的预处理

nn.py   设计了LSTM/GRU模型

generator.py 调用已有模型，生成随机诗歌或藏头诗

main.py 训练模型，并将其保存为pkl文件

poetryFromTang.txt 数据集，包含唐诗约3000句

model_gru.pkl     GRU模型的pkl文件

model_lstm.pkl    LSTM模型的pkl文件

## 运行方法
在完整的项目目录下，运行main.py文件即可训练模型并保存为pkl文件。

运行generator.py文件，输入模型的pkl文件路径，指定生成诗歌的长度和藏头诗的开头，即可生成随机诗歌或藏头诗。