import pickle


def cat_poem(l):
    """拼接诗句"""
    poem = list()
    for item in l:
        poem.append(''.join(item))
    return poem


string = ""
sentence_len = 7
sentence_num = 4
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
choice = int(input("请输入生成模式：(1表示随机诗歌，2表示藏头诗)"))
if choice == 1:
    sentence_len = int(input("请输入生成诗歌的句子长度："))
    sentence_num = int(input("请输入生成诗歌的句子数量："))
if choice == 2:
    string = input("请输入藏头诗的开头：")
    sentence_len = int(input("请输入藏头诗的句子长度："))
match choice:
    case 1:
        poem = cat_poem(model.generate_random_poem(sentence_len, sentence_num, random=True))
        for sentence in poem:
            print(sentence)
    case 2:
        poem = cat_poem(model.generate_hidden_head(string, 7, random=True))
        for sentence in poem:
            print(sentence)