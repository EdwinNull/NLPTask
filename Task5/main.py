from tqdm import tqdm
import torch
import pickle
import math

import torch.nn.functional as F
from torch import optim, device

from nn import Language
from feature import get_batch, Word_Embedding

with open("C:/Users/wz367/Downloads/poetryFromTang.txt", 'rb') as f:
    temp = f.readlines()

a = Word_Embedding(temp)
a.data_process()
batch = int(input("Setting batch size:"))
train = get_batch(a.matrix, batch)
learning_rate = 0.001
iter_times = int(input("Setting iter_times:"))

def calculate_perplexity(model, data_loader, criterion, device='cuda'):
    model.eval()
    total_losses = 0.0
    total_words = 0

    with torch.no_grad():
        for batches in data_loader:
            inputs = batches[:, :-1].to(device)
            targets = batches[:, 1:].to(device)
            outputs = model(inputs)
            losses = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_losses += losses.item() * targets.size(0)
            total_words += targets.size(0)

    avg_loss = total_losses / total_words
    perplexity = math.exp(avg_loss)
    return perplexity


strategies = ['lstm', 'gru']
train_loss_records = list()
models = list()
for i in range(2):
    model = Language(50, len(a.word_dict), 50, a.tag_dict, a.word_dict, strategy=strategies[i])
    print(len(model.word_to_num))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = F.cross_entropy
    train_loss_record = list()

    model = model.cuda()
    for iteration in range(iter_times):
        total_loss = 0
        model.train()
        progress_bar = tqdm(enumerate(train), total=len(train), desc=f"Iteration {iteration + 1}")
        for i, batch in progress_bar:
            x = batch.cuda()
            x, y = x[:, :-1], x[:, 1:]
            pred = model(x).transpose(1, 2)
            optimizer.zero_grad()
            loss = loss_fun(pred, y)
            total_loss += loss.item() / (x.shape[1] - 1)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=total_loss / (i + 1))
        train_loss_record.append(total_loss / len(train))
        # print("Iteration", iteration + 1)
        # print("Train loss:", total_loss / len(train))
    train_loss_records.append(train_loss_record)
    models.append(model)
    perp = calculate_perplexity(model, train, F.cross_entropy)
    print("Perplexity:", perp)

model = models[0]
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)



