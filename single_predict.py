import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import utils
import model.net as net
from model.data_loader import DataLoader
from evaluate import evaluate

model = torch.load(Path("model/modelsave.pth"))
model.eval()

data_dir = "data/small"
# loading vocab (we require this to map words to their indices)
vocab_path = os.path.join(data_dir, 'words.txt')
vocab = {}
with open(vocab_path) as f:
    for i, l in enumerate(f.read().splitlines()):
        vocab[l] = i

# loading tags (we require this to map tags to their indices)
tags_path = os.path.join(data_dir, 'tags.txt')
tag_map = {}
with open(tags_path) as f:
    for i, t in enumerate(f.read().splitlines()):
        tag_map[t] = i

x = input()
length = len(x)
batch_one = np.ones(length)

for i in range(length):
    batch_one[i] = vocab[x[i]]

batch_one = batch_one.reshape((1, length))
batch_one = torch.LongTensor(batch_one)
batch_one = Variable(batch_one)
#batch_one.unsqueeze(0)


y_pred = model(batch_one)
print(length)
print(y_pred)
