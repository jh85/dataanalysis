import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.autograd as autograd
from torchvision import datasets, transforms
import json
import collections
import time

def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)

class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                           replace=True, p=self.word_p)

        return negative_sample

class CBOWDataset(data.Dataset):
    def __init__(self, window_size, corpus):
        self.corpus = corpus
        self.contexts, self.targets = create_contexts_target(self.corpus, window_size=window_size)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        context =  self.contexts[index]
        target = self.targets[index]
        return torch.tensor(context), torch.tensor(target), self.corpus

def od_collate_fn(batch):
    contexts = []
    targets = []

    sample_size = 10
    power = 0.75
    corpus_use = batch[0][2]
    sampler = UnigramSampler(corpus_use, power, sample_size)

    for sample in batch:
        contexts.append(sample[0])
        targets.append(sample[1])
    negative_sample = torch.tensor(sampler.get_negative_sample(np.array(targets)))
    contexts = torch.stack(contexts, dim=0)
    targets = torch.tensor(targets)


    x = torch.cat([targets.view(-1, 1),negative_sample],dim=1)
    y = []
    for dim0 in range(x.size()[0]):
        y.append(torch.where(x[dim0] == x[dim0][0], torch.ones_like(x[dim0]), torch.zeros_like(x[dim0])).view(1, -1))
    y = torch.cat(y,dim=0)
    #y = y/(y.sum(dim=1).view(-1,1))
    return contexts, x, y

def preprocess(text):
    text = text.lower()
    text = text.replace(".", " [END] ")
    text = text.replace("\"", " [DQ] ")
    text = text.replace("”", " [DQ2] ")
    text = text.replace(" '", " [SQ] ")
    text = text.replace("' ", " [SQ] ")
#    text = text.replace("’", " [SQ2] ")
    text = text.replace(",", " [C] ")
    text = text.replace(":", " [DC] ")
    text = text.replace(";", " [SC] ")
    text = text.replace("?", " [Q] ")
    text = text.replace("!", " [X] ")
    text = text.replace("*", " [W] ")
    text = text.replace("(", " [P1] ")
    text = text.replace(")", " [P2] ")
#    text = text.replace("\n", " [R] ")
    text = text.replace("\n", "")
    text = text.replace("👌", " 👌 ")
    text = text.replace("😦", " 😦 ")
    text = text.replace("😉", " 😉 ")
    text = text.replace("🙂", " 🙂 ")
    text = text.replace("🙃", " 🙃 ")
    text = text.replace("😅", " 😅 ")
    text = text.replace("🤦", " 🤦 ")
    text = text.replace("😊", " 😊 ")
    text = text.replace("💰", " 💰 ")
    text = text.replace("💴", " 💴 ")
    text = text.replace("🥳", " 🥳 ")
    text = text.replace("😴", " 😴 ")
    text = text.replace("👍🏻", " 👍🏻 ")
    text = text.replace("🧙", " 🧙 ")
    text = text.replace("👍", " 👍 ")
    text = text.replace("😂", " 😂 ")
    text = text.replace("😜", " 😜 ")
    text = text.replace("🤗", " 🤗 ")
    text = text.replace("😞", " 😞 ")
    words = text.split()

    word_to_id = dict()
    id_to_word = dict()
    for w in words:
        if w not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[w] = new_id
            id_to_word[new_id] = w
    corpus = np.array([word_to_id[w] for w in words])
    return corpus,word_to_id,id_to_word

def load_data(filename):
    '''
    pass texts to preprocess()
    remove <doc></doc> tags for enwiki
    extract texts from JSON
    '''
    if True:
        # enwiki-latest-pages-articles.xml.bz2
        with open(filename, "r") as f:
            txt = ""
            for i,line in enumerate(f):
                if "</doc>" in line:
                    continue
                if "<doc" in line:
                    continue
                txt += line
                if i > 1000000:
                    break
            return preprocess(txt)

    if False:
        # Jsonl file
        data = []
        with open(filename) as f:
            for line in f:
                data.append(json.loads(line))
        messages = ""
        count = 0
        for d in data:
            for m in d["messages"]:
                messages += m
            count += 1
        return preprocess(messages)

class CBOW(nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.embed1 = nn.Embedding(V,H)
        self.embed2 = nn.Embedding(V,H)

    def forward(self, contexts, centers):
        '''
        contexts: [0,1,2,3,4,    6,7,8,9,10]
        centers:  [5,21,22,29,39, 3, 2,11,19, 4, 5]
        t:        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]   1 for the correct centers
        '''
        self.embed1.forward(contexts)

        h = self.embed1(contexts).sum(1) / contexts.shape[1]
        y = (self.embed2(centers).permute(1,0,2) * h).sum(2).permute(1,0)
        y = torch.sigmoid(y)
        return y

def criterion(y,t,eps=1e-7):
    return (-t*torch.log(y+eps) - (1-t)*(torch.log(1-y+eps))).sum(axis=1).mean()
    
def main():
    filename = "enwiki.txt"
    corpus, w2id, id2w = load_data(filename)
    window_size = 5
    dataset_train = CBOWDataset(window_size, corpus)
    batch_size = 200
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn = od_collate_fn
    )
    lr = 0.001
    device = "cuda"
    model = CBOW(len(id2w), 300).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    CHECKPOINT_FILE = "checkpoint_cbow.pth"
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])                            
    
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        loss_list = []
        t1 = time.time()
        for context,centers,t in dataloader_train:
            context = context.to(device)
            centers = centers.to(device)
            t = t.to(device)

            y = model(context,centers)
            loss = criterion(y,t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
        t2 = time.time()
        print(f"epoch={epoch} loss={np.mean(loss_list)} time={t2-t1}")

        checkpoint_config = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint_config, CHECKPOINT_FILE)

main()
