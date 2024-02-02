#!/usr/bin/env python
import os
import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import random 
import pickle
random.seed(42)

def gen_txt(url=''):
    words = open('names.txt', 'r').read().splitlines()
    for i in range(len(words)):
        words[i] = words[i] + '.'
    chars = get_unique_chars(words)
    stoi = {s: i for i, s in enumerate(chars)}
    print(stoi)
    # itos = {i: s for s, i in stoi.items()}

    return words, stoi

def get_unique_chars(words):
    chars = sorted(list(set(''.join(words))))
    return chars

def build_dataset(words, stoi):
    block_size = 3
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


def make_params(vocab_sz=27):
    g = torch.Generator().manual_seed(214783647)
    C = torch.randn((vocab_sz,10), generator=g) # initial embeddings
    W1 = torch.randn((30,200), generator=g)
    b1 = torch.randn(200, generator=g)
    W2 = torch.randn((200, vocab_sz), generator=g)
    b2 = torch.randn(vocab_sz, generator=g)
    paramaters = [C, W1, b1, W2, b2]

    for p in paramaters:
        p.requires_grad = True

    return paramaters

def train(Xtr, Ytr, params, itr=100000, lr=0.1, declr=0.01):
    C, W1, b1, W2, b2 = params
    for i in range(itr):
        # Minibatch creation
        ix = torch.randint(0, Xtr.shape[0], (32,)) # Creating batch sets to make training faster as to going backwards and forwards on every single word every loop
        # Forward pass
        emb = C[Xtr[ix]]

        h = torch.tanh(emb.view(emb.shape[0] , 30) @ W1 + b1)
        logits = h @ W2 + b2

        loss = F.cross_entropy(logits, Ytr[ix]) 
        # backward pass
        for p in params:
            p.grad = None
        loss.backward()
        # update 
        for p in params:
            _lr = lr if i < itr/2 else declr
            p.data += -_lr * p.grad
    return params

def validate_tr(weights, Xdev, Ydev):
    C, W1, b1, W2, b2 = weights
    emb = C[Xdev]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev)
    print(loss)

def main():
    words, stoi = gen_txt()
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1], stoi)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi)
    # Xte, Yte = build_dataset(words[n2:], stoi)
    params = make_params(len(get_unique_chars(words)))
    if os.path.isfile('mlp_weights.pkl'):
        weights = pickle.load(open('mlp_weights.pkl', 'rb'))    
    else:
        weights = train(Xtr, Ytr, params)

    validate_tr(weights, Xdev, Ydev)

    with open('mlp_weights.pkl', 'wb') as f:
        pickle.dump(weights, f)

if __name__ == '__main__':
    main()