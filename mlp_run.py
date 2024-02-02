import os
import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import random 
import pickle


def get_unique_chars(words):
    chars = sorted(list(set(''.join(words))))
    return chars

def main():
    words = open('names.txt', 'r').read().splitlines()
    for i in range(len(words)):
        words[i] = words[i] + '.'

    chars = get_unique_chars(words)
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}


    if os.path.isfile('mlp_weights.pkl'):
        weights = pickle.load(open('mlp_weights.pkl', 'rb'))    

    C, W1, b1, W2, b2 = weights
    block_size = 3
    out = []
    for _ in range(10):
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context ])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
    print(''.join(itos[i] for i in out))


if __name__ == '__main__': 
    main()