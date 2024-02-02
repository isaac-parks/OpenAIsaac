import os
import torch
import torch.nn.functional as F 
import pickle
from mlp_train import main as mlp_train

def main(words, itos, w_fname, itrs=20):
    for i in range(len(words)):
        words[i] = words[i] + '.'
    
    with open(w_fname, 'rb'):
        weights = pickle.load(open(w_fname, 'rb'))    

    C, W1, b1, W2, b2 = weights
    block_size = 3
    out = []
    for _ in range(itrs):
        context = [0] * block_size
        e_brake = 0
        while True:
            emb = C[torch.tensor([context ])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            e_brake += 1
            if ix == 0 or e_brake > 100:
                break

    return ''.join(itos[i] for i in out)


if __name__ == '__main__': 
    print("run the script 'run.py' instead.")