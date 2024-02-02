import requests as r

def get_unique_chars(words):
    chars = sorted(list(set(''.join(words))))
    return chars

def write_raw_content(dataset_fname, wordsurl='https://shorturl.at/rtxyZ', truncate=False):
    res = r.get(wordsurl)
    with open(dataset_fname, 'wb') as f:
        f.write(res.content)

def init_dataset(dataset_fname):
    words = open(dataset_fname, 'r').read().splitlines()
    for i in range(len(words)):
        words[i] = words[i] + '.'
    chars = get_unique_chars(words)
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}

    return words, stoi, itos