from collections import *    
from random import random
import scrape_lyrics
import random

def train_char_lm(data, order=4):
    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data
    for i in xrange(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        lm[history][char]+=1
    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.iteritems()]
    outlm = {hist:normalize(chars) for hist, chars in lm.iteritems()}
    return outlm

def generate_text(lm, order, nletters=1000):
    history = random.choice(lm.items())[0]
    out = []
    for i in xrange(nletters):
        c = generate_letter(lm, history, order)
        history = history[-order:] + c
        out.append(c)
    return "".join(out)

def generate_letter(lm, history, order):
    history = history[-order:]
    dist = lm[history]
    x = random.random()
    for c,v in dist:
        x = x - v
        if x <= 0: return c
            
artist = raw_input('Enter Artist: ')
order = int(raw_input('Enter character-level n-gram order: '))
(lyrics, lines, bow, line_endings) = scrape_lyrics.get_lyrics(artist)
lm = train_char_lm(lyrics, order)
print generate_text(lm, order)
