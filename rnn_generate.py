import nltk
import scrape_lyrics
import itertools
import numpy as np
from utils import *
import operator
from timeit import Timer
from datetime import datetime
import sys
import random
import get_shakespeare

class RNN:
     
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        
    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
	    # Perform forward propagation and return index of the highest score
	    o, s = self.forward_propagation(x)
	    return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N
        
    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])              
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]
        
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = model.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = model.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = model.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)
            
    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

sentence_start_token = "LINESTART"
sentence_end_token = "LINEEND"
entries = nltk.corpus.cmudict.entries()

# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - Y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, Y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if ((epoch + 1) % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, Y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen+100, epoch+1, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5 
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        for i in range(len(Y_train)):
            # One SGD step
            model.numpy_sdg_step(X_train[i], Y_train[i], learning_rate)
            num_examples_seen += 1
            
def generate_line(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    # while not new_sentence[-1] == word_to_index[sentence_end_token]:
    line_length = np.random.normal(average_line_length, line_length_variance)
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        random_word = random.choice(word_to_index.values())
        sampled_word = random_word
        # We don't want to sample unknown words
        while sampled_word == random_word:
            samples = np.random.multinomial(1, next_word_probs[0][-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str
    
def get_statistics(lines, bow):
    token_count = int(sum(bow.values()))
    unique_token_count = len(bow)
    s = 0
    line_lengths = []
    for line in lines:
	    l = line.split();
	    s += len(l)
	    line_lengths.append(len(l))
    average_line_length = s / len(lines)
    line_length_variance = np.std(line_lengths)
    average_song_length = (int(len(lines)/27))
    return (token_count, unique_token_count, average_line_length, line_length_variance, average_song_length)
    
def rhyme(inp, level):
    syllables = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
    return set(rhymes)
			
def doesRhyme(word1, word2):
    if word1.find(word2) == len(word1) - len(word2):
        return False
    if word2.find(word1) == len(word2) - len(word1): 
        return False
    return word1 in rhyme(word2, 1)
    
def calc_goodness(candidate, lyrics):
    goodness = float(0)
    if lyrics and candidate:
        if (doesRhyme(candidate[-1], lyrics[-1][-1])):
            goodness += 0.5
        if len(lyrics) > 1:
            if (doesRhyme(candidate[-1], lyrics[-2][-1])):
                goodness += 0.25
        for word in candidate:
            index = word_to_index[word]
            goodness += word_freq[word] / len(vocab)
    return goodness

# get artist parameter from user
artist = raw_input('Enter Artist: ')

# use lyric-scraping code to grab lyrics from genius.com
(lyrics, lines, bow, line_endings) = scrape_lyrics.get_lyrics(artist)

# use shakespeare sonnets as lyrics
#(lyrics, lines, bow, line_endings) = get_shakespeare.get_shake()

print "discovered %d lines of lyrics" % len(lines)

lines = ["%s %s %s" % (sentence_start_token, line, sentence_end_token) for line in lines]

# tokenizing each lyric into individual words
tokenized_lines = [nltk.word_tokenize(line) for line in lines]

# find word frequency distribution
word_freq = nltk.FreqDist(itertools.chain(*tokenized_lines))
print "found %d unique tokens." % len(word_freq.items())
vocabulary_size = len(word_freq.items())

# get most common words
vocab = word_freq.most_common(vocabulary_size)

# build index to word and word to index vectors
index_to_word = [x[0] for x in vocab]
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in line[:-1]] for line in tokenized_lines])
Y_train = np.asarray([[word_to_index[w] for w in line[1:]] for line in tokenized_lines])

np.random.seed(10)
model = RNN(vocabulary_size)
o, s = model.forward_propagation(X_train[10])

predictions = model.predict(X_train[10])

# Limit to 1000 examples to save time
print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
print "Actual loss: %f" % model.calculate_loss(X_train[:1000], Y_train[:1000])

# to avoid an extreme amount of calculation, we perform gradient checking on a smaller vocabulary
grad_check_vocab_size = 100
np.random.seed(10)
model = RNN(grad_check_vocab_size, 10, bptt_truncate=1000)
model.gradient_check([0,1,2,3], [1,2,3,4])

np.random.seed(10)
model = RNN(vocabulary_size)
# here, we train the RNN model for 30 epochs, evaluating loss after each 5 epochs
losses = train_with_sgd(model, X_train[:100], Y_train[:100], nepoch=100, evaluate_loss_after=5)

(token_count, unique_token_count, average_line_length, line_length_variance, average_song_length) = get_statistics(lines, bow)

num_lines = 30

lyrics = []

print '\n'

for i in range(num_lines):
    candidates = []
    for k in range(10):
        candidate = []
        while len(candidate) < average_line_length - line_length_variance or len(candidate) > average_line_length + line_length_variance:
            candidate = generate_line(model)
        " ".join(candidate)
        candidates.append(candidate)
    line = []
    # pick best candidate from generated lines based on goodness score
    best_goodness = 0
    best_candidate = candidates[0]
    for candidate in candidates:
        goodness = calc_goodness(candidate, lyrics)
        if goodness > best_goodness:
            best_goodness = goodness
            best_candidate = candidate
    lyrics.append(" ".join(best_candidate))
    print lyrics[-1]

