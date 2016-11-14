from __future__ import division
from collections import defaultdict
import random
import nltk
from nltk.collocations import *
import numpy as np
import sys
import scrape_lyrics
from tabulate import tabulate

class lyric_generator:
	model = dict()
	bow = defaultdict(float)
	ngram = 0

	def get_statistics(self, lines, bow):
		token_count = int(sum(bow.values()))
		unique_token_count = len(bow)
		s = 0
		line_lengths = []
		for line in lines:
			l = line.split();
			s += len(l)
			line_lengths.append(len(l))
		average_sentence_length = s / len(lines)
		sentence_length_variance = np.std(line_lengths)
		average_song_length = (int(len(lines)/27))
		return (token_count, unique_token_count, average_sentence_length, sentence_length_variance, average_song_length)

	def rhyme(self, inp, level):
		entries = nltk.corpus.cmudict.entries()
		syllables = [(word, syl) for word, syl in entries if word == inp]
		rhymes = []
		for (word, syllable) in syllables:
			rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
		return set(rhymes)
			
	def doesRhyme(self, word1, word2):
		if word1.find(word2) == len(word1) - len(word2):
			return False
		if word2.find(word1) == len(word2) - len(word1): 
			return False
		return word1 in self.rhyme(word2, 1)
		
	def generate(self, n, seed=None, max_iterations=150):
		model = self.model
		if seed is None:
			seed = random.choice(model.keys())
		output = list(seed)
		current = tuple(seed)

		for i in range(max_iterations):
			if current in model:
				possible_next_tokens = model[current]
				next_token = random.choice(possible_next_tokens)
				if next_token is None: break
				output.append(next_token)
				current = tuple(output[-n:])
			else:
				break
		return output
		
	def build_model(self, tokens, n):
		model = dict()
		if len(tokens) < n:
			return model
		for i in range(len(tokens) - n):
			gram = tuple(tokens[i:i+n])
			next_token = tokens[i+n]
			if gram in model:
				model[gram].append(next_token)
			else:
				model[gram] = [next_token]
		final_gram = tuple(tokens[len(tokens)-n:])
		if final_gram in model:
			model[final_gram].append(None)
		else:
			model[final_gram] = [None]
		return model

	def create_ngram_model(self, lyrics, lines, bow, line_endings, ngram):
		self.ngram = ngram
		tokens = nltk.word_tokenize(lyrics)
		self.bow = bow
		self.model = self.build_model(tokens, int(self.ngram))
			
	def calc_goodness(self, candidate, lyrics):
		goodness = float(1)
		if len(lyrics) > 0 and candidate:
			if (self.doesRhyme(candidate[-1], lyrics[-1][-1])):
				goodness += 0.5
			if len(lyrics) > 1:
				if (self.doesRhyme(candidate[-1], lyrics[-2][-1])):
					goodness += 0.25
			for word in candidate:
				goodness += bow[word] / len(bow)
			if candidate[-1] in line_endings:
				goodness += 0.25
		return goodness
	
	def generate_lyrics(self):
		lyrics = []
		for i in range(30):
			candidates = []
			for k in range(10):
				candidates.append(' '.join(self.generate(self.ngram, random.choice(self.model.keys()), int(np.random.normal(average_sentence_length, sentence_length_variance, 1)[0]))))
			best_score = 0
			best_candidate = ''
			for candidate in candidates:
				gscore = self.calc_goodness(candidate, lyrics)
				if (gscore > best_score):
					best_candidate = candidate
					best_score = gscore
			lyrics.append(best_candidate)
			print best_candidate

if __name__=='__main__':
	LG = lyric_generator()
	artist = raw_input('Enter Artist: ')
	ngram = int(raw_input('Enter n-gram order: '))
	(lyrics, lines, bow, line_endings) = scrape_lyrics.get_lyrics(artist)
	(token_count, unique_token_count, average_sentence_length, sentence_length_variance, average_song_length) = LG.get_statistics(lines, bow)
	table = [['Artist', artist],['Token Count', token_count],['Unique Token Count', unique_token_count],['Average Line Length', average_sentence_length],['Average Song Length', average_song_length]]
	LG.create_ngram_model(lyrics, lines, bow, line_endings, ngram)
	LG.generate_lyrics()
