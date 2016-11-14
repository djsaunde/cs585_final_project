from urlparse import urljoin
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import word_tokenize
import string
from matplotlib import pyplot
from collections import defaultdict

def get_lyrics(artist):
	BASE_URL = "http://genius.com"
	artist_url = "http://genius.com/artists/" + artist + "/"

	lyrics = [];

	response = requests.get(artist_url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36'})

	soup = BeautifulSoup(response.text, "lxml")
	for song_link in soup.select('ul.song_list > li > a'):
		link = urljoin(BASE_URL, song_link['href'])
		response = requests.get(link)
		soup = BeautifulSoup(response.text)
		lyric = soup.find('div', class_='lyrics').text.strip()
		lyrics.append(lyric)
		
	bow = defaultdict(int)
	line_endings = {}
	lines = []
	for lyric in lyrics:
		exclude = set(string.punctuation)
		lyric = ''.join(ch for ch in lyric if ch not in exclude)
		lyric = lyric.encode('ascii', 'ignore')
		lyric = lyric.lower()
		lines.extend(lyric.split('\n'))
	for line in lines[:]:
		if line == '' or 'hook' in line or 'verse' in line or 'intro' in line or 'outro' in line or 'chorus' in line or 'bridge' in line:
			lines.remove(line)
		words = word_tokenize(line)
		if len(words) > 0:
			for word in words:
				bow[word] += 1
			if word in line_endings:
				line_endings[words[-1]] += 1
			else:
				line_endings[words[-1]] = 1
				
	lyrics = ' '.join(lines)

	return (lyrics, lines, bow, line_endings)
	
if __name__ == '__main__':
	artist = raw_input('Enter Artist: ')
	(lyrics, lines, bow, line_endings) = get_lyrics(artist)
	target = open('lyrics.txt', 'w')
	target.truncate()
	
	lyrics = lyrics.encode('utf-8')
	target.write(lyrics)
	target.write('\n')
	
	target.close()
