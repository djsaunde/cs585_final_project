from collections import defaultdict
import string

def get_shake():
	f = open('shakespeare.txt')
	lines = []
	bow = defaultdict(float)
	line_endings = {}
	
	exclude = set(string.punctuation)

	for line in f:
		line = line.rstrip()
		line = ''.join(ch for ch in line if ch not in exclude)
		words = line.split()
		if len(words) > 1:
			lines.append(line)
			for word in words:
				bow[word] += 1
			if word in line_endings:
				line_endings[word] += 1
			else:
				line_endings[word] = 1
				
			
	lyrics = ' '.join(lines)

	
	
	return (lyrics, lines, bow, line_endings)

