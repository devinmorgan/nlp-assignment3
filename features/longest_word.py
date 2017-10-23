train_gold = "../data/train.gold"

with file(train_gold) as f:
	suffix_freq_count = {}
	max_length = float("-inf")
	longest_word = None
	for line in f.readlines():
		line = line.strip()
		gene = line.split("|")[2]
		for word in gene.split(" "):
			if len(word) > max_length:
				max_length = len(word)
				longest_word = word
print (longest_word, max_length)