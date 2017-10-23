train_gold = "../data/train.gold"

with file(train_gold) as f:
	suffix_freq_count = {}
	for line in f.readlines():
		line = line.strip()
		gene = line.split("|")[2]
		for word in gene.split(" "):
			suff4 = word[-4:]
			suff3 = word[-3:]
			suff2 = word[-2:]
			suffix_freq_count[suff4] = suffix_freq_count.get(suff4, 0) + 1
			suffix_freq_count[suff3] = suffix_freq_count.get(suff3, 0) + 1
			suffix_freq_count[suff2] = suffix_freq_count.get(suff2, 0) + 1

desc_freq_suff = suffix_freq_count.keys()
desc_freq_suff.sort(key=lambda x: suffix_freq_count[x], reverse=True)
for suff in desc_freq_suff:
	if suffix_freq_count[suff] > 100 and suff != suff.upper():
		print (suff, suffix_freq_count[suff])