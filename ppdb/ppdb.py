import json
# open file
fname = 'ppdb-2.0-s-lexical'

# store word pairs
word_pair = {}
with open(fname,'r') as f:
    lines = f.readlines()
    print (len(lines))
    for line in lines:
        words = line.split(' ||| ')
        word1 = words[1]
        word2 = words[2]
        word_pair[word1] = word_pair.get(word1,[])
        word_pair[word1].append(word2)

# store the dictionary in a json
with open('ppdb-output.json','w') as f:
    f.write(json.dumps(word_pair))
