# tf-idf file for feature extraction
from collections import Counter
import math
import json
fname = '../quora_duplicate_questions.tsv'
# fname = 'quora_dup_question_feature_extraction.sample.tsv'

# strip punctuations
def strip_punctuations(word):
    punctuations = [",","?",".","!"]
    if word[-1] in punctuations:
        return word[:-1]
    return word

# count idf
def count_idf(lines):
    c = Counter()
    total_count = len(lines)*2
    c['total_count_of_corpus'] = total_count
    for line in lines:
        try:
            sentence1 = [strip_punctuations(word) for word in line.split('\t')[3].split()]
            sentence2 = [strip_punctuations(word) for word in line.split('\t')[4].split()]
            sentence1 = set(sentence1)
            sentence2 = set(sentence2)
        except:
            print (line)
            print ("-------------------")
        for w in sentence1:
            c[w] += 1
        for w in sentence2:
            c[w] += 1
    for key in c.keys():
        c[key] = math.log(float(total_count)/c[key])
    return c

with open(fname, 'r') as f:
    lines = f.readlines()[1:]
    count = len(lines)
    one_set = math.ceil(count/5.0)
    line1 = lines[:one_set+1]
    line2 = lines[one_set+1:one_set*2+1]
    line3 = lines[one_set*2+1:one_set*3+1]
    line4 = lines[one_set*3+1:one_set*4+1]
    line5 = lines[one_set*4+1:]
    set1 = line2+line3+line4+line5
    set2 = line1+line3+line4+line5
    set3 = line1+line2+line4+line5
    set4 = line1+line2+line3+line5
    set5 = line1+line2+line3+line4
# count_idf(lines)
with open('tfidf2_3_4_5.json','w') as f:
    f.write(json.dumps(count_idf(set1)))
with open('tfidf1_3_4_5.json','w') as f:
    f.write(json.dumps(count_idf(set2)))
with open('tfidf1_2_4_5.json','w') as f:
    f.write(json.dumps(count_idf(set3)))
with open('tfidf1_2_3_5.json','w') as f:
    f.write(json.dumps(count_idf(set4)))
with open('tfidf1_2_3_4.json','w') as f:
    f.write(json.dumps(count_idf(set5)))
