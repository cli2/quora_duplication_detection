# author: Chong and Tianyi
# feature extraction
from collections import Counter
import tensorflow as tf
import nltk
from nltk.util import ngrams
import tf_glove
import gensim
import json
import math
import pandas as pd
import sys
import time
import numpy as np
import sys
tfidf_file_path=sys.argv[1]
result_file_path=sys.argv[2]
resultfile=open(result_file_path,'w')
# reload(sys)
# sys.setdefaultencoding("utf-8")

# open the tfidf data
fname = 'tfidf_data/'+tfidf_file_path
with open(fname, 'r') as f:
    f = f.read()
    tfidf_count = json.loads(f)
    N = tfidf_count['total_count_of_corpus']

# open the ppdb data for word alignment
fname = 'ppdb/ppdb-output.json'
with open(fname, 'r') as f:
    f = f.read()
    ppdb = json.loads(f)

# open the glove data for word embedding
fname = 'glove_parse/glove_dict_A_I.json'
with open(fname, 'r') as f:
    f = f.read()
    glove_dict_A_I = json.loads(f)
fname = 'glove_parse/glove_dict_J_S.json'
with open(fname, 'r') as f:
    f = f.read()
    glove_dict_J_S = json.loads(f)
fname = 'glove_parse/glove_dict_T_Z.json'
with open(fname, 'r') as f:
    f = f.read()
    glove_dict_T_Z = json.loads(f)
fname = 'glove_parse/glove_else.json'
with open(fname, 'r') as f:
    f = f.read()
    glove_else = json.loads(f)


class question_pair_feature_extraction(object):
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2
        self.q1 = self.q1.lower()
        self.q2 = self.q2.lower()
        # self.q1_wordlist = []
        # self.q2_wordlist = []

        self.q1_pos = nltk.pos_tag(self.q1)
        self.q2_pos = nltk.pos_tag(self.q2)

        self.q1_wordlist = nltk.wordpunct_tokenize(self.q1)
        self.q2_wordlist = nltk.wordpunct_tokenize(self.q2)
        # print (self.q1_wordlist)

        self.unigram_q1 = Counter(self.q1_wordlist)
        self.unigram_q2 = Counter(self.q2_wordlist)

        self.bigram_list_q1 = list(ngrams(self.q1_wordlist, 2))
        self.bigram_list_q2 = list(ngrams(self.q2_wordlist, 2))
        self.bigram_q1 = Counter(ngrams(self.q1_wordlist, 2))
        self.bigram_q2 = Counter(ngrams(self.q2_wordlist, 2))

        self.trigram_q1 = Counter(ngrams(self.q1_wordlist, 3))
        self.trigram_q2 = Counter(ngrams(self.q2_wordlist, 3))

        self.quadgram_q1 = Counter(ngrams(self.q1_wordlist, 4))
        self.quadgram_q2 = Counter(ngrams(self.q2_wordlist, 4))

        self.gram_Counts = Counter()  # the sum of all n-grams
        self.IDF = {}

    def lemma_n_gram_overlaps(self):
        """
        compare word n-grams in both sentences using Jaccard Similarity Coefficient
        Weighing n-grams using a sum of IDF values of words in n-gram
        Containment coefficient is used for orders n belongs to {1, 2}
        C(J, B) = |S(A, w) intersection S(B, w)| / |S(A, w)|, here w = 1, 2 corresponding to 1-gram and 2-gram
        This weighing significantly improves performances according to the paper
        J(A, B) = |A intersection B| / (|A| + |B| - |A intersection B|)
        if A and B are both empty, we define J(A, B) = 1
        Also use information about the length of Longest Common Sub-sequence compared to the length of the sentences
        :return: a float, or a tuple of 4 float between (0, 1)
        """
        # TODO: more length features can be utilized: A intersection B, A - B, B - A, A U B / A, A U B / B
        #(TODO) how does LCS combines to this feature?
        LCS = self.longest_common_subsequence()
        # print (LCS)
        uni_overlap = self.containment_similarity_coefficient_with_weight(
            set(self.unigram_q1), set(self.unigram_q2))
        bi_overlap = self.containment_similarity_coefficient_with_weight(
            set(self.bigram_q1), set(self.bigram_q2))
        tri_overlap = self.jaccard_similarity_coefficient_with_weight(
            set(self.trigram_q1), set(self.trigram_q2))
        quad_overlap = self.jaccard_similarity_coefficient_with_weight(
            set(self.quadgram_q1), set(self.quadgram_q2))
        return uni_overlap, bi_overlap, tri_overlap, quad_overlap

    def jaccard_similarity_coefficient(self, set1, set2):
        if not set1 or not set2 or len(set1) <= 1 or len(set2) <= 1:
            return 0.0
        overlap = len(set1.intersection(set2))
        return overlap * 1.0 / (len(set1) + len(set2) - overlap)

    def jaccard_similarity_coefficient_with_weight(self, set1, set2):
        if not set1 or not set2 or len(set1) <= 1 or len(set2) <= 1:
            return 0.0
        overlap = set1.intersection(set2)
        overlap_weight = sum([tfidf_count.get(w, math.log(N))
                              for w in overlap])
        return overlap_weight * 1.0 / (sum([tfidf_count.get(a, math.log(N)) for a in set1]) + sum([tfidf_count.get(b, math.log(N)) for b in set2]) - overlap_weight)

    def containment_similarity_coefficient(self, set1, set2):
        if not set1 or not set2 or len(set1) <= 1 or len(set2) <= 1:
            return 0.0
        return len(set1.intersection(set2)) * 1.0 / len(set1)

    def containment_similarity_coefficient_with_weight(self, set1, set2):
        if not set1 or not set2 or len(set1) <= 1 or len(set2) <= 1:
            return 0.0
        overlap = set1.intersection(set2)
        overlap_weight = sum([tfidf_count.get(w, math.log(N))
                              for w in overlap])
        return overlap_weight * 1.0 / sum([tfidf_count.get(a, math.log(N)) for a in set1])

    def pos_n_gram_overlaps(self):
        """
        Calculate JSC and containment coefficient for n-grams of POS tags
        :return: a float, float, or a tuple of 4 float between (0, 1)
        """
        pos_list_q1 = [pos for _, pos in self.q1_pos]
        pos_list_q2 = [pos for _, pos in self.q2_pos]

        pos_uni_q1 = set(pos_list_q1)
        pos_uni_q2 = set(pos_list_q2)
        uni_overlap = self.containment_similarity_coefficient(
            pos_uni_q1, pos_uni_q2)

        pos_bi_q1 = set(ngrams(pos_list_q1, 2))
        pos_bi_q2 = set(ngrams(pos_list_q2, 2))
        bi_overlap = self.containment_similarity_coefficient(
            pos_bi_q1, pos_bi_q2)

        pos_tri_q1 = set(ngrams(pos_list_q1, 3))
        pos_tri_q2 = set(ngrams(pos_list_q2, 3))
        tri_overlap = self.jaccard_similarity_coefficient(
            pos_tri_q1, pos_tri_q2)

        pos_quad_q1 = set(ngrams(pos_list_q1, 4))
        pos_quad_q2 = set(ngrams(pos_list_q2, 4))
        quad_overlap = self.jaccard_similarity_coefficient(
            pos_quad_q1, pos_quad_q2)
        return uni_overlap, bi_overlap, tri_overlap, quad_overlap

    def gst(self, a, b, minlength=2):
        """
        source: https://github.com/platinhom/platinhom.github.com/blob/master/_posts/2016-02-16-Greedy-String-Tiling.md
        """
        if len(a) == 0 or len(b) == 0:
            return []
        # if py>3.0, nonlocal is better

        class markit:
            a = [0]
            minlen = 2
        markit.a = [0] * len(a)
        markit.minlen = minlength

        # output index
        out = []

        # To find the max length substr (index)
        # apos is the position of a[0] in origin string
        def maxsub(a, b, apos=0, lennow=0):
            if (len(a) == 0 or len(b) == 0):
                return []
            if (a[0] == b[0] and markit.a[apos] != 1):
                return [apos] + maxsub(a[1:], b[1:], apos + 1, lennow=lennow + 1)
            elif (a[0] != b[0] and lennow > 0):
                return []
            return max(maxsub(a, b[1:], apos), maxsub(a[1:], b, apos + 1), key=len)

        # Loop to find all longest substr until the length < minlength
        while True:
            findmax = maxsub(a, b, 0, 0)
            if (len(findmax) < markit.minlen):
                break
            else:
                for i in findmax:
                    markit.a[i] = 1
                out += findmax
        return [a[i] for i in out]

    def character_n_gram_overlaps(self):
        """
        Use Jaccard Similarity Coefficient and Containment Coefficient for comparing common substrings
        in both sentences.
        IDF weights are computed on character n-gram level
        Enrich the feature by Greedy String Tiling - with LCS
        :return: float, or a tuple of 4 float between (0, 1)
        """
        #(TODO) the GST algorithm is a little bit complicated, we can see if this feature is necessary
        bigram_overlap = self.gst(self.bigram_list_q1, self.bigram_list_q2)
        ch_overlap = len(bigram_overlap) * 1.0 / len(self.bigram_q1)
        # print (ch_overlap)
        return ch_overlap

    def tf_idf(self):
        """
        For each word in the sentence we calculate tf-idf
        The similarity between two sentences is expressed as cosine similarity between corresponding TF-IDF vectors
        :return: float
        """
        word_set = set(list(self.unigram_q1) + list(self.unigram_q2))
        tf_idf1 = []
        tf_idf2 = []
        idf_no_occurence = math.log(N)
        for word in word_set:
            tf_idf1.append(self.unigram_q1.get(word, 0) *
                           tfidf_count.get(word, idf_no_occurence) * 1.0)
            tf_idf2.append(self.unigram_q2.get(word, 0) *
                           tfidf_count.get(word, idf_no_occurence) * 1.0)
        cosine = 0
        for i in range(len(word_set)):
            cosine += tf_idf1[i] * tf_idf2[i]
        len1 = math.sqrt(sum(i * i for i in tf_idf1))
        len2 = math.sqrt(sum(i * i for i in tf_idf2))
        return cosine / (len1 * len2)

    def longest_common_subsequence(self):
        """
        source:wikipedia [1]
        [1] https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
        :return: int, the length of LCS
        """
        m = len(self.q1)
        n = len(self.q2)
        dp = [[0 for _ in range(n + 1)] for __ in range(m + 1)]
        for i in range(0, m):
            for j in range(0, n):
                if self.q1[i] == self.q2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[m][n]

    # Semantic Similarity

    def semantic_composition(self):
        """
        Use simple linear combination of word vectors, where weights are represented by the TF-IDF values
        of appropriate words
        Use state-of-the-art word embedding methods: Continuous Bag of Words(CBOW) and Global Vectors(GloVe)
        Use cosine similarity to compare vectors
        :return: float
        """
        # (TODO) Learn all corpus and generate the vector or just learn current sentence pair?
        words1 = self.q1_wordlist[:]
        words2 = self.q2_wordlist[:]
        unknown = (np.random.rand(1,50)-0.5)/2.0
        def calculate_vector(word_list):
            idf_no_occurence = math.log(N)
            vector = np.zeros((1, 50))
            for word in word_list:
                ind = ord(word[0])
                if ind >= 65 and ind <= 73 or ind >= 97 and ind <= 105:
                    vector += np.array(glove_dict_A_I.get(word,unknown)) * tfidf_count.get(word, idf_no_occurence)
                elif ind >= 74 and ind <= 83 or ind >= 106 and ind <= 115:
                    vector += np.array(glove_dict_J_S.get(word,unknown)) * tfidf_count.get(word, idf_no_occurence)
                elif ind >= 84 and ind <= 96 or ind >= 116 and ind <= 122:
                    vector += np.array(glove_dict_T_Z.get(word,unknown)) * tfidf_count.get(word, idf_no_occurence)
                else:
                    vector += np.array(glove_else.get(word,unknown)) * tfidf_count.get(word, idf_no_occurence)
            return vector
        vector1 = np.squeeze(np.asarray(calculate_vector(words1)))
        vector2 = np.squeeze(np.asarray(calculate_vector(words2)))
        return np.dot(vector1, vector2) * 1.0 / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


    def word_alignment(self):
        """
        omega(A) = sum(IDF(w)), A: a set of words
        sim(S1, S2) = (omega(A1) + omega(A2)) / (omega(S1) + omega(S2))
        Sul- tan et al., 2014a; Sultan et al., 2014b; Sultan et al., 2015
        :return:
        """
        # define the function to calculate the total idf score of a set of words
        def omega(s):
            total = 0
            no_show_score = math.log(N)
            for word in s:
                score = tfidf_count.get(word, no_show_score)
                total += score
            return total
        def remove_words(word_list,remove_list):
            for word in remove_list:
                try:
                    word_list.remove(word)
                except:
                    continue
        sum_q1 = omega(self.q1_wordlist)
        sum_q2 = omega(self.q2_wordlist)
        words1 = self.q1_wordlist[:]
        words2 = self.q2_wordlist[:]
        set1 = []
        set2 = []
        # test if a word is in q2
        for word in words1:
            if word in words2:
                set1.append(word)
                set2.append(word)
        remove_words(words1,set1)
        remove_words(words2,set2)
        # test if a paraphrase of a word is in q2
        for word in words1:
            paraphrase_list = ppdb.get(word, [])
            if len(paraphrase_list) != 0:
                for word_ in words2:
                    if word_ in paraphrase_list:
                        set1.append(word)
                        set2.append(word_)
                        # print (word,word_)
                    break
        remove_words(words1,set1)
        remove_words(words2,set2)
        # test if the remaining words in q2 is in q1
        for word in words2:
            paraphrase_list = ppdb.get(word, [])
            if len(paraphrase_list) != 0:
                for word_ in words1:
                    if word_ in paraphrase_list:
                        set1.append(word_)
                        set2.append(word)
                        # print (word,word_)
                    break
        remove_words(words1,set1)
        remove_words(words2,set2)
        sum_set1 = omega(set1) * 1.0
        sum_set2 = omega(set2) * 1.0
        return (sum_set1 + sum_set2) / (sum_q1 + sum_q2)

# test cases
s = question_pair_feature_extraction('What is the step by step guide to invest in share market in india?','What is the step by step guide to invest in share market?')
#print (s.semantic_composition())
s = question_pair_feature_extraction('Method to find separation of slits using fresnel biprism?','What are some of the things technicians can tell about the durability and reliability of Laptops and its components?')
#print (s.semantic_composition())

def get_empty_feature(_id,q1, q2, is_duplicate):
    return [_id,q1, q2, is_duplicate, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
def generate_feature_vectors(fname):
    features_header = ["id","q1", "q2","result","lemma1", "lemma2", "lemma3", "lemma4", "pos1", "pos2", "pos3", "pos4", "tfidf", "word_alignment", "semantic_composition"]
    feature_output_df = pd.DataFrame(columns = features_header)
    start = time.time()
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
        # lines = f.readlines()[1:3000]
        count = len(lines)
        one_set = math.ceil(count/5.0)
        line1 = lines[:one_set+1]
        line2 = lines[one_set+1:one_set*2+1]
        line3 = lines[one_set*2+1:one_set*3+1]
        line4 = lines[one_set*3+1:one_set*4+1]
        line5 = lines[one_set*4+1:]
        _id=0
        line_all=[line1,line2,line3,line4,line5]
        for num in range(0,5):
            linechunck=line_all[num]
            start=time.time()
            for line in linechunck:
                q1, q2, is_duplicate = line.strip().split("\t")
                if len(q1) <= 1 or len(q2) <= 1:
                    # print ("Case %s is a pair of invalid sentence, output empty feature to file" %_id)
                    feature_output_df.loc[len(feature_output_df)] = get_empty_feature(_id,q1, q2, is_duplicate)
                    continue
                pair_obj = question_pair_feature_extraction(q1, q2)
                lemma1, lemma2, lemma3, lemma4 = pair_obj.lemma_n_gram_overlaps()
                pos1, pos2, pos3, pos4 = pair_obj.pos_n_gram_overlaps()
                # ch_overlap = pair_obj.character_n_gram_overlaps()
                tfidf = pair_obj.tf_idf()
                word_alignment = pair_obj.word_alignment()
                semantic_composition = pair_obj.semantic_composition()
                entry = [_id,q1, q2, is_duplicate, lemma1, lemma2, lemma3, lemma4, pos1, pos2, pos3, pos4, tfidf, word_alignment, semantic_composition]
                current_time = time.time()
                # print "time:", current_time - start
                start = current_time
                # print entry
                feature_output_df.loc[len(feature_output_df)] = entry
                _id+=1
                if int(_id)%1000==0:
                    print (_id)
            if num==0:
                feature_output_df.to_csv(result_file_path, mode = 'a')
            else:
                feature_output_df.to_csv(result_file_path, mode = 'a',header=False)
            feature_output_df = pd.DataFrame(columns = features_header)
            current_time = time.time()
            print ("time:", current_time - start)
            print ('finish one line chunck')

generate_feature_vectors('quora_lstm.tsv')
# generate_feature_vectors('quora_duplicate_questions.tsv')
