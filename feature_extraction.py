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
fname = 'tfidf1_2_3_4.json'
with open(fname, 'r') as f:
    f = f.read()
    tfidf_count = json.loads(f)
    N = sum(tfidf_count.values())


class question_pair_feature_extraction(object):
    def __init__(self):
        self.q1 = "What is the story of Kohinoor (Koh-i-Noor) Diamond?"
        self.q2 = "What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?"
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
        C(J, B) = |S(A, w) union S(B, w)| / |S(A, w)|, here w = 1, 2 corresponding to 1-gram and 2-gram
        This weighing significantly improves performances according to the paper
        J(A, B) = |A union B| / (|A| + |B| - |A union B|)
        if A and B are both empty, we define J(A, B) = 1
        Also use information about the length of Longest Common Sub-sequence compared to the length of the sentences
        :return: a float, or a tuple of 4 float between (0, 1)
        """
        # TODO: more length features can be utilized: A union B, A - B, B - A, A U B / A, A U B / B
        #(TODO) how does LCS combines to this feature?
        LCS = self.longest_common_subsequence()
        #print (LCS)
        uni_overlap = self.containment_similarity_coefficient_with_weight(
            set(self.unigram_q1), set(self.unigram_q2))
        bi_overlap = self.containment_similarity_coefficient_with_weight(
            set(self.bigram_q1), set(self.bigram_q2))
        tri_overlap = self.jaccard_similarity_coefficient_with_weight(
            set(self.trigram_q1), set(self.trigram_q2))
        quad_overlap = self.jaccard_similarity_coefficient_with_weight(
            set(self.quadgram_q1), set(self.quadgram_q2))
        print(uni_overlap, bi_overlap, tri_overlap, quad_overlap)

    def jaccard_similarity_coefficient(self, set1, set2):
        overlap = len(set1.union(set2))
        return overlap * 1.0 / (len(set1) + len(set2) - overlap)

    def jaccard_similarity_coefficient_with_weight(self, set1, set2):
        overlap = set1.union(set2)
        overlap_weight = sum([tfidf_count.get(w, math.log(N)) for w in overlap])
        return overlap_weight * 1.0 / (sum([tfidf_count.get(a, math.log(N)) for a in set1]) + sum([tfidf_count.get(b, math.log(N)) for b in set2]) - overlap_weight)

    def containment_similarity_coefficient(self, set1, set2):
        return len(set1.union(set2)) * 1.0 / len(set1)

    def containment_similarity_coefficient_with_weight(self, set1, set2):
        overlap = set1.union(set2)
        overlap_weight = sum([tfidf_count.get(w, math.log(N)) for w in overlap])
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
        print(uni_overlap, bi_overlap, tri_overlap, quad_overlap)

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
        #print (ch_overlap)

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
            tf_idf1.append(self.unigram_q1.get(word, 0) * tfidf_count.get(word, idf_no_occurence) * 1.0)
            tf_idf2.append(self.unigram_q2.get(word, 0) * tfidf_count.get(word, idf_no_occurence) * 1.0)
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

    def co_occurrence_retrieval_model(self):
        """
        Sim_CRM(w1, w2) = 2 * |c(w1) union c(w2)| / (|c(w1)| + |c(w2)|)
        """
        pass

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
        pass

    def paragraph_to_vec(self):
        """
        An unsupervised method of learning text representation
        The paragraph token acts as a memory that remembers what information is missing from the current text
        Use cosine similarity for comparing two paragraph vectors
        :return: float
        """
        pass

    def tree_LSTM(self):
        """
        RNN processes input sentences of variable length via recursive application of a transition function
        on a hidden state vector ht. For each sentence pair it creates sentence representations hL and hR using
        Tree LSTM model. Given these representations, model predicts the similarity score using a neural network
        considering distance and angle between vectors.
        source: [1] tree-LSTM https://github.com/dasguptar/treelstm.pytorch
                [2] LSTM using torch: http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-
                    glr-beginner-nlp-sequence-models-tutorial-py
        :return: float
        """
        pass

    def word_alignment(self):
        """
        omega(A) = sum(IDF(w)), A: a set of words
        sim(S1, S2) = (omega(A1) + omega(A2)) / (omega(S1) + omega(S2))
        Sul- tan et al., 2014a; Sultan et al., 2014b; Sultan et al., 2015
        :return:
        """
        pass
s = question_pair_feature_extraction()
s.lemma_n_gram_overlaps()
