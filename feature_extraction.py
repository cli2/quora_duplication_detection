# author: Chong and Tianyi
# feature extraction
from collections import Counter
import tensorflow as tf
import nltk
import tf_glove
import gensim
class question_pair_feature_extraction(object):
    def __init__(self):
        self.q1 = ""
        self.q2 = ""

        self.q1_wordlist = []
        self.q2_wordlist = []

        self.q1_pos = []
        self.q2_pos = []

        self.unigram_q1 = Counter()
        self.unigram_q2 = Counter()

        self.bigram_q1 = Counter()
        self.bigram_q2 = Counter()

        self.trigram_q1 = Counter()
        self.trigram_q2 = Counter()

        self.quadgram_q1 = Counter()
        self.quadgram_q2 = Counter()
        self.gram_Counts = Counter() # the sum of all n-grams
        self.IDF = {}
    def lemma_n_gram_overlaps(self):
        """
        compare word n-grams in both sentences using Jaccard Similarity Coefficient
        Weighing n-grams using a sum of IDF values of words in n-gram
        Containment coefficient is used for orders n belongs to {1, 2}
        C(J, B) = |S(A, w), S(B, w)| / |S(A, w)|, here w = 1, 2 corresponding to 1-gram and 2-gram
        This weighing significantly improves performances according to the paper
        J(A, B) = |A union B| / (|A| + |B| - |A union B|)
        if A and B are both empty, we define J(A, B) = 1
        Also use information about the length of longest Common Sub-sequence compared to the length of the sentences
        :return: a float, or a tuple of 4 float between (0, 1)
        """
        #(TODO) how does LCS combines to this feature?
        pass

    def pos_n_gram_overlaps(self):
        """
        Calculate JSC and containment coefficient for n-grams of POS tags
        :return: a float, float, or a tuple of 4 float between (0, 1)
        """
        q1_pos_count = Counter()
        q2_pos_count = Counter()
        for _, pos in self.q1_pos:
            q1_pos_count[pos] += 1
        for _, pos in self.q2_pos:
            q2_pos_count[pos] += 1
        pass

    def character_n_gram_overlaps(self):
        """
        Use Jaccard Similarity Coefficient and Containment Coefficient for comparing common substrings
        in both sentences.
        IDF weights are computed on character n-gram level
        Enrich the feature by Greedy String Tiling - with LCS
        :return: float, or a tuple of 4 float between (0, 1)
        """
        #(TODO) the GST algorithm is a little bit complicated, we can see if there's open source code
        pass

    def tf_idf(self):
        """
        For each word in the sentence we calculate tf-idf
        The similarity between two sentences is expressed as cosine similarity between corresponding TF-IDF vectors
        :return: float
        """

    def longest_common_subsequence(self):
        """
        source:wikipedia [1]
        [1] https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
        :return: int, the length of LCS
        """
        m = len(self.q1)
        n = len(self.q2)
        dp = [[0 for _ in xrange(m + 1)] for __ in xrange(n + 1)]
        for i in xrange(1, m):
            for j in xrange(1, n):
                if self.q1[i] == self.q2[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
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
        pass

    def tokenize_questions(self):
        self.q1_wordlist = nltk.wordpunct_tokenize(self.q1.lower())
        self.q2_wordlist = nltk.wordpunct_tokenize(self.q2.lower())

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
        :return:
        """