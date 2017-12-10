# clean the quora data set and output data files for LSTM training and feature extraction
# author: Tianyi Liu
output = open('quora_lstm.tsv', 'w')
res = ""
i = 0
with open('quora_duplicate_questions.tsv', 'r') as f:
    for line in f.readlines():
        if i == 0:
            i = 1
            continue
        try:
            _id, qid1, qid2, q1, q2, is_duplicate = line.strip().split("\t")
        except ValueError:
            _id, qid1, qid2, q1, q2, s3, is_duplicate = line.strip().split("\t")
        if len(q1) <= 1 or len(q2) <= 1:
            continue
        res += q1 + "\t" + q2 + "\t"  + "%s" % is_duplicate + "\n"
output.write(res)
output.close()
