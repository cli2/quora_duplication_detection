#Random Baseline based on data distribution 
#Anna Zheng
import numpy
import math
fname='quora_lstm.tsv'

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
	line_all=[set1,set2,set3,set4,set5]
	test_all=[line1,line2,line3,line4,line5]
	for num in range(0,5):
		linechunck=line_all[num]
		is_duplicate_count=0
		not_duplicate_count=0
		for line in linechunck:
			q1, q2, is_duplicate = line.strip().split("\t")
			if int(is_duplicate)==0:
				not_duplicate_count+=1
			else:
				is_duplicate_count+=1
		total=(not_duplicate_count+is_duplicate_count)*1.0
		distribution=[not_duplicate_count/total,is_duplicate_count/total]
		real_label=[]
		predict_result=[]
		for line in test_all[num]:
			q1, q2, is_duplicate = line.strip().split("\t")
			real_label.append(is_duplicate)
			predict_result.append(numpy.random.choice(numpy.arange(0, 2), p=distribution))
		correct=0
		for n in range(len(real_label)):
			if int(real_label[n])==int(predict_result[n]):
				correct+=1
		print (correct/len(real_label))



