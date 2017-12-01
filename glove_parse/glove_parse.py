import json
# A - I : [65, 73] U [97, 105] U [0, 65] U (122, infinity) => <= 73 or >= 97 and <= 105 and > 122
# J - S : [74, 83] U [106, 115]
# T - Z : [84, 96] U [116, 122]
glove_else = {}
glove_dict_A_I = {}
glove_dict_J_S = {}
glove_dict_T_Z = {}
with open('glove.6B.50d.txt', 'r') as glove:
	for line in glove.readlines():
		vec = line.strip().split(" ")
		ind = ord(vec[0][0])
		if ind >= 65 and ind <= 73 or ind >= 97 and ind <= 105:
			glove_dict_A_I[vec[0]] = [float(_) for _ in vec[1:]]
		elif ind >= 74 and ind <= 83 or ind >= 106 and ind <= 115:
			glove_dict_J_S[vec[0]] = [float(_) for _ in vec[1:]]
		elif ind >= 84 and ind <= 96 or ind >= 116 and ind <= 122:
			glove_dict_T_Z[vec[0]] = [float(_) for _ in vec[1:]]
		else:
			glove_else[vec[0]] = [float(_) for _ in vec[1:]]


with open('glove_dict_A_I.json', 'w') as f:
	f.write(json.dumps(glove_dict_A_I))

with open('glove_dict_J_S.json', 'w') as f:
	f.write(json.dumps(glove_dict_J_S))

with open('glove_dict_T_Z.json', 'w') as f:
	f.write(json.dumps(glove_dict_T_Z))

with open('glove_else.json', 'w') as f:
	f.write(json.dumps(glove_else))