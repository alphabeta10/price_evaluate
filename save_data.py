import pickle

def save_data(filename,data):
    with open(filename,'wb') as f:
        pickle.dump(data,f,-1)
def load_data(filename):
    with open(filename,"rb") as f:
        return pickle.load(f)
def load_stop_words(filename):
    with open(filename,encoding="utf8") as f:
        lines = f.readlines()
        list_ = []
        for line in lines:
            list_.append(line.replace("\n","").replace(" ",""))
        return list_
#print(load_stop_words("stopword.txt"))
"""
list_data = [1,2,3,4]
filter_list = [ elem for elem in list_data if elem not in [1,2]]
print(filter_list)

size = 4
array = [[3,3,3]]

x = [[5]*3]*(size-len(array))
array.extend(x)
import numpy as np
print(np.array(array).reshape(4,3))
for e in x:
    print(e)
"""