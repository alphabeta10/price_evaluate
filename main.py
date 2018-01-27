# import modules & set up logging
#import gensim, logging
import os
import nltk
import re
import pickle
import save_data
import csv
from tqdm import tqdm
import numpy as np
"""
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
lee_train_file = test_data_dir + 'lee_background.cor'
print(lee_train_file)
class MyText(object):
    def __iter__(self):
        for line in open(lee_train_file):
            # assume there's one document per line, tokens separated by whitespace
            yield line.lower().split()
sentences = MyText()
for sen in sentences:
    print(sen)
"""
word2vector = {}
def load_word2vector():
    word2vector = {}
    if not os.path.exists("doc2vector.pickle"):
        with open("E:\\文档\\nlp\\glove6b\\glove.6B.300d.txt", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                splits = line.split(" ")
                word = splits[0]
                vector = [float(v) for v in splits[1:]]
                # print(vector)
                if word not in word2vector:
                    word2vector[word] = vector
        with open("E:\\文档\\nlp\\glove6b\\doc2vector.pickle", 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2vector':word2vector
            }
            pickle.dump(data, handle, -1)  #
    else:
        with open("E:\\文档\\nlp\\glove6b\\doc2vector.pickle", 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            word2vector = data['word2vector']
    return word2vector
word2vector = load_word2vector()
def extractText(line,stopwords, isTarget=False):
    sentences = []
    sentencesToken = nltk.sent_tokenize(line)
    for i in range(len(sentencesToken)):
        tokens = nltk.word_tokenize(sentencesToken[i])
        for token in tokens:
            if token.lower() in word2vector:
                sentences.append(token.lower())
    return sentences
def filter_dataset(datas):
    filter_datas = []
    stop_words = save_data.load_stop_words("stopword.txt")
    for elem in tqdm(datas,desc="datas"):
        datas_ = []
        def filter_fn(elem_str):
            if elem_str.replace(" ", "") is not "":
                data = elem_str.replace("[rm]", "")
                filter_data = re.sub("\w", "", data)
                filter_data = filter_data.replace(" ", "")
                filter_character = []
                for ce in filter_data:
                    filter_character.append(ce.replace(" ", ""))
                filter_character = list(set(filter_character))
                def get_words(desc):
                    sum = 0
                    tokens = nltk.sent_tokenize(desc)
                    for token in tokens:
                        words = nltk.word_tokenize(token)
                        sum += len(words)
                    return sum
                for filter_elem in filter_character:
                    data = data.replace(filter_elem, "")
                if len(data.replace(" ", "")) > 0 and get_words(data) > 0:
                    return data
                else:
                   return ""
            else:
                return ""
        for i in range(len(elem)-1):
            if i==2:
                new_str =elem[i].replace("& ","").replace("/"," ").replace(",","")
            else:
                new_str = filter_fn(elem[i])
            try:
                #no_word = extractText(new_str)
                words = extractText(new_str,stop_words)
            except Exception as e:
               print(e)
               if new_str is None:
                   print("str_ ",elem[i])
               else:
                   print("new_str ",new_str)
                   words = []
            if len(words)is not 0:
                """
                print("str_ ",elem[i])
                print("new_str ",new_str)
                print("no word ",no_word)
                """
                datas_.append(words)
            else:
                datas_.append([])
        #datas_ = list(set(datas_))
        if len(datas_) is not 0:
            filter_datas.append([datas_,float(elem[-1])])
    return filter_datas
def words2array(words,size):
    array = []
    for word in words:
        if word in word2vector:
            array.append(word2vector[word])
    if len(array)>size:
        return np.array(array[:size]).reshape(size,300)
    else:
        x = [[5] * 300] * (size - len(array))
        array.extend(x)
        return np.array(array).reshape(size,300)

save_fileName = "all_data04.pltk"
if not os.path.exists(save_fileName):
    #lt50_datas.txt
    #D:\\pychardir\\price_evaluate\\data\\train.tsv
    with open("D:\\pychardir\\price_evaluate\\data\\train.tsv", 'r', encoding='utf8') as f:
        p = csv.reader(f, delimiter="\t")
        index = 0
        datas = []
        for i, r in tqdm(enumerate(p), desc="iteratordata"):
            if index == 0:
                print(r)
                index += 1
            else:
                elem = r[-1].strip().replace("\n", "").replace(" [rm]", "")
                datas.append([r[1],r[2], r[3], r[4],r[6], elem, r[5]])
    fiter_datas = filter_dataset(datas)
    saveData = {
        "all_data": fiter_datas
    }
    save_data.save_data(save_fileName,saveData)
else:
    saveData = save_data.load_data(save_fileName)

    name_max = 0
    category_max = 0
    brand_max = 0
    sum_len = 0
    for elem in saveData["all_data"]:
        data = elem[0]
        if sum_len == 0:
            print(data)
            sum_len+=1
        name_temp_len = len(data[0])
        category_temp_len = len(data[2])
        brand_temp_len = len(data[3])
        if name_temp_len >name_max:
            name_max = name_temp_len
        if category_temp_len > category_max:
            category_max = category_temp_len
        if brand_temp_len > brand_max:
            brand_max = brand_temp_len
    print("name max ",name_max)
    print("categroy name ",category_max)
    print("brand name ",brand_max)
def generator_batch(batch_size,datas):
    def generator_data(datas,batch_size):
        data_size = len(datas)
        for i in range(0,data_size,batch_size):
            yield datas[i:min(i+batch_size,data_size)]
    batchs = []
    for words_price in tqdm(generator_data(datas,batch_size),desc="load_data"):
       batchs.append(words_price)
       break
    return batchs
batches = generator_batch(2,saveData["all_data"])
"""
for b in batches:
   npdata = np.zeros((len(b),100,300))
   labels = np.zeros((len(b),1))
   for k in range(len(b)):
       labels[k] =b[k][1]
       npdata[k,:,:] = words2array(b[k][0],100)
   print(npdata)
   print(labels)
   break
   """
#import TWord2VectorModel0 as model
#model.train_model(batches)

