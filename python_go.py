import logging
import re
from tqdm import tqdm  #
import nltk
lines = ["hello world","think you"]
MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3
for line in lines:
    print(line.split())
x = list(map(lambda x:x.split(),lines))
def create_dict1(dict_path, corpus, max_vocab=None):
    logging.info("Create dict {}.".format(dict_path))
    counter = {}
    for line in corpus:
        for word in line:
            new_word_ = word.replace("[rm]", "").lower().replace("\n","")
            new_word = "".join([a for a in new_word_ if a.isalpha()])
            if new_word.replace(" ","") is not "":
                if new_word not in counter.keys():
                    counter[new_word] = 0
                counter[new_word] += 1
            elif "-" not in word or "/" not in word:
                new_word2 = re.sub("\D","",new_word_)
                if new_word2.replace(" ","") is not "":
                    if new_word2 not in counter.keys():
                        counter[new_word2] = 0
                    counter[new_word2] += 1

    for mark_t in MARKS:
        if mark_t in counter:
            del counter[mark_t]
            logging.warning("{} appears in corpus.".format(mark_t))

    counter = list(counter.items())
    counter.sort(key=lambda x: -x[1])
    words = list(map(lambda x: x[0], counter))
    words = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO] + words
    if max_vocab:
        words = words[:max_vocab]

    tok2id = dict()
    id2tok = dict()
    with open(dict_path, 'w',encoding="utf8") as dict_file:
        for idx, tok in enumerate(words):
            print(idx, tok, file=dict_file)
            tok2id[tok] = idx
            id2tok[idx] = tok

    logging.info(
        "Create dict {} with {} words.".format(dict_path, len(words)))
    return (tok2id, id2tok)
def create_dict(dict_path, corpus, max_vocab=None):
    logging.info("Create dict {}.".format(dict_path))
    counter = {}
    for line in corpus:
        for word in line:
            try:
                counter[word] += 1
            except:
                counter[word] = 1

    for mark_t in MARKS:
        if mark_t in counter:
            del counter[mark_t]
            logging.warning("{} appears in corpus.".format(mark_t))

    counter = list(counter.items())
    counter.sort(key=lambda x: -x[1])
    words = list(map(lambda x: x[0], counter))
    words = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO] + words
    if max_vocab:
        words = words[:max_vocab]

    tok2id = dict()
    id2tok = dict()
    with open(dict_path, 'w',encoding="utf8") as dict_file:
        for idx, tok in enumerate(words):
            print(idx, tok, file=dict_file)
            tok2id[tok] = idx
            id2tok[idx] = tok

    logging.info(
        "Create dict {} with {} words.".format(dict_path, len(words)))
    return (tok2id, id2tok)
def go1():
    data_test_file = "data/test.tsv"
    data_train_file = "data/train.tsv"
    with open(data_train_file,encoding="utf8") as f:
        lines = f.readlines()
    print(lines[0].split("\t"))

    data = list(map(lambda x:x.split("\t"),lines[1:]))
    names = []
    category_name = []
    brand_names = []
    desc = []
    for l in data:
        if "&" in l[3]:
            for elem in l[3].split("&"):
                if "/" in elem:
                    for e in elem.split("/"):
                        category_name.append(e.strip())
                else:
                    category_name.append(elem.strip())
        elif "/" in l[3]:
            for e in l[3].split("/"):
                category_name.append(e.strip())
        elif l[3].replace(" ", "") is not "":
            category_name.append(l[3].replace(" ", ""))
        names.append(l[1].strip())
        if l[4].replace(" ", "") is not "":
            brand_names.append(l[4].strip())
        if l[-1].strip() is not "":
            desc.append(l[-1].strip().replace("\n", ""))
    set_brand_names = list(set(brand_names))
    set_category_names = list(set(category_name))
    set_names = list(set(names))
    list_desc = []
    for sentence in desc:
        tokens = nltk.sent_tokenize(sentence)
        temp_sentence_word = []
        for token in tokens:
            words = nltk.word_tokenize(token)
            for word in words:
                temp_sentence_word.append(word)
        list_desc.append(temp_sentence_word)
    list_desc.append(brand_names)
    list_desc.append(category_name)
    list_desc.append(names)
    tok2id,id2tok = create_dict("dict2.txt",list_desc)

def go2():
    desc = ["Cute Lululemon tank top size 4. Overall this looks really cute. There is some spots with pilling. Also the strap in the back is a list slightly yellow. The cup inserts are not included. There's a minor stain on the front of the shirt."]
    for data in desc:#
        tokens = nltk.sent_tokenize(data)
        for token in tokens:
            words = nltk.word_tokenize(token)
            print(words)
        #print(tokens)
    counter = {}
    counter["go"] = 1
    counter["dictc"] = 1

    bool_ = "go" in counter
    print(bool_)
    print(counter)
def handle(line_data):
    desc = line_data[-1]
    name = line_data[1]
    brand_name = line_data[4]
    category_name = line_data[3]
    desc = extractText(desc)
    name = extractText(name)
    brand_name = extractText(brand_name)
    category_name = extractText(category_name)
    print("desc",desc)
    print("name",name)
    print("brand_name",brand_name)
    print("category_name",category_name)
    print("---------------------------")
def extractText(line):
    sentences = []  # List[List[str]]
    sentencesToken = nltk.sent_tokenize(line)
    for i in range(len(sentencesToken)):
        tokens = nltk.word_tokenize(sentencesToken[i])
        tempWords = []
        for token in tokens:
            tempWords.append(token)  # Create the vocabulary and the training sentences
        sentences.append(tempWords)
    return sentences
l_data = ['', 'name', 'item_condition_id', 'category_name', 'brand_name', 'price', 'shipping', '']
handle(l_data)
def go3():
    data_train_file = "data/train.tsv"
    with open(data_train_file, encoding="utf8") as f:
        lines = f.readlines()
    print(lines[0].split("\t"))

    datas = list(map(lambda x: x.split("\t"), lines[1:]))
    for data in tqdm(datas[:10],desc="handler data"):
        handle(data)
#go3()
import numpy as np
x = np.random.randint(1,10)
print(x)

sampple = [1,2,3,4,5,6,7,8,9,10,11,12,13]
print(sampple[x])
sample = [[[119, 120, 121, 65, 122, 123, 54, 124, 125, 65, 126, 127, 128, 65, 129, 121, 21, 130, 53, 30],[1,2,3,4]], [[131, 132, 133, 82, 134]], [], [[135]], '59.0']
desc = []
for elem in sample[0]:
    desc.extend(elem)
print(desc)