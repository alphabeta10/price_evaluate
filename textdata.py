# Copyright 2015 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Loads the dialogue corpus, builds the vocabulary
"""

import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string
import collections

from read_doc import ReadDoc
class Batch:
    """Struct containing batches info
    desc, name,brand_name,category_name,target_price
    """
    def __init__(self):
        self.desc = []
        self.other_elem = []
        self.target = []
        self.desc_length = []
        self.other_elem_length = []

class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """
    availableCorpus = collections.OrderedDict([  # OrderedDict because the first element is the default choice
        ('cornell', ReadDoc)
    ])

    @staticmethod
    def corpusChoices():
        """Return the dataset availables
        Return:
            list<string>: the supported corpus
        """
        return list(TextData.availableCorpus.keys())

    def __init__(self, filename,doc_dir,maxLength,filterVocab,vocabularySize):

        self.filename = filename
        self.maxLength = maxLength
        self.filterVocab = filterVocab
        self.vocabularySize = vocabularySize
        self.doc_dir = doc_dir
        basePath = self._constructBasePath()
        self.fullSamplesPath = basePath + '.pkl'  # Full sentences length/vocab
        self.filteredSamplesPath = basePath + '-length{}-filter{}-vocabSize{}.pkl'.format(
            self.maxLength,
            self.filterVocab,
            self.vocabularySize,)
        self.padToken = -1  # Padding
        self.goToken = -1  # Start of sequence
        self.eosToken = -1  # End of sequence
        self.unknownToken = -1  # Word dropped from vocabulary

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.word2id = {}
        self.id2word = {}  # For a rapid conversion (Warning: If replace dict by list, modify the filtering to avoid linear complexity with del)
        self.idCount = {}  # Useful to filters the words (TODO: Could replace dict by list or use collections.Counter)

        self.loadCorpus()
        self._printStats()
        self._printSamples()
    def _printSamples(self):
        for i in range(10):
            sample_index = np.random.randint(1, self.getSampleSize())
            print(self.trainingSamples[sample_index])
    def _printStats(self):
        print('Loaded {}: {} words, {} QA'.format("doc", len(self.word2id), len(self.trainingSamples)))

    def _constructBasePath(self):
        """Return the name of the base prefix of the current dataset
        """
        path = os.path.join(self.doc_dir,'samples/')
        if os.path.exists(path) is False:
            os.makedirs(path)
        path += 'dataset-{}'.format("price_evaluate")
        return path

    def makeLighter(self, ratioDataset):
        """Only keep a small fraction of the dataset, given by the ratio
        """
        #if not math.isclose(ratioDataset, 1.0):
        #    self.shuffle()  # Really ?
        #    print('WARNING: Ratio feature not implemented !!!')
        pass

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)

    def _createBatch(self, samples):
        batch = Batch()
        #desc, name,brand_name,category_name,target_price
        batchSize = len(samples)
        for i in range(batchSize):
            sample = samples[i]
            descs = sample[0]
            desc_merge = []
            for elem in descs:
                desc_merge.extend(elem)
            other_merge = []
            name = sample[1]
            brand_name = sample[2]
            category_name = sample[3]
            for elem in name:
                other_merge.extend(elem)
            for elem in brand_name:
                other_merge.extend(elem)
            for elem in category_name:
                other_merge.extend(elem)
            target_price = float(sample[4])
            batch.desc.append(desc_merge)
            batch.desc_length.append(len(desc_merge))
            batch.other_elem.append(other_merge)
            batch.other_elem_length.append(len(other_merge))
            batch.target.append(target_price)
        return batch

    def getBatches(self,batchSize):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle()

        batches = []

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """

            for i in range(0, self.getSampleSize(), batchSize):
                yield self.trainingSamples[i:min(i + batchSize, self.getSampleSize())]

        # TODO: Should replace that by generator (better: by tf.queue)

        for samples in genNextSamples():
            batch = self._createBatch(samples)
            batches.append(batch)
        return batches

    def getSampleSize(self):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.trainingSamples)

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2id)

    def loadCorpus(self):
        """Load/create the conversations data
        """
        datasetExist = os.path.isfile(self.filteredSamplesPath)
        if not datasetExist:  # First time we load the database: creating all files
            print('Training samples not found. Creating dataset...')

            datasetExist = os.path.isfile(self.fullSamplesPath)  # Try to construct the dataset from the preprocessed entry
            if not datasetExist:
                print('Constructing full dataset...')
                corpusData = TextData.availableCorpus["cornell"](os.path.join(self.doc_dir,self.filename))
                self.createFullCorpus(corpusData.getConversations())
                self.saveDataset(self.fullSamplesPath)
            else:
                self.loadDataset(self.fullSamplesPath)
            self._printStats()

            print('Filtering words (vocabSize = {} and wordCount > {})...'.format(
                self.vocabularySize,
                self.filterVocab
            ))
            #self.filterFromFull()  # Extract the sub vocabulary for the given maxLength and filterVocab

            # Saving
            print('Saving dataset...')
            self.saveDataset(self.filteredSamplesPath)  # Saving tf samples
        else:
            self.loadDataset(self.filteredSamplesPath)

        assert self.padToken == 0

    def saveDataset(self, filename):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """

        with open(os.path.join(filename), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2id': self.word2id,
                'id2word': self.id2word,
                'idCount': self.idCount,
                'trainingSamples': self.trainingSamples
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.idCount = data.get('idCount', None)
            self.trainingSamples = data['trainingSamples']

            self.padToken = self.word2id['<pad>']
            self.goToken = self.word2id['<go>']
            self.eosToken = self.word2id['<eos>']
            self.unknownToken = self.word2id['<unknown>']  # Restore special words

    def createFullCorpus(self, conversations):
        """Extract all data from the given vocabulary.
        Save the data on disk. Note that the entire corpus is pre-processed
        without restriction on the sentence length or vocab size.
        """
        # Add standard tokens
        self.padToken = self.getWordId('<pad>')  # Padding (Warning: first things to add > id=0 !!)
        self.goToken = self.getWordId('<go>')  # Start of sequence
        self.eosToken = self.getWordId('<eos>')  # End of sequence
        self.unknownToken = self.getWordId('<unknown>')  # Word dropped from vocabulary

        # Preprocessing data

        for conversation in tqdm(conversations, desc='Extract conversations'):
            self.extractConversation(conversation)

        # The dataset will be saved in the same order it has been extracted

    def extractConversation(self, line_data):
        """Extract the sample lines from the conversations
        Args:
            conversation (Obj): a conversation object containing the lines to extract
        """

        # Iterate over all the lines of the conversation
        desc = line_data[-1]
        name = line_data[1]
        brand_name = line_data[4]
        category_name = line_data[3]
        desc  = self.extractText(desc)
        name = self.extractText(name)
        brand_name = self.extractText(brand_name)
        category_name = self.extractText(category_name)
        target_price = line_data[5]

        self.trainingSamples.append([desc, name,brand_name,category_name,target_price])

    def extractText(self, line):
        """Extract the words from a sample lines
        Args:
            line (str): a line containing the text to extract
        Return:
            list<list<int>>: the list of sentences of word ids of the sentence
        """
        sentences = []  # List[List[str]]
        # Extract sentences
        sentencesToken = nltk.sent_tokenize(line)

        # We add sentence by sentence until we reach the maximum length
        for i in range(len(sentencesToken)):
            tokens = nltk.word_tokenize(sentencesToken[i])

            tempWords = []
            for token in tokens:
                tempWords.append(self.getWordId(token))  # Create the vocabulary and the training sentences

            sentences.append(tempWords)

        return sentences

    def getWordId(self, word, create=True):

        word = word.lower()
        # At inference, we simply look up for the word
        if not create:
            wordId = self.word2id.get(word, self.unknownToken)
        # Get the id if the word already exist
        elif word in self.word2id:
            wordId = self.word2id[word]
            self.idCount[wordId] += 1
        # If not, we create a new entry
        else:
            wordId = len(self.word2id)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
            self.idCount[wordId] = 1

        return wordId
"""
data = TextData("train.tsv","data",100,"test_",800)
self.desc = []
self.other_elem = []
self.target = []
self.desc_length = []
self.other_elem_length = []
batches = data.getBatches(10)
print(batches[0].desc)
print(batches[0].other_elem)
print(batches[0].target)
print(batches[0].desc_length)
print(batches[0].other_elem_length)
"""