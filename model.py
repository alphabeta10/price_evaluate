import tensorflow as tf
import numpy as np
from tqdm import tqdm
from textdata import TextData
class BidirectionLSTM(object):
    def __init__(self,encoder_hidden_units,input_embedding_size,bath_size):
        self.textData = TextData("train.tsv", "data", 100, "test_", 800)
        self.vocab_size = self.textData.getVocabularySize()
        self.input_embedding_size = input_embedding_size
        self.encoder_hidden_units = encoder_hidden_units
        self.batch_size =bath_size
        self.buildNetwork()
    def buildNetwork(self):
        tf.reset_default_graph()
        with tf.name_scope("minibatch"):
            self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name="encoder_inputs")
            self.other_encoder = tf.placeholder(shape=(None, None), dtype=tf.int32, name="other_encoder_inputs")
            self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
            self.other_encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='other_encoder_inputs_length')

            self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.float32, name="decoder_targets")
        with tf.name_scope("embedding"):
            embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.input_embedding_size], -1.0, 1.0), dtype=tf.float32)
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
        other_encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings,self.other_encoder)

        other_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.encoder_hidden_units)
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.encoder_hidden_units)
        _,other_final_state = tf.nn.dynamic_rnn(cell=other_encoder_cell,inputs=other_encoder_inputs_embedded,
                                                sequence_length=self.other_encoder_inputs_length,time_major=True,dtype=tf.float32)
        ((_, _),
         (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                                                             cell_bw=encoder_cell,
                                                                                             inputs=encoder_inputs_embedded,
                                                                                             sequence_length=self.encoder_inputs_length,
                                                                                             dtype=tf.float32,
                                                                                             time_major=True)
        encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h,other_final_state.h), 1)
        fc_layer = tf.contrib.layers.fully_connected
        full_connect_units = 1024
        ouput_m = fc_layer(encoder_final_state_h,full_connect_units)
        ouput_m1 = fc_layer(ouput_m,512)
        self.final_output = fc_layer(ouput_m1, 1,activation_fn=None)
        self.cost = tf.reduce_sum(tf.pow(self.final_output-self.decoder_targets,2))/2
        self.error = tf.sqrt(tf.reduce_mean(tf.pow(tf.log(self.final_output+1)-tf.log(self.decoder_targets+1),2)))
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cost)

    def batchHandle(self,inputs, max_sequence_length=None):
        sequence_lengths = [len(seq) for seq in inputs]
        batch_size = len(inputs)
        if max_sequence_length is None:
            max_sequence_length = max(sequence_lengths)
        inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD
        for i, seq in enumerate(inputs):
            for j, element in enumerate(seq):
                inputs_batch_major[i, j] = element
        inputs_time_major = inputs_batch_major.swapaxes(0, 1)
        return inputs_time_major, sequence_lengths
    def train(self):
        batches = self.textData.getBatches(self.batch_size)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        batches_in_epoch = 100
        j = 0
        for i in range(100):
            for nextBatch in tqdm(batches,desc="batch_train"):
                feedDict = {}
                feedDict[self.encoder_inputs],feedDict[self.encoder_inputs_length] = self.batchHandle(nextBatch.desc,max(nextBatch.desc_length))
                feedDict[self.other_encoder],feedDict[self.other_encoder_inputs_length] = self.batchHandle(nextBatch.other_elem,max(nextBatch.other_elem_length))
                feedDict[self.decoder_targets] = [nextBatch.target]
                _, l,output_,error_ = sess.run([self.optimizer, self.cost,self.final_output,self.error], feedDict)
                j +=1
                if j == 0 or j % batches_in_epoch == 0:
                    print("batch {}".format(i))
                    print(" minibatch loss:{}".format(l))
                    print("error rate ",error_)
                    print()

model = BidirectionLSTM(256,256,100)
model.train()