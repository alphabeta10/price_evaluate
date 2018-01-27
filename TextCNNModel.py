import csv
#import gensim
import tensorflow as tf
from tqdm import tqdm
import nltk
import numpy as np
import pickle
import tensorflow.contrib.slim as slim
import os
def get_words(desc):
    sum = 0
    tokens = nltk.sent_tokenize(desc)
    for token in tokens:
        words = nltk.word_tokenize(token)
        sum +=len(words)
    return sum
def read_data(fname):
    with open(fname,'r',encoding='utf-8-sig') as f:
        p = csv.reader(f,delimiter="\t")
        index = 0
        datas = []
        for i,r in tqdm(enumerate(p),desc="iteratordata"):
            if index==0:
                print(r)
                index+=1
            else:
                elem = r[-1].strip().replace("\n","").replace(" [rm]","")
                datas.append([elem,r[5]])


tf.reset_default_graph()
weights = {
            'wc1':tf.Variable(tf.truncated_normal([3,300,1,256],stddev=0.0001)),
            'wc2':tf.Variable(tf.truncated_normal([3,300,1,256],stddev=0.0001)),
            'wc3':tf.Variable(tf.truncated_normal([3,300,1,256],stddev=0.0001)),
            'wc4':tf.Variable(tf.truncated_normal([3,300,1,256],stddev=0.0001)),
            'wc5':tf.Variable(tf.truncated_normal([3,300,1,256],stddev=0.0001)),
            'wc6':tf.Variable(tf.truncated_normal([3,300,1,256],stddev=0.0001)),
            'fu1':tf.Variable(tf.truncated_normal([150*50*128,1024],stddev=0.0001)),
            'out':tf.Variable(tf.truncated_normal([1024,1],stddev=0.0001))
        }
biases = {
            'bc1':tf.Variable(tf.truncated_normal([256],stddev=0.0001)),
            'bc2':tf.Variable(tf.truncated_normal([256],stddev=0.0001)),
            'bc3':tf.Variable(tf.truncated_normal([256],stddev=0.0001)),
            'bc4':tf.Variable(tf.truncated_normal([256],stddev=0.0001)),
            'bc5':tf.Variable(tf.truncated_normal([256],stddev=0.0001)),
            'bc6':tf.Variable(tf.truncated_normal([256],stddev=0.0001)),
            'bd1':tf.Variable(tf.truncated_normal([1024],stddev=0.0001)),
            'bout':tf.Variable(tf.truncated_normal([1],stddev=0.0001)),
        }
def conv2d(name, x, W, b, strides=1, padding='VALID'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)
def maxpool2d(name, x, k=2, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, k, 1], padding=padding, name=name)

def norm(name, x, l):
    return tf.nn.lrn(x, l, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
def construct_net(inputs,conv_name,pool_name,weigt_name,biase_name,input_len):
        net = conv2d(conv_name,inputs,weights[weigt_name],biases[biase_name],padding="VALID")
        net = maxpool2d(pool_name,net,input_len-3,padding="VALID")
        return net
def buildNet(name,condition,category_name,brand_name,on_or_line,desc,name_len=17,condition_len=5,category_len=9,brand_len=8,on_or_line_len=5,desc_len=200):
    net0 = construct_net(name,"conv0","pool0",'wc1','bc1',name_len)
    net0 = tf.reshape(net0, shape=[-1, 1 * 1 * 256])
    net1 = construct_net(condition,"conv1","pool1",'wc2','bc2',condition_len)
    net1 = tf.reshape(net1, shape=[-1, 1 * 1 * 256])
    net2 = construct_net(category_name,"conv2","pool2",'wc3','bc3',category_len)
    net2 = tf.reshape(net2, shape=[-1, 1 * 1 * 256])
    net3 = construct_net(brand_name,"conv3","pool3",'wc4','bc4',brand_len)
    net3 = tf.reshape(net3, shape=[-1, 1 * 1 * 256])
    net4 = construct_net(on_or_line,"conv4","pool4",'wc5','bc5',on_or_line_len)
    net4 = tf.reshape(net4, shape=[-1, 1 * 1 * 256])
    net5 = construct_net(desc,"conv5","pool5",'wc6','bc6',desc_len)
    net5 = tf.reshape(net5, shape=[-1, 1 * 1 * 256])
    net = tf.concat([net0,net1,net2,net3,net4,net5],axis=1,name="concat")
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 4096, scope='fc9')
        net = slim.fully_connected(net, 4096, scope='fc10')
        net = slim.fully_connected(net, 1, activation_fn=None,scope='fc11')
        net = tf.reshape(net, shape=(-1, 1), name="reshape2")
    return net

def train_model(batches):

    name = tf.placeholder(dtype=tf.float32, shape=(None, None, None), name='input_name')
    condition = tf.placeholder(dtype=tf.float32, shape=(None, None, None), name='input_condition')
    category_name = tf.placeholder(dtype=tf.float32, shape=(None, None, None), name='input_categroy_name')
    brand_name = tf.placeholder(dtype=tf.float32, shape=(None, None, None), name='input_brand_name')
    on_or_line = tf.placeholder(dtype=tf.float32, shape=(None, None, None), name='input_on_or_line')
    desc = tf.placeholder(dtype=tf.float32, shape=(None, None, None), name='input_desc')
    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.float32, name="decoder_targets")
    name_len = 17
    condition_len = 5
    category_len = 9
    brand_len = 8
    on_or_line_len = 5
    desc_len = 200
    name_reshape = tf.reshape(name, shape=[-1, name_len, 300,1])
    condition_reshape = tf.reshape(condition, shape=[-1, condition_len, 300,1])
    category_name_reshape = tf.reshape(category_name, shape=[-1, category_len, 300,1])
    brand_name_reshape = tf.reshape(brand_name, shape=[-1, brand_len, 300,1])
    on_or_line_reshape = tf.reshape(on_or_line, shape=[-1, on_or_line_len, 300,1])
    desc_reshape = tf.reshape(desc, shape=[-1, desc_len, 300,1])
    out = buildNet(name_reshape,condition_reshape,category_name_reshape,brand_name_reshape,on_or_line_reshape,desc_reshape)
    cost = tf.reduce_mean(tf.pow(out - decoder_targets, 2))
    error = tf.sqrt(tf.reduce_mean(tf.pow(tf.log(out + 1) - tf.log(decoder_targets + 1), 2)))
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = 0.001
    params = tf.trainable_variables()
    opt = tf.train.AdadeltaOptimizer(
        learning_rate, epsilon=1e-6)
    gradients = tf.gradients(cost, params)
    clipped_gradients, norm_ = \
        tf.clip_by_global_norm(gradients, 1.0)
    updates = opt.apply_gradients(
        zip(clipped_gradients, params),
        global_step=global_step)
    with open("doc2vector.pickle", 'rb') as handle:
        data = pickle.load(handle)
        word2vector = data['word2vector']
    def words2array(words, size,word_len):
        array = []
        for word in words:
            if word in word2vector:
                array.append(word2vector[word])
        if len(array) > size:
            return np.array(array[:size]).reshape(size, word_len)
        else:
            x = [[0] * word_len] * (size - len(array))
            array.extend(x)
            return np.array(array).reshape(size, word_len)
    def conver_batch(b,name_len=17,condition_len=5,category_len=9,brand_len=8,on_or_line_len=5,desc_len=200,word_len=300):
        name = np.zeros((len(b), name_len, 300))
        condition = np.zeros((len(b), condition_len, 300))
        category_name = np.zeros((len(b), category_len, 300))
        brand_name = np.zeros((len(b), brand_len, 300))
        on_or_line = np.zeros((len(b), on_or_line_len, 300))
        desc = np.zeros((len(b), desc_len, 300))
        labels = np.zeros((len(b), 1))
        for k in range(len(b)):
            labels[k] = b[k][1]
            all = b[k][0]
            name[k,:,:] = words2array(all[0],name_len,word_len)
            condition[k,:,:] = words2array(all[1],condition_len,word_len)
            category_name[k,:,:] = words2array(all[2],category_len,word_len)
            brand_name[k,:,:] = words2array(all[3],brand_len,word_len)
            on_or_line[k,:,:] = words2array(all[4],on_or_line_len,word_len)
            desc[k,:,:] = words2array(all[5],desc_len,word_len)
        return name,condition,category_name,brand_name,on_or_line,desc,labels
    saver = tf.train.Saver(max_to_keep=3)
    check_dir = "./ckpt_dir4"
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    init = tf.global_variables_initializer()
    j = 0
    batches_in_epoch = 100
    sess = tf.InteractiveSession()
    try:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(check_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("restort session")
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(30000):
            for bathe in tqdm(batches,desc="batch_data"):#cost,optimizer,out
                name_, condition_, category_name_, brand_name_, on_or_line_, desc_, labels_ = conver_batch(bathe)
                cost_, error_,_,out_,= sess.run([cost,error,updates,out], feed_dict={name:name_,condition:condition_,category_name:category_name_,
                                                                                     brand_name:brand_name_,on_or_line:on_or_line_,desc:desc_,decoder_targets:labels_})
                if j == 0 or j % batches_in_epoch == 0:
                    print("batch {}".format(i))
                    print("ouput is ", out_)
                    print("ouput shape is ", out_.shape)
                    print("output shape is ", cost_.shape)
                    print("cost is ",cost_)
                    print("errror ",error_)
                    print("data_target_",labels_)
                    saver.save(sess, check_dir + "/model.ckpt", global_step=j)
                    """
                    print(np.array(dout_).shape)
                    #print(np.array(net_).shape)
                    print("dout \n",dout_)
                    #print("net \n",net_)
                    """
                j += 1
    except (KeyboardInterrupt, SyntaxError):
        print("save model")
        saver.save(sess, check_dir + "/model.ckpt", global_step=j)


