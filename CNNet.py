import tensorflow as tf
import numpy as np
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

relu = tf.nn.relu

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape,
              stddev=tf.sqrt(2./(shape[0]+shape[1])))
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W,pad='SAME', strides=1):
    """2D Convolutional operation. Default stride of 1"""
    return tf.nn.conv2d(x, W,
               strides=[1, strides, strides, 1], padding=pad)

def ReparamTrickBN(W, b, beta, gamma, mu, sigma2):
    W_rep = W * gamma / np.sqrt(sigma2)
    b_rep = (b - mu) * gamma / np.sqrt(sigma2) + beta
    return W_rep, b_rep


class CNNet(object):

    def __init__(self, name, lr = 0.001,
                 activationFunc = relu,
                 flag_bNorm=True,
                 loadInstance = None):
        
    
        # ---------- RESET GRAPH ----------
        tf.reset_default_graph()

        # ---------- OUTPUT FOLDERS ----------
        self.logsFolder = 'log_files/' # useful for tensorboard logs
        self.saveFolder = 'Models/'    # useful to restore the model

        # ---------- ATTRIBUTES ----------
        self.name = name
        self.lr = lr
        self.mean = None
        if(flag_bNorm):
            self.act = lambda net: activationFunc(self.batchNorm(net))
        else:
            self.act = activationFunc  
        

        # ---------- DATA PLACEHOLDERS ----------
        with tf.variable_scope('Input'):
            # tf Graph Input:  mnist data image of shape 28*28*1
            self.X = tf.placeholder(tf.float32, [None,784], name='X')
            # 0-9 digits recognition,  10 classes
            self.y = tf.placeholder(tf.float32, [None,10], name='y')
            # Dropout
            self.is_training = tf.placeholder(tf.bool, name='flag_train')

        # ---------- GRAPH DEFINITION ----------
        with tf.name_scope('Model'):
            self.CNN_var_init()
            self.logits = self.CNN_feedforward()
            self.prob = tf.nn.softmax(self.logits)

        with tf.name_scope('Loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.y))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, name="train_op")

        with tf.name_scope('Accuracy'):
            self.pred = tf.argmax(self.logits, axis=1, name="prediction")
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(self.pred, tf.argmax(self.y, axis=1)), tf.float32))

        # ---------- TRACK BATCH LOSS AND ACCURACY ----------
        s1 = tf.summary.scalar("loss", self.loss)
        s2 = tf.summary.scalar("acc", self.accuracy)
        self.summary_op = tf.summary.merge_all()
        
        # SESSION INITIALIZATION & RESTORE
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if (loadInstance):
            self.restore(loadInstance)
    
    def __del__(self):
        self.sess.close()
        
    # ----------------------------------------------------------
    # ----------------------INITIALIZATION----------------------
    # ----------------------------------------------------------


    def CNN_var_init(self):

        # Convolutional layer 1
        self.W_c1 = weight_variable([5,5,1,20], 'W_c1')   #[5,5,1,20]
        self.b_c1 = bias_variable([20], 'b_c1')#[20]
        
        # Fully connected layer 1
        self.W_fc1 = weight_variable([15680,1000], 'W_fc1') #[1000,10]
        self.b_fc1 = bias_variable([1000], 'b_fc1')#[10] 

        self.W_fc2 = weight_variable([1000,10], 'W_fc2') #[1000,10]
        self.b_fc2 = bias_variable([10], 'b_fc2')#[10]

    # ----------------------------------------------------------
    # ---------------------DATA PREPROCESSING-------------------
    # ----------------------------------------------------------

    def preproc(self, x, mean=None):
        """Preprocessing Input data by substracting mean"""

        # x = x*2 - 1.0
        # per-example mean subtraction (http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing)
        if(not(mean)):
            mean = tf.reduce_mean(x, axis=1, keepdims=True)
        return x - mean



    # ----------------------------------------------------------
    # ------------------------FEEDFORWARD-----------------------
    # ----------------------------------------------------------

    def batchNorm(self, net, decay=0.99):
        """Batch Normalization Layer from Tensorflow"""
        net = tf.contrib.layers.batch_norm(net,
            decay=decay,
            scale=True,
            updates_collections=None,
            is_training=self.is_training)
        return net

    def CNN_feedforward(self):    
        """Feedforward Architecture, classifying images"""
        
        from tensorflow.contrib.layers import flatten

        net = self.preproc(tf.reshape(self.X,[-1,28,28,1]))
        # Convolutional layer 1 & max pooling
        with tf.name_scope('Conv1'):
            net = conv2d(net, self.W_c1)+ self.b_c1
            net = self.act(net)

        # Flattening
        net = flatten(net)

        
        # Fully connected layer 1, sigmoid activation
        with tf.name_scope('FC1'):
            net = tf.matmul(net, self.W_fc1) + self.b_fc1
            net = self.act(net)
            
        # Fully connected layer 1, softmax activation
        with tf.name_scope('FC2'):
            net = tf.matmul(net, self.W_fc2) + self.b_fc2
        
        return net
    
    # ----------------------------------------------------------
    # -----------------------SAVE & RESTORE---------------------
    # ----------------------------------------------------------
    
    def save(self, path=None):
        """Save Tensorflow Model"""
        if(path is None):
            path = self.saveFolder+self.name
        save_path = self.saver.save(self.sess, path)
        print("INFO: TF Model saved in file: %s" % path)
        
    def restore(self, path=None):    
        """Restore TF model from the file"""
        if(path is None):
            path = self.saveFolder+self.name
        self.saver.restore(self.sess, save_path=path)



    # ----------------------------------------------------------
    # --------------------------LEARNING------------------------
    # ----------------------------------------------------------
        
    def train(self, batch_size=128, epochs=100,
              display_step=1, saveBest=True):
        """Train the model going through MNIST train dataset #epochs times"""
        print(self.name)
        print("* START TRAIN {l_r: %.4f; epochs: %d; batch: %d, TrAcc: %.1f %%, ValAcc: %.1f %%}"%\
          (self.lr, epochs, batch_size, 100*self.benchmark('TRAINING'),
           100*self.benchmark('VALIDATION')))


        # op to write logs to Tensorboard
        sum_wr = tf.summary.FileWriter(self.logsFolder, self.sess.graph)
        
        # Training cycle
        t0 = time.time()
        total_batch = mnist.train.num_examples // batch_size
        
        for epoch in range(epochs):
            avg_cost = 0.

            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c, summary = self.sess.run(
                    [self.train_op, self.loss, self.summary_op],
                     feed_dict={self.X: batch_xs, self.y: batch_ys,
                                self.is_training: True})
                sum_wr.add_summary(summary, epoch * total_batch + i)
                avg_cost += c / total_batch

            # Print training results per epoch   
            vl_acc = self.benchmark('VALIDATION')
            print("  [%.1f] Epoch: %02d | Loss=%.9f | ValAcc=%.3f"%(
                time.time()-t0, epoch+1, avg_cost, (vl_acc*100)))
                 
        
        # Evaluating model with the test accuracy
        print ("* END TRAIN in %.1f seconds => TestAcc: %.3f"%(
            time.time()-t0, 100*self.benchmark('TEST')))   
        


    # ----------------------------------------------------------
    # --------------------------TESTING-------------------------
    # ----------------------------------------------------------        
    
    def benchmark(self, dataset='TEST', batch_size = 1000):
        if    (dataset=='TRAINING'):
            benchmark_data = mnist.train
        elif  (dataset=='VALIDATION'):
            benchmark_data = mnist.validation
        else:#(dataset=='TEST'):
            benchmark_data = mnist.test
       
        N = benchmark_data.num_examples // batch_size
        total_acc = 0

        # op to write logs to Tensorboard
        sum_wr = tf.summary.FileWriter(self.logsFolder, self.sess.graph)
        for batch_i in range(N):
            xs, ys = benchmark_data.next_batch(batch_size, shuffle=False)
            step_acc, summ = self.sess.run([self.accuracy, self.summary_op],
              {self.X: xs, self.y: ys, self.is_training: False})
            total_acc+= step_acc
            sum_wr.add_summary(summ, batch_i)
               
        return total_acc / N
    
    def predict(self, image):
        return self.sess.run(self.pred,{self.X: image, self.is_training: False})