import tensorflow as tf
import numpy as np

class TGP(object):
    def highway_layer(self, x, size, bias_init):
        #variables
        carry_w = tf.get_variable("carry_w", [size, size])
        carry_b = tf.get_variable("carry_b", [size], initializer=tf.constant_initializer(bias_init))
        mlp_w = tf.get_variable("mlp_w", [size, size])
        mlp_b = tf.get_variable("mlp_b", [size], initializer=tf.constant_initializer())

        g = tf.sigmoid(tf.matmul(x, carry_w) + carry_b)

        return g*tf.nn.relu(tf.matmul(x, mlp_w) + mlp_b) + (1-g)*x

    def conv_maxpool_layer(self, x, vsize, esize, fwidth, fnum, maxlen):
        #embedding lookup
        emb = tf.get_variable("embedding", [vsize, esize], initializer=tf.random_uniform_initializer(-0.5/esize, 0.5/esize))
        inputs = tf.nn.embedding_lookup(emb, x)
        inputs = tf.expand_dims(inputs, -1)

        #apply convolutional filters on the characters
        filter_w = tf.get_variable("filter_w", [fwidth, esize, 1, fnum], \
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        filter_b = tf.get_variable("filter_b", [fnum], initializer=tf.constant_initializer())
        conv = tf.nn.conv2d(inputs, filter_w, strides=[1,1,1,1], padding="VALID")
        conv_activated = tf.nn.relu(tf.nn.bias_add(conv, filter_b))

        #max pooling over time steps
        h = tf.nn.max_pool(conv_activated, ksize=[1,(maxlen-fwidth+1),1,1], strides=[1,1,1,1], padding="VALID")
        h = tf.squeeze(h)

        return h

    def text_layer(self, x, vsize, esize, fnum, maxlen, pool_window, config):
        #variables
        emb = tf.get_variable("embedding", [vsize, esize], initializer=tf.random_uniform_initializer(-0.5/esize, 0.5/esize))
        filter_w = tf.get_variable("filter_w", [3, esize, 1, fnum], initializer=tf.truncated_normal_initializer(stddev=0.1))
        filter_b = tf.get_variable("filter_b", [fnum], initializer=tf.constant_initializer())
        attn_w = tf.get_variable("attn_w", [fnum, fnum])
        attn_v = tf.get_variable("attn_v", [fnum], initializer=tf.random_uniform_initializer())

        #constants
        zero_state = tf.zeros([config.batch_size, esize])

        #embedding lookup
        inputs = tf.nn.embedding_lookup(emb, x)
        inputs_rev = tf.reverse(inputs, [False, True, False])

        #transform sent input from [batch_size,sent_len,hidden_size] to [sent_len,batch_size,hidden_size]
        inputs_s = [tf.squeeze(input_, [1]) for input_ in tf.split(1, maxlen, inputs)]
        inputs_rev_s = [tf.squeeze(input_, [1]) for input_ in tf.split(1, maxlen, inputs_rev)]

        #run lstm and get hidden states
        with tf.variable_scope("lstm-forward"):
            lstm_fw = tf.nn.rnn_cell.BasicLSTMCell(esize, forget_bias=1.0)
            fw_outputs, _ = tf.nn.rnn(lstm_fw, inputs_s, \
                initial_state=lstm_fw.zero_state(config.batch_size, tf.float32))
            #insert zero state at the front and drop the last state [h_A, h_B, h_C] -> [0, h_A, h_B]
            fw_outputs.insert(0, zero_state)
            fw_outputs = fw_outputs[:-1]
        with tf.variable_scope("lstm-backward"):
            lstm_bw = tf.nn.rnn_cell.BasicLSTMCell(esize, forget_bias=1.0)
            bw_outputs, _ = tf.nn.rnn(lstm_bw, inputs_rev_s, \
                initial_state=lstm_bw.zero_state(config.batch_size, tf.float32))
            #reverse the time steps [j_C, j_B, j_A] -> [j_A, j_B, j_C]
            bw_outputs = tf.unpack(tf.reverse(tf.pack(bw_outputs), [True, False, False]))
            #insert zero state at the end and drop the first state [j_A, j_B, j_C] -> [j_B, j_C, 0]
            bw_outputs.append(zero_state)
            bw_outputs = bw_outputs[1:]

        #reshape outputs from [sent_len,batch_size,hidden_size] to [batch_size,sent_len,hidden_size]
        fw_outputs = tf.reshape(tf.concat(1, fw_outputs), [config.batch_size,-1,esize])
        bw_outputs = tf.reshape(tf.concat(1, bw_outputs), [config.batch_size,-1,esize])

        #concatenate the left right context and word embeddings [batch_size, sent_len, hidden_size*3]
        lrw_concat = tf.concat(2, [fw_outputs, inputs, bw_outputs])
        #reshape into [batch_size, sent_len*3, hidden_size, 1]
        lrw_concat = tf.reshape(lrw_concat, [config.batch_size, -1, esize, 1])

        #convolutional layer
        conv = tf.nn.conv2d(lrw_concat, filter_w, strides=[1,3,1,1], padding="VALID")
        conv_activated = tf.nn.relu(tf.nn.bias_add(conv, filter_b))

        #maxpool layer
        pooled = tf.nn.max_pool(conv_activated, ksize=[1,pool_window,1,1], strides=[1,1,1,1], padding="VALID")
        pooled = tf.reshape(pooled, [-1, fnum])

        #compute attention weights
        mm = tf.nn.tanh(tf.matmul(pooled, attn_w))
        inner = tf.reshape(tf.reduce_sum(mm*attn_v, 1), [config.batch_size,-1])
        attn = tf.nn.softmax(inner)

        #compute weighted sum given the attention weights
        h = tf.reduce_sum(tf.reshape(tf.reshape(attn, [-1,1])*pooled, [config.batch_size,-1,fnum]), 1)

        return h, attn

    def rbf_layer(self, x, psize):
        #variables
        mu_init = np.array(range(psize), dtype=np.float32) / psize 
        mu = tf.get_variable("mu", [psize], initializer=tf.constant_initializer(mu_init))
        sigma = tf.get_variable("sigma", [psize], initializer=tf.constant_initializer(np.sqrt(0.5/psize)))

        # computing output = exp( -(x - mu)^2 / (2*sigma^2) )
        h = tf.exp(tf.square(tf.tile(tf.reshape(x, [-1,1]),[1,psize]) - mu) / (-2*tf.square(sigma))) 

        return h

    def get_mu_sigma(self, scope_name):
        with tf.variable_scope(scope_name, reuse=True):
            return tf.get_variable("mu"), tf.get_variable("sigma")

    def __init__(self, is_training, vocab_size, num_steps, num_classes, num_timezones, loc_vsize, desc_vsize, \
        name_vsize, config):
        self.config = config
        m_in = config.text_filter_number + config.time_size + config.day_size + config.offset_size + \
            config.timezone_size + config.loc_filter_number + config.desc_filter_number + \
            config.name_filter_number + config.usertime_size

        ##############
        #placeholders#
        ##############
        self.x = tf.placeholder(tf.int32, [None, num_steps])
        self.y = tf.placeholder(tf.int32, [None])
        self.time = tf.placeholder(tf.float32, [None])
        self.day = tf.placeholder(tf.int32, [None])
        self.offset = tf.placeholder(tf.float32, [None])
        self.timezone = tf.placeholder(tf.int32, [None])
        self.loc = tf.placeholder(tf.int32, [None, config.loc_maxlen])
        self.desc = tf.placeholder(tf.int32, [None, config.desc_maxlen])
        self.name = tf.placeholder(tf.int32, [None, config.name_maxlen])
        self.usertime = tf.placeholder(tf.float32, [None])
        self.noise = tf.placeholder(tf.float32, [None, m_in])

        ##############
        #text network#
        ##############
        hiddens = []
        if config.text_filter_number > 0:
            with tf.variable_scope("text"):
                hidden_text, self.text_attn = self.text_layer(self.x, vocab_size, config.text_emb_size, \
                    config.text_filter_number, num_steps, config.text_pool_window, config)
                hiddens.append(hidden_text)

        ######################
        #time feature network#
        ######################
        if config.time_size > 0:
            with tf.variable_scope("time"):
                self.hidden_time = self.rbf_layer(self.time, config.time_size)
                hiddens.append(self.hidden_time)

        #####################
        #day feature network#
        #####################
        if config.day_size > 0:
            with tf.variable_scope("day"):
                self.day_embedding = tf.get_variable("embedding", [7, config.day_size], \
                    initializer=tf.random_uniform_initializer(-0.05/(config.day_size), 0.05/(config.day_size)))
                hidden_day = tf.nn.embedding_lookup(self.day_embedding, self.day)
                hiddens.append(hidden_day)

        ########################
        #offset feature network#
        ########################
        if config.offset_size > 0:
            with tf.variable_scope("offset"):
                self.hidden_offset = self.rbf_layer(self.offset, config.offset_size)
                hiddens.append(self.hidden_offset)

        ##########################
        #timezone feature network#
        ##########################
        if config.timezone_size > 0:
            with tf.variable_scope("timezone"):
                self.timezone_embedding = tf.get_variable("embedding", [num_timezones, config.timezone_size], \
                    initializer=tf.random_uniform_initializer(-0.05/(config.timezone_size), 0.05/(config.timezone_size)))
                hidden_timezone = tf.nn.embedding_lookup(self.timezone_embedding, self.timezone)
                hiddens.append(hidden_timezone)

        ##########################
        #location feature network#
        ##########################
        if config.loc_filter_number > 0:
            with tf.variable_scope("location"):
                hidden_loc = self.conv_maxpool_layer(self.loc, loc_vsize, config.loc_emb_size, \
                    config.loc_filter_width, config.loc_filter_number, config.loc_maxlen)
                hiddens.append(hidden_loc)

        #############################
        #description feature network#
        #############################
        if config.desc_filter_number > 0:
            with tf.variable_scope("description"):
                hidden_desc = self.conv_maxpool_layer(self.desc, desc_vsize, config.desc_emb_size, \
                    config.desc_filter_width, config.desc_filter_number, config.desc_maxlen)
                hiddens.append(hidden_desc)

        ######################
        #name feature network#
        ######################
        if config.name_filter_number > 0:
            with tf.variable_scope("name"):
                hidden_name = self.conv_maxpool_layer(self.name, name_vsize, config.name_emb_size, \
                    config.name_filter_width, config.name_filter_number, config.name_maxlen)
                hiddens.append(hidden_name)

        ###################################
        #user created time feature network#
        ###################################
        if config.usertime_size > 0:
            with tf.variable_scope("usertime"):
                self.hidden_usertime = self.rbf_layer(self.usertime, config.usertime_size)
                hiddens.append(self.hidden_usertime)

        ##################################
        #penultimate representation layer#
        ##################################
        #variables
        self.dense_w = tf.get_variable("dense_w", [m_in, config.rep_hidden_size])
        self.dense_b = tf.get_variable("dense_b", [config.rep_hidden_size], initializer=tf.constant_initializer())

        #concatenate all features
        hidden = tf.concat(1, hiddens)

        #add gaussian noise
        hidden = hidden + self.noise

        #dropout
        if is_training and config.keep_prob < 1.0:
           hidden = tf.nn.dropout(hidden, config.keep_prob, seed=config.seed)
    
        #penultimate representation
        self.rep = tf.nn.tanh(tf.matmul(hidden, self.dense_w) + self.dense_b)

        #######################
        #output layer and loss#
        #######################
        #variables
        self.softmax_w = tf.get_variable("softmax_w", [config.rep_hidden_size, num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [num_classes], initializer=tf.constant_initializer())

        #compute crossent and loss
        logits = tf.matmul(self.rep, self.softmax_w) + self.softmax_b
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.y)
        self.cost = tf.reduce_sum(crossent) / config.batch_size
        self.probs = tf.nn.softmax(logits)

        if not is_training:
            return
        else:
            #compute sparsity loss
            sparsity_loss = tf.reduce_mean(tf.abs((self.rep-1)*(self.rep+1)))
            self.cost += config.alpha*sparsity_loss

        #run optimiser and backpropagate gradients
        self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.cost)
