# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:31:32 2018

"""

from GCNdata import Data
import toolz
import numpy as np
import tensorflow as tf
from time import time
import argparse
import copy
from tqdm import tqdm 
import scipy.sparse as sp



NUM = 1
def parse_args(name,factor,seed,batch_size):
        
    parser = argparse.ArgumentParser(description="Run .")  
    parser.add_argument('--name', nargs='?', default= name )    
    parser.add_argument('--model', nargs='?', default='CVAE')
    parser.add_argument('--path', nargs='?', default='./datasets/processed/'+name,
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=name,
                        help='Choose a dataset.')
#    parser.add_argument('--epoch', type=int, default=100,#ciaoDVD 600 #Amazon_App 200 ML&dianping 100
#                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=factor,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default = 10e-4,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--Topk', type=int, default=10)
    parser.add_argument('--seed', type=int, default=seed)  
    return parser.parse_args()

class FM(object):
    def __init__(self,args,data,hidden_factor, learning_rate, lamda_bilinear, optimizer_type):
        # bind params to class
        self.args = args
        # bind params to class
        self.data = data
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.lamda_bilinear = lamda_bilinear
        self.optimizer_type = optimizer_type
        self.n_attribute = len(self.data.item_side_entity)
        self.n_slot = self.n_attribute + 1
        # init all variables in a tensorflow graph
        np.random.seed(args.seed)
        self.att_dim = 64
        self.path_num = 5
        self.num_a = np.sum([self.data.entity_num[key] for key in self.data.item_side_entity])
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        tf.reset_default_graph()  # 重置默认图       
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            # Input data.
            self.weights = self._initialize_weights()
            self.feedback = tf.placeholder(tf.int32, shape=[None, 7])  # None * 2+ 5
            self.all_attributes = tf.placeholder(tf.int32, shape=[self.n_item,self.n_attribute,NUM])
            self.item_attribute = tf.reduce_mean(tf.nn.embedding_lookup(self.weights['attribute_embeddings'],self.all_attributes),axis=2)#[n_item,num_attri,k]
            self.visacou = []
            self.train_phaise = tf.placeholder(tf.float32,shape=[1,1])

            self.user_idxs = tf.placeholder(tf.int32, shape=[None] ) 
            self.item_pos = self.feedback[:,1]     
            self.item_neg = tf.placeholder(tf.int32, shape=[None,None] )  # None    


            try:
                visual =  tf.constant(self.data.vis,dtype=tf.float32)
                self.visacou.append(visual)
            except:
                pass 
            try:
                acoustic =  tf.constant(self.data.acou,dtype=tf.float32)
                self.visacou.append(acoustic)
            except:
                pass             
            if self.visacou != []:
                self.visacou = tf.stack(self.visacou,axis=1)#[n_item,1/2,k]
                try:
                    self.item_content = tf.concat([self.weights['item_embeddings2'],self.item_attribute,self.visacou],axis=1)#[n_item,num_attri,k]    
                except:
                    self.item_content = self.visacou#[n_item,num_attri,k]    
                    
            else:
                self.item_content = tf.concat([self.weights['item_embeddings2'],self.item_attribute],axis=1)


            self.users_idx = self.feedback[:,0]#none
            self.items_idx = self.feedback[:,1]#none
            self.item_sequence_idx = self.feedback[:,2:] # none * 5
            
            
            self.users_embeddings = tf.nn.embedding_lookup(self.weights['user_embeddings'],self.users_idx)#none * k
            self.item_sequence_content = tf.nn.embedding_lookup(self.item_content,self.item_sequence_idx)#none * 5 * num_attri * k
 
            self.item_sequence_content = tf.reduce_sum(self.item_sequence_content,axis=2)#none * 5 * k

            self.user_vae = []
            self.vae_loss = []

            self.p = []
            self.KL = []

            for i in range(5):
                p,KL,mu,sigma = self.encoder(self.item_sequence_content[:,i,:],0) 
                self.p.append(p)
                self.KL.append(KL)

            self.KL = tf.reduce_sum(tf.stack(self.KL,axis=-1))

                
            self.user_vae = tf.stack(self.p,axis=1) # none * meta-path * k

#            


            att_user = tf.reduce_sum(tf.stop_gradient(tf.nn.l2_normalize(self.user_vae,-1)) * tf.nn.l2_normalize(self.weights['path'],-1),axis=-1,keep_dims=True)# none * path * 1

            att_user = tf.nn.softmax( att_user,axis=1 ) #tf.nn.softmax(self.att_user,axis=1)#none*path*1

            self.user_emb = tf.reduce_sum(self.user_vae * att_user,axis=1) #+ self.user_prior # none * k
            self.output = tf.matmul(self.user_emb,self.weights['item'],transpose_b=True)

            self.postive_item = tf.batch_gather(self.output,tf.expand_dims(self.item_pos,axis=1)) # none
            self.negative_item = tf.batch_gather(self.output,self.item_neg) # none * none
            
            
#            self.loss_rec = tf.nn.l2_loss(tf.sigmoid(self.postive_item)-1) + tf.nn.l2_loss(tf.sigmoid(self.negative_item))
            self.loss_rec = self.pairwise_loss(self.postive_item -  tf.reduce_max(self.negative_item,axis=1))


            
            self.vae_loss = tf.reduce_sum(self.vae_loss)
            self.co_KL = 0
            for wgt in tf.trainable_variables():
                self.co_KL += self.lamda_bilinear*tf.nn.l2_loss(wgt)     

            self.loss = self.loss_rec + self.KL + self.co_KL



            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                            
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            
            out = self.output
            self.out_all_topk = tf.nn.top_k(out,1000)

            # init
            self.sess = self._init_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True
        return tf.Session(config=config)
#    def (self,):
    def jieduan(self,x):
        return tf.layers.dense(tf.stop_gradient(x),self.hidden_factor)#6.	
    def scaled_dot_product_attention(self, queries, keys, values):
        self.num_heads = 1
        batch_size, num_queries, sequence_length = tf.shape(queries)[0], tf.shape(queries)[1], tf.shape(values)[1]
        Q, K, V = self.jieduan(queries), self.jieduan(keys), self.jieduan(values)
        Q = tf.transpose(tf.reshape(Q, [batch_size, num_queries, self.num_heads, self.hidden_factor]), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, [batch_size, sequence_length, self.num_heads, self.hidden_factor]), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, [batch_size, sequence_length, self.num_heads, self.hidden_factor]), [0, 2, 1, 3])
        S = tf.matmul(tf.nn.softmax(tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / tf.sqrt(float(self.hidden_factor))), V)
        S = tf.reshape(tf.transpose(S, [0, 2, 1, 3]), [batch_size, num_queries, self.num_heads * self.hidden_factor])
        return S#tf.keras.layers.LayerNormalization(axis=-1)(S)#tf.stop_gradient(S)* tf.Variable(np.ones([1,1,self.hidden_factor]),dtype=tf.float32)
    def pairwise_loss(self,inputx):
#        input none*1
#        label none*1
        hinge_pair = tf.maximum(tf.minimum(inputx,10),-10)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(hinge_pair)))
        return loss          
    
    def encoder(self,x,prior):
#        l2_loss = tf.constant(0.0)
#        x = x / tf.reduce_sum(x,axis=1,keep_dims=True)
        W_encoder_hidden_mu = self.weight_variable([self.hidden_factor,self.hidden_factor])
#        user_att = tf.matmul(prior,W_encoder_hidden_mu,transpose_b = True) #none * input_dim
        
        
        b_encoder_hidden_mu = self.bias_variable([self.hidden_factor])       

#        l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)
          
        mu_encoder = tf.matmul(x, W_encoder_hidden_mu) + b_encoder_hidden_mu#none * k 均值
    
        W_encoder_hidden_logvar = self.weight_variable([self.hidden_factor,self.hidden_factor])
        b_encoder_hidden_logvar = self.bias_variable([self.hidden_factor])
#        l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)
        
        # Sigma encoder
        logvar_encoder = tf.matmul(x, W_encoder_hidden_logvar) + b_encoder_hidden_logvar#none * k 方差
        
        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon') * self.train_phaise# 0 for test, 1 for train
        
        # Sample latent variable
        std_encoder = tf.exp(0.5 * logvar_encoder)
        self.z_s = mu_encoder + tf.multiply(std_encoder, epsilon) #none*k

        KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder , 2) - tf.exp(logvar_encoder))                

        return self.z_s,KLD,mu_encoder,logvar_encoder
        
    def _initialize_weights(self):
        all_weights = dict()
        
        all_weights['user_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_user, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['item'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['item_embeddings2'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_item,1, self.hidden_factor]),dtype = tf.float32) # features_M * K
        with tf.variable_scope('attributes'):
#            all_weights['attributes_att'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_attribute, self.hidden_factor]),dtype = tf.float32) # features_M * K

            all_weights['attribute_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.num_a, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['path'] =  tf.Variable( np.random.normal(0.0, 0.01,[1,self.path_num,self.att_dim]),dtype=tf.float32)
        all_weights['wgt'] =  tf.Variable( np.ones([1,self.path_num,1]),dtype=tf.float32)

        return all_weights


    def partial_fit(self, data):  # fit a batch
        
        feed_dict = {self.item_neg:np.random.randint(0,self.n_item,[len(data['feedback']),5]),self.feedback: data['feedback'],self.all_attributes:data['all_attributes'],self.train_phaise:[[1]]}
        loss_rec, opt = self.sess.run((self.loss_rec, self.optimizer), feed_dict=feed_dict)

        
    def topk(self,user_item_feedback,all_attributes):
        
        feed_dict = {self.feedback: user_item_feedback,self.all_attributes:all_attributes,self.train_phaise:[[0]]}
        _, self.prediction = self.sess.run(self.out_all_topk,feed_dict)     
        return self.prediction
    def weight_variable(self,shape):
      initial = tf.truncated_normal(shape, stddev=0.001)
      return tf.Variable(initial)
    
    def bias_variable(self,shape):
      initial = tf.constant(0., shape=shape)
      return tf.Variable(initial)

class Train_MF(object):
    def __init__(self,args,data):
        self.args = args
        self.data = data
        self.batch_size = args.batch_size
        self.epoch = data.epoch *2
        self.TopK = args.Topk
        self.entity = self.data.entity
        self.user_side_entity = self.data.user_side_entity
        self.item_side_entity = self.data.item_side_entity          
        # Data loadin
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.num_topk_user = 500


    
        self.item_attributes = self.collect_attributes()
    # Training\\\建立模型
        self.model = FM(self.args,self.data ,args.hidden_factor,args.lr, args.lamda, args.optimizer)
    def train(self):  # fit a dataset
        # Check Init performance
        #初始结果   
        MAP_valid = 0
        PosSample= np.array(self.data.train) # Array形式的二元组（user,item），none * 2
        PosSample_with_p5 = np.concatenate([PosSample,np.array([\
                    self.data.latest_interaction[(line[0],line[1])] for line in PosSample])],axis =1)#none*2+5
        for epoch in tqdm(range(0,self.epoch+1)): #每一次迭代训练
            np.random.shuffle(PosSample)
            #sample负样本采样

            for user_chunk in toolz.partition_all(self.batch_size,[i for i in range(len(PosSample))] ):                
                chunk = list(user_chunk)
                feedback = PosSample_with_p5[chunk]#none*2+5
                self.feed_dict = {'feedback':feedback,'all_attributes':self.item_attributes}
                loss =  self.model.partial_fit(self.feed_dict)
#                loss =  [0,0,0]
            t2 = time()
         # evaluate training and validation datasets
            if epoch % int(self.epoch/10) == 0:
                print(loss)
                with open("./final_result.txt","a") as f:
                    f.write("%s \t %s\t %s  \t ||||"%(self.args.dataset,self.args.model,epoch))
                for topk in [50]:
                    init_test_TopK_test = self.evaluate_TopK(self.data.test,topk) 
                    init_test_TopK_valid =0,0,0
                    print("Epoch %d Top%d \t TEST SET:%.4f MAP:%.4f,NDCG:%.4f,PREC:%.4f;[%.1f s]\n"
                      %(epoch,topk,init_test_TopK_valid[2],init_test_TopK_test[0],init_test_TopK_test[1],init_test_TopK_test[2], time()-t2))
                    with open("./final_result.txt","a") as f:
                        f.write("Top-%d \t %.4f TEST SET,%.4f,%.4f,%.4f\t||||"
                          %(topk,init_test_TopK_valid[2],init_test_TopK_test[0],init_test_TopK_test[1],init_test_TopK_test[2]))                
                with open("./final_result.txt","a") as f:
                    f.write("\n")
            if MAP_valid < np.sum(init_test_TopK_test):
                MAP_valid = np.sum(init_test_TopK_test)
                result_print = init_test_TopK_test
        with open("./result.txt","a") as f:
            f.write("%s,%s,%.4f,%.4f,%.4f\n"%(self.args.name,self.args.model,result_print[0],result_print[1],result_print[2]))
 
    
    def collect_attributes(self):
        #return item * 
        attributes = []
        start_index = 0
        for entity in self.data.item_side_entity:
            key = 'item_' + entity
            attribute_item = self.data.dict_forward[key]
            attribute = []
            for item in range(self.n_item):
                list_ = attribute_item[item]
                if len(list_) <=NUM:
                    attribute.append(list_+[-1 for i in range(NUM-len(list_))])
                else:
                    attribute.append(list_[:NUM])
            attribute = np.array(attribute)
            attributes.append(attribute + start_index)
            start_index = start_index + self.data.entity_num[entity]
        return np.stack(attributes,axis=1)

    
        
    def sample_negative(self, data,num=10):
        samples = np.random.randint( 0,self.n_item,size = (len(data)))

        return samples


    def evaluate_TopK(self,test,topk):
        test_candidate = copy.deepcopy(np.array(test))#none * 2
        size = len(test_candidate)
        result_MAP = []
        result_PREC = []
        result_NDCG = []
        num = self.num_topk_user
        #meta-path feature
        last_iteraction = [] #none*5
        for line in test_candidate:
            #meta-path特征
            user,item = line
            last_iteraction.append(self.data.latest_interaction[(user,item)])
            
        last_iteraction = np.array(last_iteraction)
        for _ in range(int(size/num+1)):
            user_item_block = test_candidate[_*num:(_+1)*num]
            last_iteraction_block = last_iteraction[_*num:(_+1)*num]
            feedback_block = np.concatenate((user_item_block,last_iteraction_block),axis=1)

            self.score = self.model.topk(feedback_block,self.item_attributes) #none * 50
            prediction = self.score 
            assert len(prediction) == len(feedback_block)
            for i,line in enumerate(user_item_block):
                user,item = line
                n = 0 
                for it in prediction[i]:
                    if n> topk -1:
                        result_MAP.append(0.0)
                        result_NDCG.append(0.0)
                        result_PREC.append(0.0)  
                        n=0
                        break
                    elif it == item:   
                        result_MAP.append(1.0)
                        result_NDCG.append(np.log(2)/np.log(n+2))
                        result_PREC.append(1/(n+1))
                        n=0
                        break
                    elif it in self.data.set_forward['train'][user]:
                        continue
                    else:
                        n = n + 1   
        return  [np.mean(result_MAP),np.mean(result_NDCG),np.mean(result_PREC)] 
def cVAE_main(name,factor,seed,batch_size):    
#name,factor,Topk,seed ,batch_size = 'ML',64,10,0,2048
    args = parse_args(name,factor,seed,batch_size)
    data = Data(args,seed)#获取数据
    session_DHRec = Train_MF(args,data)
    session_DHRec.train()
        # 