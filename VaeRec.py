

from GCNdata import Data
import toolz
import numpy as np
import tensorflow as tf
from time import time
import argparse
import copy
from tqdm import tqdm 
from scipy.sparse import coo_matrix 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

num_length = 5
def parse_args(name,factor,Topk,feature,AggMethod,noise_num,KL,kl_co):
       
    parser = argparse.ArgumentParser(description="Run .")  
    parser.add_argument('--name', nargs='?', default= name )    
    parser.add_argument('--model', nargs='?', default='VAE_'+feature+str(AggMethod)+str(noise_num)+str(KL)+str(kl_co))
    parser.add_argument('--path', nargs='?', default='./datasets/processed/'+name,
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=name,
                        help='Choose a dataset.')
#    parser.add_argument('--epoch', type=int, default=100,#ciaoDVD 600 #Amazon_App 200 ML&dianping 100
#                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=factor,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default = 10e-5,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--Topk', type=int, default=10)
    parser.add_argument('--AggMethod', nargs='?', default= AggMethod ) 
    parser.add_argument('--noise_num', type=float, default= noise_num ) 
    parser.add_argument('--KL', nargs='?', default= KL )     
    parser.add_argument('--feature', nargs='?', default= feature )     
    parser.add_argument('--kl_co', type=float, default=kl_co)
    return parser.parse_args()

class PathCount(object):
    def __init__(self,args,data):
        self.args = args
        self.data = data
        self.user_side_entity = self.data.user_side_entity
        self.item_side_entity = self.data.item_side_entity          
        # Data loadin
        coo_UI = coo_matrix(data.matrix['user_item'])
        coo = dict()
        for key in self.item_side_entity:
            coo[key] = coo_matrix(data.matrix['item'+key])
        if self.args.feature == 'semantic':
            path = dict()
            path['IUI'] = (coo_UI.T).dot(coo_UI)
            path['MC'] = coo_matrix(self.data.markov)
            self.path_name = ['IUI','MC']
            self.PathCount = {}
            for key in self.item_side_entity:
                r_key = 'I'+key+'I'
                path[r_key] = (coo[key]).dot(coo[key].T)                
                self.path_name.append(r_key)
            # picture

            try:
                path['pic'] = np.matmul(self.data.pic,self.data.pic.T)
                self.path_name.append('pic')
            except:
                pass
            try:
                path['acou'] = np.matmul(self.data.acou,self.data.acou.T)
                self.path_name.append('acou')
            except:
                pass
            for key in self.path_name:
                keep = int(self.args.noise_num) #*len(data.train)/data.entity_num['user']
                try:
                    mat = path[key].toarray()
                except:
                    mat = path[key]
                self.PathCount[key] = self._keep_topk(mat,keep,True) 
        else:
            path = dict()
            path['I'] = coo_matrix(np.diag(np.ones(self.data.entity_num['item'])))
            self.path_name = ['I']
            self.PathCount = {'I':path['I'].toarray()}
            
        
    def _keep_topk(self, ui_connect, topK=10,ones=False):
        U, _ = ui_connect.shape
        res = np.zeros([U, _])
        for uid in range(U):
            u_ratings = ui_connect[uid].flatten()
            num = topK #int(len(u_ratings > 0)*topK)
            topk_bids = np.argpartition(-u_ratings, num).flatten()[:num]  # descending order
            topk_bids = [c for c in topk_bids if u_ratings[c] > 0]  # those less than 1000 non-zero entries, need to be removed zero ones
            for bid in topk_bids:
                if ones:
                    res[uid,bid] = 1.0;           
                else:
                    res[uid,bid] = (u_ratings[bid])
        return res      
class PVAE(object):
    def __init__(self,args,data,path,hidden_factor, learning_rate, lamda_bilinear, optimizer_type):
        # bind params to class
        # bind params to class
        self.args = args
        self.data = data
        self.path = path        
        self.path_name = self.path.path_name
        self.path_num =  len(self.path_name)
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.input_dim = self.n_item
        self.hidden_encoder_dim = hidden_factor
        self.hidden_decoder_dim = hidden_factor
        self.hidden_factor = hidden_factor
        self.learning_rate = learning_rate
        self.lam = lamda_bilinear
        self.optimizer_type = optimizer_type
        self.att_dim = 64
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []


        # init all variables in a tensorflow graph

        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        tf.Graph # 重置默认图       
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            self.counts_tf = self.counts2tf()

            # Input data.
            self.train_phaise = tf.placeholder(tf.float32,shape=[1,1])
            self.user_idxs = tf.placeholder(tf.int32, shape=[None] ) 
            self.item_pos = tf.placeholder(tf.int32, shape=[None] )  # None       
            self.item_neg = tf.placeholder(tf.int32, shape=[None,None] )  # None    
            self.recent = tf.placeholder(tf.int32, shape=[None, num_length])  # None * 2+ 5
            self.recentk = self.recent
            self.count_input = dict()
         
            for key in self.path_name:
                self.count_input[key] = tf.reduce_mean(tf.nn.embedding_lookup(self.counts_tf[key],self.recentk),axis=1)# [none ,item]


            # Variables. and count
            self.weights = self._initialize_weights() # k * k,  k
            self.user_prior = tf.nn.embedding_lookup(self.weights['user'],self.user_idxs) # none * k
            
            self.user_vae = []
            self.vae_loss = []

            self.p = []
            self.KL = []

            for key in self.path_name:
                p,KL,mu,sigma = self.encoder(self.count_input[key],self.user_prior) 
                self.p.append(p)
                self.KL.append(KL)

            self.KL = tf.reduce_sum(tf.stack(self.KL,axis=-1))

                
            self.user_vae = tf.stack(self.p,axis=1) # none * meta-path * k

#            
            if self.args.feature =='semantic':
                if self.args.AggMethod =='mean':
    #                self.user_emb = self.scaled_dot_product_attention(self.user_vae,self.user_vae,self.user_vae)
                    self.user_emb = tf.reduce_mean(self.user_vae,axis=1)  + self.user_prior# none * k
                    self.output = tf.matmul(self.user_emb,self.weights['item'],transpose_b=True)
    
    
                elif self.args.AggMethod == 'wgt':
                                    
                        
                    att_user = tf.nn.softmax(self.weights['wgt'],axis=1 ) #
                    self.user_emb = tf.reduce_sum(self.user_vae*att_user,axis=1) + self.user_prior# none * k
                    self.output = tf.matmul(self.user_emb,self.weights['item'],transpose_b=True)
    
                elif self.args.AggMethod == 'att':
                                    
    #                    att_user = tf.reduce_sum(self.user_vae * self.weights['path'],axis=-1,keep_dims=True)# none * path * 1
      
                        att_user = tf.reduce_sum(tf.stop_gradient(tf.nn.l2_normalize(self.user_vae,-1)) * tf.nn.l2_normalize(self.weights['path'],-1),axis=-1,keep_dims=True)# none * path * 1
    #                    self.att_user = tf.nn.softmax(self.weights['wgt'],axis=1 ) #tf.nn.softmax(self.att_user,axis=1)#none*path*1
                        att_user = tf.nn.softmax( att_user,axis=1 ) #tf.nn.softmax(self.att_user,axis=1)#none*path*1
    
                        self.user_emb = tf.reduce_sum(self.user_vae * att_user,axis=1) #+ self.user_prior # none * k
                        self.output = tf.matmul(self.user_emb,self.weights['item'],transpose_b=True)
                elif self.args.AggMethod == 'user_att':
                                    
                        
                        att_user = tf.matmul(tf.nn.l2_normalize(self.user_prior,-1) , tf.nn.l2_normalize(self.weights['path'][0],-1),transpose_b=True)# none * path * 1
                        att_user = tf.expand_dims(att_user,axis=-1)
                        att_user = tf.nn.softmax( att_user,axis=1 ) #tf.nn.softmax(self.att_user,axis=1)#none*path*1
    
                        self.user_emb = tf.reduce_sum(self.user_vae * att_user,axis=1) + self.user_prior # none * k
                        self.output = tf.matmul(self.user_emb,self.weights['item'],transpose_b=True)
            else:
                self.user_emb =  self.user_vae[:,0,:] +  self.user_prior
                self.output = tf.matmul(self.user_emb,self.weights['item'],transpose_b=True)
            
            self.postive_item = tf.batch_gather(self.output,tf.expand_dims(self.item_pos,axis=1)) # none
            self.negative_item = tf.batch_gather(self.output,self.item_neg) # none * none
            
            
#            self.loss_rec = tf.nn.l2_loss(tf.sigmoid(self.postive_item)-1) + tf.nn.l2_loss(tf.sigmoid(self.negative_item))
            self.loss_rec = self.pairwise_loss(self.postive_item -  tf.reduce_max(self.negative_item,axis=1))


            self.vae_loss = tf.reduce_sum(self.vae_loss)
            self.co_KL = 0
            for wgt in tf.trainable_variables():
                self.co_KL += self.lam*tf.nn.l2_loss(wgt)     
            if self.args.KL == 'KL':
                self.loss = self.loss_rec + self.args.kl_co*self.KL + self.co_KL
            else:
                self.loss = self.loss_rec + 0.0*self.KL + self.co_KL

#            self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(										
#            			logits=self.OUT, labels=self.label))          
#            
            
  
            
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer1 = tf.train.AdamOptimizer(self.learning_rate)
                grads = self.optimizer1.compute_gradients(self.loss)
                for i, (g, v) in enumerate(grads):
                    if g is not None:
                        grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
                self.optimizer = self.optimizer1.apply_gradients(grads)    
            # init
            self.sess = self._init_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

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
        W_encoder_hidden_mu = self.weight_variable([self.input_dim,self.hidden_factor])
#        user_att = tf.matmul(prior,W_encoder_hidden_mu,transpose_b = True) #none * input_dim
        
        
        b_encoder_hidden_mu = self.bias_variable([self.hidden_factor])       

#        l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)
          
        mu_encoder = tf.matmul(x, W_encoder_hidden_mu) + b_encoder_hidden_mu#none * k 均值
    
        W_encoder_hidden_logvar = self.weight_variable([self.input_dim,self.hidden_factor])
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
    def MF(self,x,user,query,type1 = 'user'):
        l2_loss = tf.constant(0.0)
        if type1 =='user':
            self.W = self.weight_variable([self.n_user,self.hidden_factor])
            hidden_encoder = tf.nn.embedding_lookup(self.W,user) #none * hidden_encoder_dim
        else:
            self.W = self.weight_variable([self.n_word,self.hidden_factor])   
            self.W = tf.concat(axis=0, values=[self.W, tf.zeros([1, self.hidden_factor])])

            hidden_encoder = tf.reduce_mean(tf.nn.embedding_lookup(self.W,query),axis=1) #none * hidden_encoder_dim

        H = self.weight_variable([self.n_item,self.hidden_factor])




        l2_loss += tf.nn.l2_loss(self.W)
        l2_loss += tf.nn.l2_loss(H)
        # Hidden layer encoder      
        self.x_hat = tf.matmul(hidden_encoder, H ,transpose_b=True)
        #loss
        self.BCE = tf.nn.l2_loss(self.x_hat-x)   
        loss = tf.reduce_mean(self.BCE)
        
        regularized_loss = loss + self.lam * l2_loss

        
        return hidden_encoder,regularized_loss


    def _init_session(self):
        # adaptively growing video memory
#        config = tf.ConfigProto()
#        config.gpu_options.allow_growth = True
#        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        return tf.Session()
    def counts2tf(self):
        counts_tf = dict()
        for key in self.path_name:
            counts_tf[key] = tf.constant(self.path.PathCount[key],dtype=tf.float32)
#        if self.args.noise_num > 0:
#            for key in self.path_name:
#                noise = np.zeros([self.n_user,self.n_item])
#                for i in range(self.n_user):
#                    noise[i][np.random.randint(0,self.n_item,self.args.noise_num)] = 1
#                counts_tf[key] = tf.constant(self.path.PathCount[key] + noise,dtype=tf.float32)
#                
        
        return counts_tf 

    def _initialize_weights(self):
        weights = dict()
        weights['user'] =  tf.Variable( np.random.normal(0.0, 0.01,[self.n_user ,self.att_dim]),dtype=tf.float32)
        weights['item'] =  tf.Variable( np.random.normal(0.0, 0.01,[self.n_item,self.hidden_factor]),dtype=tf.float32)
        weights['path'] =  tf.Variable( np.random.normal(0.0, 0.01,[1,self.path_num,self.att_dim]),dtype=tf.float32)
        weights['wgt'] =  tf.Variable( np.ones([1,self.path_num,1]),dtype=tf.float32)

        return  weights
    def weight_variable(self,shape):
      initial = tf.truncated_normal(shape, stddev=0.001)
      return tf.Variable(initial)
    
    def bias_variable(self,shape):
      initial = tf.constant(0., shape=shape)
      return tf.Variable(initial)
    
    def partial_fit(self, data):  # fit a batch       
        feed_dict = {self.user_idxs:data['user'],self.train_phaise:[[1]],self.recent:data['recent'],
                     self.item_pos:data['pos'],self.item_neg:data['neg']}

        loss,loss_rec,loss_VAE, opt = self.sess.run((self.loss_rec,self.KL,self.co_KL, self.optimizer), feed_dict=feed_dict)
        return [loss,loss_rec,loss_VAE]


    def topk(self,user_item_block,recent,Topk=500):
        users = user_item_block[:,0]
        feed_dict = {self.user_idxs:users,self.train_phaise:[[0]],self.recent:recent}
        self.f_result = self.sess.run(self.output,feed_dict)
 
        self.prediction = np.argsort(self.f_result)[:,::-1][:,:Topk]
  
        return self.prediction


class Train_MF(object):
    def __init__(self,args,data,path):
        self.args = args
        self.data = data
        self.path = path
        self.batch_size = args.batch_size
        self.epoch = data.epoch
        self.TopK = args.Topk
        self.entity = self.data.entity
        self.user_side_entity = self.data.user_side_entity
        self.item_side_entity = self.data.item_side_entity          
        # Data loadin
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.num_topk_user = 500

            
        print("DHRec: dataset=%s, factors=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, optimizer=%s"
              %(args.dataset, args.hidden_factor, self.epoch, args.batch_size, args.lr, args.lamda, args.optimizer))

    
#        self.item_attributes = self.collect_attributes()



    # Training\\\建立模型
        self.model = PVAE(self.args,self.data ,path,args.hidden_factor,args.lr, args.lamda, args.optimizer)
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
            NegSample = self.sample_negative(PosSample,5)#采样，none * NG
            for user_chunk in toolz.partition_all(self.batch_size,[i for i in range(len(PosSample))] ):                
                chunk = list(user_chunk)
                item_neg = np.array(NegSample[chunk],dtype = np.int)#none*5
                train_chunk_p5 = PosSample_with_p5[chunk]#none*2+5
                    
                self.feed_user = train_chunk_p5[:,0] # [none]
                self.feed_item_pos = train_chunk_p5[:,1]# [none]
                self.recent_intera =  train_chunk_p5[:,2:]# [none,10]
                self.feed_item_neg = item_neg# [none,5]
                #meta-path feature
                feed_dict = {'user':self.feed_user,'recent':self.recent_intera,
                             'pos':self.feed_item_pos,'neg':self.feed_item_neg}
                loss =  self.model.partial_fit(feed_dict)
#                loss =  [0,0,0]
            t2 = time()
         # evaluate training and validation datasets
            if epoch % int(self.epoch/10) == 0:
                print("Loss %.4f\t%.4f\t%.4f"%(loss[0],loss[1],loss[2]))
                with open("./final_result.txt","a") as f:
                    f.write("%s \t %s\t %s  \t ||||"%(self.args.dataset,self.args.model,epoch))
                for topk in [50]:
                    init_test_TopK_test = self.evaluate_TopK(self.data.test,topk) 
                    init_test_TopK_valid = 0,0,0
                    
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
 

        
    def sample_negative(self, data,num=10):
        samples = np.random.randint( 0,self.n_item,size = (len(data),num))
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

            self.score = self.model.topk(user_item_block,last_iteraction_block) #none * 50
            prediction = self.score 
            assert len(prediction) == len(user_item_block)
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
def VAE_run(name,factor,Topk,feature,AggMethod,noise_num,KL,kl_co):
#name,factor,Topk,seed = 'Grocery',64,10,0
#AggMethod,noise_num,KL = 'att',20,'KL'
#feature = 'semantic'
#kl_co = 1
#seed = 0
    args = parse_args(name,factor,Topk,feature,AggMethod,noise_num,KL,kl_co)
    data = Data(args,0)#获取数据
    path = PathCount(args,data)
    print(path.path_name)
    ##
    session_DHRec = Train_MF(args,data,path)
    session_DHRec.train()
#### 