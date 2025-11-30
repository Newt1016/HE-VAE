
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
    parser = argparse.ArgumentParser(description="Run Mixture-of-Experts VAE.")
    parser.add_argument('--name', nargs='?', default=name)

    parser.add_argument('--model', nargs='?', default='MoE_VAE_'+feature+str(AggMethod)+str(noise_num)+str(KL)+str(kl_co))
    parser.add_argument('--path', nargs='?', default='./datasets/processed/'+name,
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=name,
                        help='Choose a dataset.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=factor,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default = 1e-5,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type.')
    parser.add_argument('--Topk', type=int, default=10)
    parser.add_argument('--AggMethod', nargs='?', default=AggMethod)
    parser.add_argument('--noise_num', type=float, default=noise_num)
    parser.add_argument('--KL', nargs='?', default=KL)
    parser.add_argument('--feature', nargs='?', default=feature)
    parser.add_argument('--kl_co', type=float, default=kl_co)
    return parser.parse_args()

class PathCount(object):

    def __init__(self,args,data):
        self.args = args
        self.data = data
        self.user_side_entity = self.data.user_side_entity
        self.item_side_entity = self.data.item_side_entity
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
                keep = int(self.args.noise_num)
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
            num = topK
            topk_bids = np.argpartition(-u_ratings, num).flatten()[:num]
            topk_bids = [c for c in topk_bids if u_ratings[c] > 0]
            for bid in topk_bids:
                if ones:
                    res[uid,bid] = 1.0;
                else:
                    res[uid,bid] = (u_ratings[bid])
        return res

class PVAE_MoE(object):
    def __init__(self,args,data,path,hidden_factor, learning_rate, lamda_bilinear, optimizer_type):
        self.args = args
        self.data = data
        self.path = path
        self.path_name = self.path.path_name
        self.path_num = len(self.path_name)
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
 
        self.expert_input_dim = self.n_item
        self.hidden_factor = hidden_factor
        self.learning_rate = learning_rate
        self.lam = lamda_bilinear
        self.optimizer_type = optimizer_type
        self.att_dim = 64
        self._init_graph()

    def _init_graph(self):
        tf.Graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.counts_tf = self.counts2tf()

            # Input data
            self.train_phaise = tf.placeholder(tf.float32, shape=[1, 1])
            self.user_idxs = tf.placeholder(tf.int32, shape=[None])
            self.item_pos = tf.placeholder(tf.int32, shape=[None])
            self.item_neg = tf.placeholder(tf.int32, shape=[None, None])
            self.recent = tf.placeholder(tf.int32, shape=[None, num_length])
            
            # Variables
            self.weights = self._initialize_weights()

            # --- 多专家编码器 (Mixture-of-Experts Encoder) ---
            self.expert_mus = []
            self.expert_logvars = []

            # 为每个元路径（模态）创建一个专家编码器
            for key in self.path_name:
                # 1. 准备每个专家的输入
                expert_input = tf.reduce_mean(tf.nn.embedding_lookup(self.counts_tf[key], self.recent), axis=1) # [batch_size, n_item]
                
                # 2. 调用专家编码器，为每个专家创建独立的权重
                with tf.variable_scope("expert_" + key):
                    mu_expert, logvar_expert = self.expert_encoder(expert_input)
                    self.expert_mus.append(mu_expert)
                    self.expert_logvars.append(logvar_expert)

            # --- 专家乘积融合 (Product-of-Experts Fusion) ---
            # PoE假设先验为N(0,I)。融合后的分布的均值和方差计算如下：
            # 融合分布的方差的倒数 = sum(每个专家分布的方差的倒数) + 先验分布的方差的倒数(1)
            # 融合分布的均值 = 融合分布的方差 * sum(每个专家分布的均值 / 每个专家分布的方差)
            
            # 计算每个专家的精度（方差的倒数）
            precisions = [tf.exp(-logvar) for logvar in self.expert_logvars]
            
            # 融合后的精度 (加上先验 N(0,I) 的精度, I的逆是I)
            combined_precision = 1.0 + tf.reduce_sum(precisions, axis=0)
            

            combined_variance = 1.0 / combined_precision
            self.combined_logvar = tf.log(combined_variance)

            mu_precision_sum = tf.reduce_sum([mu * prec for mu, prec in zip(self.expert_mus, precisions)], axis=0)
            self.combined_mu = combined_variance * mu_precision_sum

     
            epsilon = tf.random_normal(tf.shape(self.combined_logvar), name='epsilon') * self.train_phaise
            std_encoder = tf.exp(0.5 * self.combined_logvar)
            self.z_s = self.combined_mu + tf.multiply(std_encoder, epsilon)

 
            self.user_emb = self.z_s
            self.output = tf.matmul(self.user_emb, self.weights['item'], transpose_b=True)

            self.postive_item = tf.batch_gather(self.output, tf.expand_dims(self.item_pos, axis=1))
            self.negative_item = tf.batch_gather(self.output, self.item_neg)
            
            self.loss_rec = self.pairwise_loss(self.postive_item - tf.reduce_max(self.negative_item, axis=1))

            # KL散度现在基于融合后的分布与标准正态先验N(0,I)计算
            self.KL_divergence = -0.5 * tf.reduce_sum(1 + self.combined_logvar - tf.pow(self.combined_mu, 2) - tf.exp(self.combined_logvar), axis=1)
            self.KL_divergence = tf.reduce_mean(self.KL_divergence)

            self.co_KL = 0
            for key in self.weights.keys():
                self.co_KL += self.lam * tf.nn.l2_loss(self.weights[key])

            if self.args.KL == 'KL':
                self.loss = self.loss_rec + self.args.kl_co * self.KL_divergence + self.co_KL
            else:
                self.loss = self.loss_rec + self.co_KL

            # Optimizer
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            
            # Init session
            self.sess = self._init_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def expert_encoder(self, x):
        """
        一个独立的专家编码器网络。
        输入一个模态的特征，输出该模态的潜在分布参数。
        """
        # 编码器的隐藏层
        W_hidden = self.weight_variable([self.expert_input_dim, self.hidden_factor * 2])
        b_hidden = self.bias_variable([self.hidden_factor * 2])
        hidden = tf.nn.relu(tf.matmul(x, W_hidden) + b_hidden)

        # 输出均值 mu
        W_mu = self.weight_variable([self.hidden_factor * 2, self.hidden_factor])
        b_mu = self.bias_variable([self.hidden_factor])
        mu_encoder = tf.matmul(hidden, W_mu) + b_mu

        # 输出log方差 logvar
        W_logvar = self.weight_variable([self.hidden_factor * 2, self.hidden_factor])
        b_logvar = self.bias_variable([self.hidden_factor])
        logvar_encoder = tf.matmul(hidden, W_logvar) + b_logvar
        
        return mu_encoder, logvar_encoder

    def pairwise_loss(self, inputx):
        hinge_pair = tf.maximum(tf.minimum(inputx, 10), -10)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(hinge_pair)))
        return loss

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def counts2tf(self):
        counts_tf = dict()
        for key in self.path_name:
            counts_tf[key] = tf.constant(self.path.PathCount[key], dtype=tf.float32)
        return counts_tf

    def _initialize_weights(self):
        weights = dict()

        weights['user'] = tf.Variable(np.random.normal(0.0, 0.01, [self.n_user, self.att_dim]), dtype=tf.float32)
        weights['item'] = tf.Variable(np.random.normal(0.0, 0.01, [self.n_item, self.hidden_factor]), dtype=tf.float32)
        return weights

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial)

    def partial_fit(self, data):
        feed_dict = {self.user_idxs: data['user'], self.train_phaise: [[1]], self.recent: data['recent'],
                     self.item_pos: data['pos'], self.item_neg: data['neg']}
        loss, loss_rec, loss_VAE, opt = self.sess.run((self.loss, self.loss_rec, self.KL_divergence, self.optimizer), feed_dict=feed_dict)
        return [loss, loss_rec, loss_VAE]

    def topk(self, user_item_block, recent, Topk=500):
        users = user_item_block[:, 0]
        feed_dict = {self.user_idxs: users, self.train_phaise: [[0]], self.recent: recent}
        f_result = self.sess.run(self.output, feed_dict)
        prediction = np.argsort(f_result)[:, ::-1][:, :Topk]
        return prediction

class Train_MF(object):

    def __init__(self,args,data,path):
        self.args = args
        self.data = data
        self.path = path
        self.batch_size = args.batch_size
        self.epoch = data.epoch
        self.TopK = args.Topk
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.num_topk_user = 500

        print("DHRec-MoE: dataset=%s, factors=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, optimizer=%s"
              %(args.dataset, args.hidden_factor, self.epoch, args.batch_size, args.lr, args.lamda, args.optimizer))

        # 实例化新的多专家VAE模型
        self.model = PVAE_MoE(self.args, self.data, path, args.hidden_factor, args.lr, args.lamda, args.optimizer)

    def train(self):
        MAP_valid = 0
        PosSample= np.array(self.data.train)
        PosSample_with_p5 = np.concatenate([PosSample,np.array([
                    self.data.latest_interaction[(line[0],line[1])] for line in PosSample])],axis =1)

        for epoch in tqdm(range(0, self.epoch + 1)):
            np.random.shuffle(PosSample_with_p5)
            NegSample = self.sample_negative(PosSample_with_p5, 5)

            for user_chunk in toolz.partition_all(self.batch_size, [i for i in range(len(PosSample_with_p5))]):
                chunk = list(user_chunk)
                item_neg = np.array(NegSample[chunk], dtype=np.int32)
                train_chunk_p5 = PosSample_with_p5[chunk]

                feed_dict = {
                    'user': train_chunk_p5[:, 0],
                    'pos': train_chunk_p5[:, 1],
                    'recent': train_chunk_p5[:, 2:],
                    'neg': item_neg
                }
                loss = self.model.partial_fit(feed_dict)
            
            t2 = time()
            if epoch % int(self.epoch/10) == 0:
                print("Epoch %d: Loss=%.4f, Rec_Loss=%.4f, KL_Loss=%.4f" % (epoch, loss[0], loss[1], loss[2]))
                with open("./final_result.txt", "a") as f:
                    f.write("%s \t %s\t %s  \t ||||" % (self.args.dataset, self.args.model, epoch))
                
                for topk in [50]:
                    init_test_TopK_test = self.evaluate_TopK(self.data.test, topk)
                    print("Epoch %d Top%d \t TEST SET: MAP:%.4f, NDCG:%.4f, PREC:%.4f; [%.1f s]\n"
                          % (epoch, topk, init_test_TopK_test[0], init_test_TopK_test[1], init_test_TopK_test[2], time() - t2))
                    with open("./final_result.txt", "a") as f:
                        f.write("Top-%d \t TEST SET,%.4f,%.4f,%.4f\t||||"
                                % (topk, init_test_TopK_test[0], init_test_TopK_test[1], init_test_TopK_test[2]))
                with open("./final_result.txt", "a") as f:
                    f.write("\n")

                if MAP_valid < np.sum(init_test_TopK_test):
                    MAP_valid = np.sum(init_test_TopK_test)
                    result_print = init_test_TopK_test

        with open("./result.txt", "a") as f:
            f.write("%s,%s,%.4f,%.4f,%.4f\n" % (self.args.name, self.args.model, result_print[0], result_print[1], result_print[2]))

    def sample_negative(self, data, num=10):
        samples = np.random.randint(0, self.n_item, size=(len(data), num))
        return samples

    def evaluate_TopK(self, test, topk):
        test_candidate = copy.deepcopy(np.array(test))
        size = len(test_candidate)
        result_MAP, result_PREC, result_NDCG = [], [], []
        num = self.num_topk_user
        
        last_iteraction = np.array([self.data.latest_interaction[(user, item)] for user, item in test_candidate])

        for _ in range(int(size / num + 1)):
            user_item_block = test_candidate[_ * num:(_ + 1) * num]
            if len(user_item_block) == 0: continue
            
            last_iteraction_block = last_iteraction[_ * num:(_ + 1) * num]
            
            prediction = self.model.topk(user_item_block, last_iteraction_block)
            assert len(prediction) == len(user_item_block)

            for i, line in enumerate(user_item_block):
                user, item = line
                n = 0
                for it in prediction[i]:
                    if n > topk - 1:
                        result_MAP.append(0.0)
                        result_NDCG.append(0.0)
                        result_PREC.append(0.0)
                        break
                    elif it == item:
                        result_MAP.append(1.0)
                        result_NDCG.append(np.log(2) / np.log(n + 2))
                        result_PREC.append(1 / (n + 1))
                        break
                    elif it in self.data.set_forward['train'][user]:
                        continue
                    else:
                        n = n + 1
                else: # if loop doesn't break
                    result_MAP.append(0.0)
                    result_NDCG.append(0.0)
                    result_PREC.append(0.0)
        
        return [np.mean(result_MAP), np.mean(result_NDCG), np.mean(result_PREC)]

def MoE_VAE_run(name,factor,Topk,feature,AggMethod,noise_num,KL,kl_co):
    args = parse_args(name,factor,Topk,feature,AggMethod,noise_num,KL,kl_co)
    data = Data(args, 0) 
    path = PathCount(args, data)
    session_DHRec = Train_MF(args, data, path)
    session_DHRec.train()

