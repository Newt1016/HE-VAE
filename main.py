
from cVaeRec import cVAE_main
from sVaeRec import sVAE_run
from moe import MoE_VAE_run

factor = 64
seed = 0                                        
batch_size = 2048
Topk = 10
#noise_len = 20
kl_co = 1.0
noise_len_hyper = {'ML':[5,10,20,30,40,50],
'Kindle':[5,10,20,30,40,50],
'Games':[5,10,20,30,40,50],
'tiktok':[5,10,20,30,40,50],
'dianping':[5,10,20,30,40,50],
'yelp':[5,10,20,30,40,50]}#

L = {'CiaoDVD':10,'ML':50,'tiktok':20,'dianping':20,'Kindle':20,'Games':20,'yelp':20}
lamda= {'ML':1.0,'tiktok':1.0,'dianping':1.0,'Kindle':1.0,'Games':1.0,'yelp':1.0}


for data in ['ML','dianping','Kindle','Games']: #'ML','dianping','Kindle','Games','tiktok','yelp'
    for i in range(5):

            # VAE_run(data,factor,Topk,'semantic' ,'att',L[data],'KL',lamda[data])
            # sVAE_run(data,factor,Topk,'semantic' ,'att',L[data],'KL',lamda[data])
            # VAE_run(data,factor,Topk,'semantic','att',L[data],'None',lamda[data])
            # cVAE_main(data,factor,seed,batch_size)

            #    AggMethod='PoE' 专家乘积 (Product of Experts)
            MoE_VAE_run(data,factor,Topk,'semantic','PoE',L[data],'KL',lamda[data])

