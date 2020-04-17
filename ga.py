#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:44:16 2020

@author: bvegetas
"""
import numpy as np

def ga(fun,x,para,crossover,mutation,n_iter=20,limitation=None,postprocess=None,cvpara=None,mupara=None,pppara=None,rndpool=None):
    '''
    bestx,byproduct=ga(fun,shape,para,crossover,mutation,limitation=None,postprocess=None,cvpara=None,mupara=None,pppara=None,rndpool=None)
    基于遗传算法，对目标函数fun的取值最小化
   
    Parameters
    ----------
    fun : function
        待优化的目标函数，形式为(目标函数值,其他返回值)=fun(待优化变量,para)，其中para为fun的其他参数
    x : numpy.array
        待优化变量的初始值，每行为一个初始值，每列为一维待优化变量
    para : tuple
        目标函数fun的其他参数
    crossover : function
        杂交方法，形式为：待优化变量,rndpool=crossover(待优化变量,rndpool,cvpara)
    mutation : function
        变异方法，形式为：待优化变量,rndpool=mutation(待优化变量,rndpool,mupara)
    n_iter : int, optional
        程序最大迭代次数. The default is 20.
    limitation : function, optional
        对待优化变量的额外约束条件的检验及修复，形式为：待优化变量=limitation(待优化变量) The default is None.
    postprocess : function, optional
        迭代完成后对待优化变量的额外后处理，形式为：待优化变量,rndpool=postprocess(待优化变量,rndpool,pppara) The default is None.
    cvpara : tuple, optional
        函数crossover的额外参数集. The default is None.
    mupara : tuple, optional
        函数mutation的额外参数集. The default is None.
    pppara : tuple, optional
        函数postprocess的额外参数集. The default is None.
    rndpool : numpy.array(dtype=float), optional
        本程序支持使用真随机数序列覆盖系统的伪随机数产生器。
        若不需要使用真随机数，请忽略该参数
        若要使用真随机数，请将真随机数序列转换为服从[0,1]均匀分布的浮点型随机数序列并传入该参数
        The default is None.

    Returns
    -------
    bestx : numpy.ndarray
        待优化变量的优化结果
    byproduct : tuple
        目标函数fun的其他返回值
    rndpool : numpy.array, optional
        运算完成后未使用的真随机序列。若程序未指定真随机数序列，则无该返回项
    '''
    if rndpool is None:
        rndpool=np.random.rand(1000000)
        retrnd=False
    else:
        retrnd=True
    #检验及修复
    if not (limitation is None):
        x=limitation(x)
    for epåk in range(n_iter):
        
        
        
        
        #个体选拔
        y=fun(x,para)
        if y is tuple:
            y=-np.exp(y[0])
        else:
            y=-np.exp(y)
        y/=y.sum()
        p=np.c_[np.zeros((1,1)),y.reshape((1,-1)).dot(np.triu(np.ones((len(y),len(y)))))].reshape(-1)
        p[np.isnan(p)]=0
        p/=p.max()
        z=np.zeros(x.shape)
        for k in range(z.shape[0]):
            try:
                z[k,:]=np.where((p[:-1]<=rndpool[k])*(p[1:]>rndpool[k]))[0][0]
            except IndexError:
                None
        rndpool=rndpool[k:]
        
        #杂交
        if not (cvpara is None):
            x,rndpool=crossover(z,rndpool,cvpara)
        else:
            x,rndpool=crossover(z,rndpool)
        
        #变异方法
        if not (mupara is None):
            x,rndpool=mutation(x,rndpool,mupara)
        else:
            x,rndpool=mutation(x,rndpool)
        #检验及修复
        if not (limitation is None):
            x=limitation(x)
        #额外后处理
        if not (postprocess is None):
            if not (pppara is None):
                x,rndpool=postprocess(x,rndpool,pppara)
            else:
                x,rndpool=postprocess(x,rndpool)
        
    y=fun(x,para)
    if y is tuple:
        byp=True
    else:
        byp=False
    if retrnd:
        if byp:
            return x,y[1:],rndpool
        return x,rndpool
    elif byp:
            return x,y[1:]
    return x
    
    
"""以下为示例代码"""
def fun(x,para):
    #用遗传算法解方程组exp(x)+x^2=para
    return np.sum((np.exp(x)+x**2-para)**2,axis=1)

def crossover(x,rndpool):
    #杂交方法为抽取2个个体进行染色体半数基因的交换，待交换的基因随机抽取
    v=np.argsort(rndpool[:x.shape[0]])
    rndpool=rndpool[x.shape[0]:]
    w=np.argsort(rndpool[:x.shape[1]])
    rndpool=rndpool[x.shape[1]:]
    y=np.zeros(x.shape)
    y[0,w[:int(len(w)/2)]]=x[v[0],w[:int(len(w)/2)]]
    y[0,w[int(len(w)/2):]]=x[v[1],w[int(len(w)/2):]]
    y[1,w[:int(len(w)/2)]]=x[v[1],w[:int(len(w)/2)]]
    y[1,w[int(len(w)/2):]]=x[v[0],w[int(len(w)/2):]]
    y[2:,:]=x[v[2:],:]
    return y,rndpool

def mutation(x,rndpool,mupara):
    #变异方法为抽取一个基因将其重设为服从参数为mupara[0]的指数分布的随机数，变异概率为mupara[1]
    for k in range(x.shape[0]):
        if rndpool[0]<mupara[1]:
            d=int(rndpool[1]*x.shape[1])
            if len(mupara)>=2+x.shape[1]:
                x[k,d]=np.log(1-rndpool[2])*mupara[0]+mupara[2+d]
            else:
                x[k,d]=np.log(1-rndpool[2])*mupara[0]
            rndpool=rndpool[3:]
        else:
            rndpool=rndpool[1:]
    return x,rndpool
def limitation(x):
    #约束条件：所有变量之和不超过维数的3倍
    #对于不满足约束条件的变量，则令变量的每个值*3/变量之和
    for k in range(x.shape[0]):
        if x[k,:].sum()>3*x.shape[1]:
            x[k,:]=x[k,:]*3/x[k,:].sum()
    return x
            
def postprocess(x,rndpool,para):
    #对x施以微小扰动
    x+=(rndpool[:np.prod(x.shape)].reshape(x.shape)-0.5)*para
    return x,rndpool[np.prod(x.shape):]

x=np.random.rand(128,3)+0.5
x=ga(fun,x,np.linspace(4,6,3),crossover,mutation,n_iter=1000,limitation=limitation,postprocess=postprocess,
     cvpara=None,mupara=(2,0.1,0.5,0.6,0.7),pppara=1e-2)
y=fun(x,np.linspace(4,6,3))
bestx=x[y==y.min(),:]
besty=y.min()
print('1000次迭代后，方程的近似解为',bestx,'，误差平方和为',besty)