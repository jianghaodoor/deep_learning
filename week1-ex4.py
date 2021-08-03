import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid,sigmoid_backward,relu,relu_backward
import lr_utils
from scipy.ndimage import _nd_image
import scipy.misc

np.random.seed(1)
#初始化参数
#先是建立两层网络参数
def initaialize_parameters(n_x,n_h,n_y):
    """

    :param n_x: 输入节点的数量
    :param n_h: 隐藏层节点数量
    :param n_y: 输出层节点数量
    :return:
        parameters-包含参数变量的字典
        有parameters W1 b1 W2 b2
    """
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    assert (W1.shape==(n_h,n_x))
    assert (b1.shape==(n_h,1))
    assert (W2.shape==(n_y,n_h))
    assert (b2.shape==(n_y,1))

    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return parameters

print("——————————测试初始化函数————————")
parameters=initaialize_parameters(3,2,1)
print("W1="+str(parameters["W1"]))
print("b1="+str(parameters["b1"]))
print("W2="+str(parameters["W2"]))
print("b2="+str(parameters["b2"]))


def initialize_parameters_deep(layers_dims):
    """

    :param layers_dims: 包含我们网络中每个图层的节点数量的列表
    :return: parameters是包含w1 w2 b1 b2的字典
            Wl-权重矩阵，维度为（layer_dims[l],layer_dims[l-1]）
            b1-偏向两，维度为(layer_dims[l],1)
    """
    np.random.seed(3)
    parameters={}
    L=len(layers_dims)

    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])/np.sqrt(layers_dims[l-1])
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))
        #确保数据正确
        assert (parameters["W"+str(l)]).shape==(layers_dims[l],layers_dims[l-1])
        assert (parameters["b"+str(l)]).shape==(layers_dims[l],1)
    return parameters

print("——————测试intialize_parameters_deep-------")
layer_dims=[5,4,3]
parameters=initialize_parameters_deep(layer_dims)
print("W1="+str(parameters["W1"]))
print("b1="+str(parameters["b1"]))
print("W2="+str(parameters["W2"]))
print("b2="+str(parameters["b1"]))


#前向传播
#线性部分（linear）
def linear_forward(A,W,b):
    """
    实现前向传播的线性部分
    :param A:
    :param W:
    :param b:
    :return:parameters-包含参数的字典
            a-激活功能的输入，
    """
    Z=np.dot(W,A)+b
    assert (Z.shape==(W.shape[0],A.shape[1]))
    cache=(A,W,b)
    return Z,cache
print("========测试linear_forward======")
A,W,b=testCases.linear_forward_test_case()
Z,linear_cache=linear_forward(A,W,b)
print("Z="+str(Z))

#线性+激活部分
def linear_activation_forward(A_prev,W,b,activation):
    """
    先是实现线性，再是激活的这一层的传播
    :param A_prev: 来自上一层激活，（上一层的节点数，样本数量）
    :param W: 权重矩阵，（本层的节点数量，前一层的节点数量）
    :param b: 偏向量，（当前层的节点数量，1）
    :param activation:
    :return: A-激活函数的输出
    cache：包含“linear_cache” “activation_cache”的字典
    """
    if activation=="sigmoid":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)#这里的意思是，A以及activiation_cache被赋予同样的值
    elif activation=="relu":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)

    assert (A.shape==(W.shape[0],A_prev.shape[1]))
    cache=(linear_cache,activation_cache)
    return A,cache

print("----测试linear_activation_forward------")
A_prev,W,b=testCases.linear_activation_forward_test_case()
A,linear_activation_cache=linear_activation_forward(A_prev,W,b,activation="sigmoid")
print("sigmoid,A="+str(A))

A,linear_activation_cache=linear_activation_forward(A_prev,W,b,activation="relu")
print("Relu,A="+str(A))

def L_model_forward(X,parameters):
    """
     实现[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID计算前向传播，也就是多层网络的前向传播，
     为后面每一层都执行LINEAR和ACTIVATION
    :param X: 数据，维度（输入节点数量，实例数）
    :param parameters: initialize_parameters_deep()的输出
    :return:
    AL-最后的激活值
    cache-包含以下内容的缓存列表：
        linear-relu-forward()的每个cache（有l-1个，索引为0到l-2）
        linear-sigmoid-forward()的cache（只有一个，索引为l-1）
    """
    caches=[]
    A=X
    L=len(parameters)//2
    for l in range(1,L):
        A_prev=A
        A,cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters['b'+str(l)],"relu")
        caches.append(cache)
    AL,cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    assert (AL.shape==(1,X.shape[1]))
    return AL,caches
print("====测试L_model_forward======")
X,parameters=testCases.L_model_forward_test_case()
AL,caches=L_model_forward(X,parameters)
print("AL="+str(AL))
print("caches的长度为="+str(len(caches)))


def compute_cost(AL,Y):
    """
    计算成本函数
    :param AL: 与标签对应的概率向量，维度为（1，）
    :param Y:标签向量（1，数量）
    :return:
        cost-交叉成本
    """
    m=Y.shape[1]
    cost=-(1/m)*np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),(1-Y)))
    cost=np.squeeze(cost)
    assert (cost.shape==())
    return cost

#测试computer_cost
print("======测试compute_cost======")
Y,AL=testCases.compute_cost_test_case()
print("cost="+str(compute_cost(AL,Y)))

def linear_backward(dZ,cache):
    """
    为单层实现反向传播的线性部分（第L层）
    :param dz:相对于（当前第l层）线性输出的成本梯度
    :param cache:来自当前层传播值的元组（A_prev，W,b）
    :return:
        dA_prev:相对于激活（前一层l-1）的成本梯度，与A_prev梯度相同
        dW：相对于当前层l的成本梯度，与W的梯度相同
        db：相对于b的成本梯度，与b维度相同
    """
    A_prev,W,b=cache
    m=A_prev.shape[1]
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)
    assert (dA_prev.shape==A_prev.shape)
    assert (dW.shape==W.shape)
    assert (db.shape==b.shape)
    return dA_prev,dW,db
print("=========测试Linear_backward====")
dZ,linear_cache=testCases.linear_backward_test_case()
dA_prev,dW,db=linear_backward(dZ,linear_cache)
print("dW="+str(dA_prev))
print("dW="+str(dW))
print("db="+str(db))

#线性激活部分
def linear_activation_backward(dA,cache,activation="relu"):
    """

    :param dA: 当前l层激活后的梯度值
    :param cache: 是一套元组，值为linear_cache,acitvation_cache
    :param activation:激活函数的名称，要么为sigmoid，要么为relu
    :return:
        dA_prev:前面一层的成本梯度值，与A_prev维度相同
        dW-相对于W的成本梯度值，与W的维度相同
        db-相对于b的成本梯度值，与b的维度相同
    """
    linear_cache,activation_cache=cache
    if activation=="relu":
        dZ=relu_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
    elif activation=="sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
    return dA_prev,dW,db
#测试linear_cativation_backward
print("======测试linear_activation_backward======")
AL,linear_activation_cache=testCases.linear_activation_backward_test_case()
dA_prev,dW,db=linear_activation_backward(AL,linear_activation_cache,activation="sigmoid")
print("sigmoid:")
print("dA_prev="+str(dA_prev))
print("dW="+str(dW))
print("db="+str(db)+'\n')

dA_prev,dW,db=linear_activation_backward(AL,linear_activation_cache,activation="relu")
print("relu:")
print("dA_prev="+str(dA_prev))
print("dW="+str(dW))
print("db="+str(db))

#构建多层模型向后传播函数
def L_model_backward(AL,Y,caches):
    """

    :param AL: 概率向量，正向传播的输出
    :param Y: Y-标签向量
    :param caches:

        linear_activation_forward("relu")的cache，不包含输出层
        linear_activation_forward("sigmoid")的cache
    :return: 具有梯度值的字典
        grads["dA"+str(l)]=
        grads["dw"+str(l)]=
        grads["db"+str(l)]=
    """
    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)
    dAL=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))

    current_cache=caches[L-1]
    grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_backward(dAL,current_cache,"sigmoid")

    for l in reversed(range(L-1)):
        current_cache=caches[l]
        dA_prev_temp,dW_temp,db_temp=linear_activation_backward(grads["dA"+str(l+2)],current_cache,"relu")
        grads["dA"+str(l+1)]=dA_prev_temp
        grads["dW"+str(l+1)]=dW_temp
        grads["db"+str(l+1)]=db_temp
    return grads

#测试L_model_backward
print("=========测试L_model_backward========")
AL,Y_assess,caches=testCases.L_model_backward_test_case()
grads=L_model_backward(AL,Y_assess,caches)
print("dW1="+str(grads["dW1"]))
print("db1="+str(grads["db1"]))
print("dA1="+str(grads["dA1"]))

def updata_parameters(parameters,grads,learning_rate):
    """
    使用梯度下降更新参数
    :param parameters:parameters-包含参数的字典
    :param grads: 包含梯度值的字典，是L_model_backward的输出
    :param learning_rate:
    :return: parameters-包含更新参数的字典
            参数["w"+str(l)]=...
            参数["b"+str(l)]=...
    """
    L=len(parameters)//2
    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters
#测试update_parameters
print("=========测试update_paraeters")
parameters,grads=testCases.update_parameters_test_case()
parameters=updata_parameters(parameters,grads,0.1)
print("W1="+str(parameters['W1']))
print("b1="+str(parameters['b1']))
print("W2="+str(parameters['W2']))
print("b2="+str(parameters['b2']))

#搭建两层神经网络
def  two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=30000,print_cost=False,isPlot=True):
    """
    实现两层神经网络
    :param X: 输入数据，维度为（n_x,例子数）
    :param Y: 标签（1，数量）
    :param layers_dims: 层数向量，维度为（n_y,n_h,n_y）
    :param learning_rate:
    :param num_iterations:
    :param print_cost:是否打印成本值
    :param isPlot:是否会指出误差值图谱
    :return:
    parameters-包含W1 b1 W2 b2的字典变量
    """
    np.random.randn(1)
    grads={}
    costs=[]
    (n_x,n_h,n_y)=layer_dims
    """
    初始化参数
    """
    parameters=initaialize_parameters(n_x,n_h,n_y)
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    "开始迭代"
    for i in range(0,num_iterations):
        #前向传播
        A1,cache1=linear_activation_forward(X,W1,b1,"relu")
        A2,cache2=linear_activation_forward(A1,W2,b2,"sigmoid")

        #计算成本
        cost=compute_cost(A2,Y)

        #后向传播
        ##初始化后向传播
        dA2=-(np.divide(Y,A2)-np.divide(1-Y,1-A2))
        dA1,dW2,db2=linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1=linear_activation_backward(dA1,cache1,"relu")

        #向后传播完成后的数据保存到grads
        grads["dW1"]=dW1
        grads["db1"]=db1
        grads['dW2']=dW2
        grads["db2"]=db2
        #更新参数
        parameters=updata_parameters(parameters,grads,learning_rate)
        W1=parameters["W1"]
        b1=parameters["b1"]
        W2=parameters["W2"]
        b2=parameters["b2"]
        #打印成本值，如果Pirnt_cost=False则忽略
        if i %100==0:
            #记录成本
            costs.append(cost)
            #是否打印成本值
            if print_cost:
                print("第",i,"次迭代，成本值为：",np.squeeze(cost))
    #迭代完成，根据条件绘图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iteartion(per tens)")
        plt.title("Learning rate = "+str(learning_rate))
        plt.show()

    return parameters

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes =lr_utils.load_dataset()
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

n_x=12288
n_h=7
n_y=1
layer_dims=(n_x,n_h,n_y)
# parameters=two_layer_model(train_x,train_set_y,layers_dims=(n_x,n_h,n_y),num_iterations=2500,print_cost=False,isPlot=False)

#建立预测函数
def predict(X,y,parameters):
    """
    预测神经网络层的结果
    :param X:
    :param y:
    :param parameters:
    :return:
    """
    m=X.shape[1]
    n=len(parameters)//2#神经网络的层数
    p=np.zeros((1,m))
    #根据参数前向传播
    probas,caches=L_model_forward(X,parameters)
    for i in range(0,probas.shape[1]):
        if probas[0,i]>0.5:
            p[0,i]=1
        else:
            p[0,i]=0
    print("准确度： "+str(float(np.sum((p==y))/m)))
    return p
# predictions_train=predict(train_x,train_y,parameters)
# predictions_test=predict(test_x,test_y,parameters)


#搭建多层神经网络
def L_layer_model(X,Y,layer_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False,isPlot=True):
    """

    :param X:
    :param Y:
    :param layer_dims: 层数向量，维度为（n_x,n_h,n_h...,n_h,n_y）
    :param learning_rate: 学习率
    :param num_iterations:
    :param print_cost:
    :param isPlot:
    :return:
    parameters-模型学习的参数，可以用这个来预测
    """
    np.random.seed(1)
    costs=[]
    parameters=initialize_parameters_deep(layer_dims)
    for i in range(0,num_iterations):
        AL,caches=L_model_forward(X,parameters)
        cost=compute_cost(AL,Y)
        grads=L_model_backward(AL,Y,caches)
        parameters=updata_parameters(parameters,grads,learning_rate)
        #打印成本值，如果print_cost=False
        if i%100==0:
            costs.append(cost)
            if print_cost:
                print("第",i,"次迭代，成本值为：",np.squeeze(cost))
        #绘图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations(per tens)")
        plt.title("Learning_rate="+str(learning_rate))
        plt.show()
    return parameters

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=lr_utils.load_dataset()
train_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
train_x=train_x_flatten/255
train_y=train_set_y
test_x=test_x_flatten/255
test_y=test_set_y
layer_dims=[12288,20,7,5,1]
parameters=L_layer_model(train_x,train_y,layer_dims,num_iterations=2500,print_cost=True,isPlot=True)
pred_train=predict(train_x,train_y,parameters)
pred_test=predict(test_x,test_y,parameters)

#分析：判断被错误标记的模型
def print_mislabeled_images(classes,X,y,p):
    """
    绘制预测和实际不同的图像
    :param classes:
    :param X: X-数据集
    :param Y: 实际的标签
    :param p: 预测
    :return:
    """
    a=p+y
    mislabeled_indices=np.array(np.where(a==1))
    plt.rcParams['figure.figsize']=(40.0,40.0)
    num_images=len(mislabeled_indices[0])
    for i in range(num_images):
        index=mislabeled_indices[1][i]

        plt.subplot(2,num_images,i+1)
        plt.imshow(X[:,index].reshape(64,64,3),interpolation='nearest')
        plt.axis('off')
        plt.title("Prediciton: "+classes[int(p[0,index])].decode("utf-8")+"\n Class："+classes[y[0,index]].decode("utf-8"))


print_mislabeled_images(classes,test_x,test_y,pred_test)
# plt.show()