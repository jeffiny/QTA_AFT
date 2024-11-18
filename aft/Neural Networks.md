1.      什么是Mini-batch gradient descent，优势是什么
	mini-batch梯度下降（MBGD，Mini-Batch Gradient Descent）就是把数据集分为若干个子集，然后每次把每个子集作为我们的训练数据进行梯度下降。例如样本数为m，每一个batch的大小为64，那么我们就可以分为m/64个样本，如果m%64不等于0说明还有剩的样本，则第m/64+1个batch不足64，大小就等于m%64。
```
repeat num iterations{
	遍历每一个batch{
		1.前向传播：（1）计算Z=W*X+b
				    （2）计算激活项的值A=g(Z)
		2.计算代价J
		3.反向传播求解梯度
		4.更新权重
	}
}
```
优势：当训练集很大时，mini-batch能加快我们的参数更新速度。

2.      在pytorch中的固定流程
```python
pytorch官方的代码：
#step1:获取训练设备
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
#step2：定义类
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x): //前向传播
        x = self.flatten(x) //转换数据成连续数组
        logits = self.linear_relu_stack(x)
        return logits
#step3：进行预测
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```
根据理论的神经网络代码：
```python
'''定义神经网络'''
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten=nn.Flatten()
        self.hidden1=nn.Linear(28*28,128)
        self.hidden2=nn.Linear(128,128)
        self.hidden3=nn.Linear(128,64)
        self.out=nn.Linear(64,10)
    def forward(self,x):
        x=self.flatten(x)
        x=self.hidden1(x)
        x=torch.relu(x)
        x=self.hidden2(x)
        x=torch.sigmoid(x)
        x=self.hidden3(x)
        x=torch.relu(x)
        x=self.out(x)
        return x
 
model=NeuralNetwork().to(device)
print(model)
 
'''建立损失函数和优化算法'''
#交叉熵损失函数
loss_fn=nn.CrossEntropyLoss()
# 优化算法为随机梯度算法/Adam优化算法
optimizer=torch.optim.Adam(model.parameters(),lr=0.005)
 
'''定义训练函数'''
def train(dataloader,model,loss_fn,optimizer):
    model.train()
    # 记录优化次数
    num=1
    for X,y in dataloader:
        X,y=X.to(device),y.to(device)
        # 自动初始化权值w
        pred=model.forward(X)
        loss=loss_fn(pred,y) # 计算损失值
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value=loss.item()
        print(f'loss:{loss_value},[numbes]:{num}')
        num+=1
 
train(train_dataloader,model,loss_fn,optimizer)
```
3.      以MLP举例，描述数据从输入到输出的维度变化
MLP原理图如下：
![[image-20241118143227335.png]]
隐藏层：全连接层+激活函数，增加非线性结构，使得数据从N维到M维，
而经过若干隐藏层后，得到输出层（全连接层+激活函数），M维到L维
4.      常用的激活函数和优缺点
（1）Sigmoid 函数
Sigmoid 激活函数的数学表达式为：
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$
---

**Sigmoid 的优点**
- Sigmoid 函数的输出范围是 \( (0, 1) \)，因此常用于二分类任务的概率预测。
- -梯度平滑，便于求导，也防止模型训练过程中出现突变的梯度

--- 
**Sigmoid 的缺点**
- 容易造成梯度消失。我们从导函数图像中了解到sigmoid的导数都是小于0.25的，那么在进行反向传播的时候，梯度相乘结果会慢慢的趋向于0。这样几乎就没有梯度信号通过神经元传递到前面层的梯度更新中，因此这时前面层的权值几乎没有更新，这就叫梯度消失。除此之外，为了防止饱和，必须对于权重矩阵的初始化特别留意。如果初始化权重过大，可能很多神经元得到一个比较小的梯度，致使神经元不能很好的更新权重提前饱和，神经网络就几乎不学习。
- 函数输出不是以 0 为中心的，梯度可能就会向特定方向移动，从而降低权重更新的效率
- Sigmoid 函数执行指数运算，计算机运行得较慢，比较消耗计算资源。

---
（2） Tanh函数
$$
\text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$
---
**Tanh函数的优点**
- tanh 的输出间隔为 1，并且整个函数以 0 为中心，比 sigmoid 函数更好；
- 在 tanh 图中，负输入将被强映射为负，而零输入被映射为接近零。
--- 
**Tanh函数的缺点**
- 仍然存在梯度饱和的问题
- 依然进行的是指数运算
（3）ReLU函数
$$
\text{ReLU}(x) = \max(0, x)​
$$
---
**ReLU函数的优点**
- ReLU解决了梯度消失的问题，当输入值为正时，神经元不会饱和
- 由于ReLU线性、非饱和的性质，在SGD中能够快速收敛
- 计算复杂度低，不需要进行指数运算
---
**ReLU函数的缺点**
- 与Sigmoid一样，其输出不是以0为中心的
- Dead ReLU 问题。当输入为负时，梯度为0。这个神经元及之后的神经元梯度永远为0，不再对任何数据有所响应，导致相应参数永远不会被更新

5.      梯度爆炸和梯度消失是什么
梯度消失：梯度趋近于零，网络权重无法更新或更新的很微小，网络训练再久也不会有效果；  
梯度爆炸：梯度呈指数级增长，变的非常大，然后导致网络权重的大幅更新，使网络变得不稳定。
6.      关于RNN
a)        lstm和gru的原理
---
lstm原理:
遗忘门：通过x和ht的操作，并经过sigmoid函数，得到0,1的向量，0对应的就代表之前的记忆某一部分要忘记，1对应的就代表之前的记忆需要留下的部分 ===>代表复习上一门线性代数所包含的记忆，通过遗忘门，忘记掉和下一门高等数学无关的内容（比如矩阵的秩）

输入门：通过将之前的需要留下的信息和现在需要记住的信息相加，也就是得到了新的记忆状态。===>代表复习下一门科目高等数学的时候输入的一些记忆（比如洛必达法则等等），那么已经线性代数残余且和高数相关的部分（比如数学运算）+高数的知识=新的记忆状态

输出门：整合，得到一个输出===>代表高数所需要的记忆，但是在实际的考试不一定全都发挥出来考到100分。因此，则代表实际的考试分数
```python
# 初始化隐藏状态和细胞状态
h_prev = zero_vector(n_hidden)  # 初始隐藏状态，通常为零向量
c_prev = zero_vector(n_hidden)  # 初始细胞状态，通常为零向量

# 遍历输入序列
for t in range(1, T+1):  # T 是时间步数
    # 遗忘门 (Forget Gate)
    f_t = sigmoid(W_f * x_t + U_f * h_prev + b_f)
    
    # 输入门 (Input Gate)
    # 输入门由一个sigmoid激活函数和一个tanh激活函数组成。
    # sigmoid函数决定哪些信息是重要的，而tanh函数则生成新的候选信息。
    i_t = sigmoid(W_i * x_t + U_i * h_prev + b_i)
    
    # 候选状态 (Candidate Cell State)
    c_candidate = tanh(W_c * x_t + U_c * h_prev + b_c)
    
    # 更新细胞状态 (Cell State Update)
    c_t = f_t * c_prev + i_t * c_candidate
    
    # 输出门 (Output Gate)
    # 输出门由一个sigmoid激活函数和一个tanh激活函数组成。
    # sigmoid函数决定哪些信息应该被输出，而tanh函数则处理记忆单元的状态以准备输出。
    o_t = sigmoid(W_o * x_t + U_o * h_prev + b_o)
    
    # 当前隐藏状态 (Hidden State)
    h_t = o_t * tanh(c_t)
    
    # 更新隐藏状态和细胞状态
    h_prev = h_t
    c_prev = c_t

```

---
gru原理：
- **更新门（Update Gate）**：决定当前时间步的状态需要从过去记住多少，以及从当前输入中更新多少。
- **重置门（Reset Gate）**：决定是否丢弃之前时间步的信息（短期依赖和长期依赖的切换）。
-GRU 的工作过程
1. **输入信息的筛选（重置门）**
    - 类似整理笔记时，决定是否参考旧的笔记内容。
    - 如果重置门的值接近 0，则完全忽略旧的笔记，只关注新信息。
2. **新旧信息的融合（更新门）**
    - 类似更新日记，决定当天的事情（新信息）是否覆盖之前的记录（旧记忆）。
    - 更新门的值接近 1 时，更倾向于记住新信息；值接近 0 时，更倾向于保留旧信息。
3. **状态更新**
    - 整合新信息和历史信息，生成当前时间步的状态（记忆），并传递到下一步。
gru伪代码：
```python
# 初始化隐藏状态
h_prev = zero_vector(n_hidden)  # 初始隐藏状态，通常为零向量

# 遍历输入序列
for t in range(1, T+1):  # T 是时间步数
    # 更新门 (Update Gate):决定信息保留/丢弃
    # W_z，U_z，b_z权重矩阵,h_prev 隐藏状态
    z_t = sigmoid(W_z * x_t + U_z * h_prev + b_z)
    
    # 重置门 (Reset Gate):决定是否清空旧记忆
    # W_r，U_r，b_r权重矩阵,h_prev 隐藏状态
    r_t = sigmoid(W_r * x_t + U_r * h_prev + b_r)
    
    # 候选隐藏状态 (Candidate Hidden State):生成新记忆
    # W_h，U_h，b_h权重矩阵, (r_t * h_prev)表示将前一个时间步的隐藏状态与重置门结合以控制影响程度
    h_candidate = tanh(W_h * x_t + U_h * (r_t * h_prev) + b_h)
    
    # 当前隐藏状态 (Current Hidden State):融合新旧信息
    # z_t * h_prev保留历史信息，(1 - z_t) * h_candidate表示新信息
    h_t = z_t * h_prev + (1 - z_t) * h_candidate
    
    # 更新隐藏状态
    h_prev = h_t
    
```
b)       lstm和gru的参数估计
lstm参数估计：
问题：
1、在确保了数据与网络的正确性之后，使用默认的超参数设置，观察loss的变化，初步定下各个超参数的范围，再进行调参。对于每个超参数，我们在每次的调整时，只去调整一个参数，然后观察loss变化，千万不要在一次改变多个超参数的值去观察loss。

2、对于loss的变化情况，主要有以下几种可能性：上升、下降、不变，对应的数据集有train与val（validation），那么进行组合有如下的可能：

train loss 不断下降，val loss 不断下降——网络仍在学习；

train loss 不断下降，val loss 不断上升——网络过拟合；

train loss 不断下降，val loss 趋于不变——网络欠拟合；

train loss 趋于不变，val loss 趋于不变——网络陷入瓶颈；

train loss 不断上升，val loss 不断上升——网络结构问题；

train loss 不断上升，val loss 不断下降——数据集有问题；

其余的情况，也是归于网络结构问题与数据集问题中。

3、当loss趋于不变时观察此时的loss值与1-3中计算的loss值是否相同，如果相同，那么应该是在梯度计算中出现了nan或者inf导致oftmax输出为0。

此时可以采取的方式是减小初始化权重、降低学习率。同时评估采用的loss是否合理。

方法：
1、当网络过拟合时，可以采用的方式是正则化（regularization）与丢弃法（dropout）以及BN层（batch normalization），正则化中包括L1正则化与L2正则化，在LSTM中采用L2正则化。另外在使用dropout与BN层时，需要主要注意训练集和测试集上的设置方式不同，例如在训练集上dropout设置为0.5，在验证集和测试集上dropout要去除。

2、当网络欠拟合时，可以采用的方式是：去除 / 降低 正则化；增加网络深度（层数）；增加神经元个数；增加训练集的数据量。

3、设置early stopping，根据验证集上的性能去评估何时应该提早停止。

4、对于LSTM，可使用softsign（而非softmax）激活函数替代tanh（更快且更不容易出现饱和（约0梯度））

5、尝试使用不同优化算法，合适的优化器可以是网络训练的更快，RMSProp、AdaGrad或momentum（Nesterovs）通常都是较好的选择。

6、使用梯度裁剪（gradient clipping），归一化梯度后将梯度限制在5或者15。

7、学习率（learning rate）是一个相当重要的超参数，对于学习率可以尝试使用余弦退火或者衰减学习率等方法。

7、可以进行网络的融合（网络快照）或者不同模型之间的融合。


gru的参数
input_size: input的特征维度
hidden_size: 隐藏层的宽度
num_layers: 单元的数量（层数），默认为1，如果为2以为着将两个GRU堆叠在一起，当成一个GRU单元使用。
bias: True or False，是否使用bias项，默认使用
batch_first: Ture or False, 默认的输入是三个维度的，即：(seq, batch, feature)，第一个维度是时间序列，第二个维度是batch，第三个维度是特征。如果设置为True，则(batch, seq, feature)。即batch，时间序列，每个时间点特征。
dropout：设置隐藏层是否启用dropout，默认为0
bidirectional：True or False, 默认为False，是否使用双向的GRU，如果使用双向的GRU，则自动将序列正序和反序各输入一次。

c)        它们是如何解决梯度消失的问题的
    GRU和LSTM中的门控设计策略能够有助于缓解梯度消失或梯度爆炸问题。主要是解决长序列梯度计算中幂指数大小的问题（长序列意味着高阶乘积计算，容易导致梯度极大或极小），可以通过门控设计来直接减少高阶乘积大小（直接替换高阶乘积计算，替换为合理数值），从而缓解梯度消失或梯度爆炸问题。

RNN伪代码
```python
# 初始化隐藏状态
h_prev = zero_vector(n_hidden)  # 初始隐藏状态，通常为零向量

# 遍历输入序列
for t in range(1, T+1):  # T 是时间步数
    # 计算当前时间步的隐藏状态 
    # 常用的激活函数是 tanh或ReLU
    h_t = activation_function(W_xh * x_t + W_hh * h_prev + b_h)
    
    # 计算当前时间步的输出
    # 输出函数通常是 Softmax（用于分类）或 Linear（用于回归）。
    y_t = output_function(W_hy * h_t + b_y)
    
    # 更新隐藏状态
    h_prev = h_t

```
