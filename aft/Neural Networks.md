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

6.      关于RNN

a)        lstm和gru的原理

b)       lstm和gru的参数估计

c)        它们是如何解决梯度消失的问题的