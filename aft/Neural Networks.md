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

4.      常用的激活函数和优缺点

5.      梯度爆炸和梯度消失是什么

6.      关于RNN

a)        lstm和gru的原理

b)       lstm和gru的参数估计

c)        它们是如何解决梯度消失的问题的