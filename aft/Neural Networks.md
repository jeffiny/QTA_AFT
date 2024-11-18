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

2.      在pytorch中的固定流程、
```python

```

3.      以MLP举例，描述数据从输入到输出的维度变化

4.      常用的激活函数和优缺点

5.      梯度爆炸和梯度消失是什么

6.      关于RNN

a)        lstm和gru的原理

b)       lstm和gru的参数估计

c)        它们是如何解决梯度消失的问题的