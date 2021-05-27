# Predict-Crypto-using-LSTM
参考大佬 https://github.com/NourozR/Stock-Price-Prediction-LSTM 的程序，用LSTM预测ETH的价格。由于ETH等Crypto的价格波动率较大，效果还有待提高。
作为一种递归神经网络，LSTM具有记忆性，能够在生成每一个新的神经节点的时候考虑到过去重要的信息。市场对于价格应该是具有记忆性的，所以使用这个模型是合理的。

但是很显然，这个预测是非常弱的。要提高预测的准确性，应该在数据和算法这两方面都加强。

1）数据方面
准确预测市场是一项复杂的任务，因为一个特定的方向移动有数百万个事件和先决条件。所以我们需要尽可能多地捕捉这些前提条件。除去价格信息，还有Technical indicators(like exponential moving average, momentum, Bollinger bands), Fundamental analysis( like 10-K report, News sentiment analysis), Correlated assets( any type, not necessarily stocks, such as commodities, FX, indices, or even fixed income securities)等等非常多的侧面和替代信息可以挖掘。
获得数据之后的预处理也很重要。毕竟rubbish in, rubbish out。主要是要降噪，把数据中的随机因素尽量洗掉。我之前做的一个模型用random matrix theory可以把相关性矩阵中由随机性造成的相关性清洗，这在金融当中（像maximum sharp, minimum variance等用到矩阵的算法）应该可以有不错的应用。在这一次查询数据的过程中，我了解到Fourier transforms也可以有效降噪（random walk），得到真实的股票走势。下一次可以试试。

2）算法方面
数据处理的每个阶段都有不同的要求，应该在不同的阶段用不同的算法。比如说，可以在生成时间序列的时候用LSTM，在鉴别数据的时候用CNN。当然，深度学习里面有很多fancy的算法，还没有被用在金融领域上。今后我要广泛了解算法特征，再选择合适的使用在金融上。
