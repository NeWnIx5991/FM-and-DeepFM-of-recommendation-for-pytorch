# FM-and-DeepFM-of-recommendation-for-pytorch
implement of FM and DeepFM with pytorch

数据集：
adult_data.csv
关于人口普查信息，特征为每个人的个人信息

任务：
预测这个人的年薪，二分类

过程及结果：
==========================================
FM
==========================================
1、LR（无交叉二阶特征） + 所有特征离散化（非one-hot encoding）
  运行 30 epoch
  train : 
    epoch : 29 ,Loss : 165.12835693359375
  test 
    accuracy : 75.50%
  
2、LR（无交叉二阶特征） + 所有特征离散化（one-hot encoding）
  运行 30 epoch
  train : 
    epoch : 29 ,Loss : 165.09274291992188
  test 
    accuracy : 75.60%
    
3、LR（有交叉二阶特征） + 所有特征离散化（非one-hot encoding）
  运行 100 epoch
  train : 
    epoch : 99 ,Loss : 190.89146423339844
  test 
    accuracy : 80.97%
    
4、LR（有交叉二阶特征） + 所有特征离散化（one-hot encoding）
  运行 100 epoch
  train : 
    epoch : 99 ,Loss : 116.1517333984375
  test 
    accuracy : 83.48%
   
=========================================
DeepFM
=========================================
1、LR + DNN = DeepFM + （有交叉二阶特征） + 所有特征离散化（one-hot encoding）
  运行 100 epoch
  train : 
    epoch : 99 ,Loss : 116.04943084716797
  test 
    accuracy : 83.37%
 
2、LR + DNN = DeepFM + （有交叉二阶特征） + 所有特征离散化（one-hot encoding） + relu
  运行 100 epoch
  train : 
    epoch : 99 ,Loss : 113.17699432373047
  test 
    accuracy : 83.44%
    

3、LR + DNN = DeepFM + （有交叉二阶特征） + 所有特征离散化（one-hot encoding） + relu + dropout
  运行 100 epoch
  这里和下面都有微小提升，结果就不展示了
    

4、LR + DNN = DeepFM + （有交叉二阶特征） + 所有特征离散化（one-hot encoding） + relu + dropout + 神经元个数更改
  运行 100 epoch
  
 代码是DeepFM，FM就是把相应的DNN部分去掉即可。
