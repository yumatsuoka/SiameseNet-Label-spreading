# SiameseNet-Label-spreading
## Abstract
深層距離学習を用いた教師データ生成@FSS2016で使用したソースコード  

## Requirements
-Python (checked Python 2.7.6)  
-chainer(checked chainer 1.14.0  

## Result
Plot feature vectors from MNIST images extracted by Siamese Net  
![Alt text](./dump_vec.png)

To generate target data, apply graph-based semi-supervised learning method  
to them based on their data structures.  
This method is really immature on generation of target data.  
