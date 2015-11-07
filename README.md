# PRML
## fpgrowth
* 在算法中使用了一种称为频繁模式树（Frequent Pattern Tree）的数据结构。FP-tree是一种特殊的前缀树，由频繁项头表和项前缀树构成。
* 主要思路是，利用1-item中的频繁模式，对数据中的每个记录中的模式进行排序，得到最有可能的模式，然后构造一棵FPtree和itemtable，通过列表和树可以找到以某个item为后缀的前缀组合，进行分析
* 使用方式直接运行 fpgrowth.py 文件即可，它所以来的数据在data中

## EM algorithm  
* create_config.py --> to use the json file to save the init EM parameter. include Mu and Sigma  
* datagen.py --> to create the datum which obey some distribution  
* EM.py --> acording the json file to estimate the parameter.
* gmm.py --> to create Gaussian distribution datum

