# LanguageModel_OnlineMovieComments
LLM基础项目。这是一个用于学习和实践的项目集合。<br>
目的是验证较轻量的LLM模型的可训练性，如这里用124M gpt(LayerNorm而非RMSNorm，与原始gpt保持一致，占用空间稍大)，初步认定它做后续agent的backbone。一方面是为了方便部署在家中搭建的分布式集群上，另一方面是为了方便部署在ESP32板上。<br>

#### LLM finetune
实验任务：尝试在IMDB电影评论数据集上训练，以确定评论是积极的还是消极的。<br>
<br>
数据获取：IMDB数据集简单地从斯坦福数据库下载，该数据集是通过爬虫从imdb电影评论中获取的：<br>
http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz<br>
<br>
模型构建：基础的GPT-2（124M），从以下链接下载:<br> 
https://openaipublic.blob.core.windows.net/gpt-2/models<br>
<br>
数据集被划分为训练集、验证集和测试集。训练框架部分基于Huggingface Transformers库，大部分自行构建.<br>
<br>
训练目标与basline：官方以sklearn得出的效果作为baseline：<br>
Training Accuracy: 50.01%<br>
Validation Accuracy: 50.14%<br>
Test Accuracy: 49.91%<br>
<br>
经过微调后的结果：
