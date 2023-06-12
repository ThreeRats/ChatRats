# ChatRats
bupt nlp课设 基于chatglm微调搭建的智能聊天机器人

## 运行方法

### 微调方法

相关代码在./ptuning目录下，顺序执行./ptuning/fine-ptuning.ipynb文件中所有的代码块即可。

微调过程中每1000 steps会保存一个checkpoint，保存的目录在./ptuning/output目录下。由于该目录下的文件过大，没有上传到GitHub上。

> 模型下载说明:
> 
> 微调过程中会从huggingface拉取chatGLM-6b-int4预训练模型到./chatglm-6b-int4目录下（自动创建），大概需要占用10GB磁盘空间。

### web端部署

前端VUE，后端Flask。

web端展示了三个不同的模型，分别是使用中文医疗问答数据集进行ptuning微调后的模型、原始chatGLM-6b-int4预训练模型、langchain+chatGLM-6b-int4模型。

```bash
# cd到./app目录下
$ cd ./app

# 配置python环境
$ pip install -r requirements.txt

# 生成知识库文档对象
# ./app目录下会生成docs.pkl文件
$ python text_split.py

# 部署
# 正常情况下./app目录下会生成vector_store目录（索引文件，用来快速检索）、embeddings.txt文件
# 如果不指定端口，默认部署在6006端口
$ python app.py -port 1234
```

## 文件说明

- chatglm-6b-int4目录：在微调过程中会从huggingface上拉取模型到该目录下（文件过大，未上传GitHub）

- app目录
    * dataset目录：langchain基于的知识库，数据集
    * static目录：css、js、imgs等web文件
    * templates目录：html文件
    * app.py：后端文件
    * build_knowledge_base.py：构建知识库
    * build_vec_store.py：构建本地向量库
    * model_api.py：模型接口文件
    * preprocess_medi.py：预处理原始数据集
    * send_message.py：langchain接口文件
    * text_split.py：切分文本，生成文档对象

- data目录：处理后的中文医疗问答数据集

- ptuning目录：
  - utils目录：模型结构、训练、配置文件
  - finetue-ptuning.ipynb：微调文件
  - output目录：在微调过程中保存的checkpoints（文件过大，未上传GitHub）

## TODO:

 - [ ] ptuning目录下jupyter文件
