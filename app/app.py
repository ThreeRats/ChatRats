from flask import Flask, render_template, jsonify, request
from model_api import *
import argparse
from send_message import Message

def get_model_from_checkpoint():
    """
    加载预训练的模型,参数在ptuning_checkpoint文件中
    """
    return ChatRats(
        model_name_or_path='../chatglm-6b-int4',
        ptuning_checkpoint='../ptuning/output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-6000',
    )

def get_pretrained_model():
    """
    加载chatglm预训练模型
    """
    return ChatRats(
        model_name_or_path="../chatglm-6b-int4",
    )

app = Flask(__name__)
# 初始化模型,类型默认是1
print("加载微调模型")
chatrat = get_model_from_checkpoint()
now_type = 1

# 初始化langchain object
prompt = Message()

print('test:\n', prompt.get_context("肚子痛"))

@app.route('/')
def index():
    """
    根路由
    """
    return render_template("index.html")

@app.route("/sendChat", methods=['POST'])
def sendChat():
    """
    获取提问信息
    """
    query_dict = request.form
    
    print('请求: ', query_dict)
    
    global chatrat, now_type
    
    # 判断询问的模型
    model_type = query_dict.get("type", None)
    query_text = query_dict.get("chatText", None)
    
    print('原始: ', query_text)
    
    if model_type is not None and now_type != int(model_type):
        # 需要切换模型
        if model_type == 1:
            del chatrat
            # 加载微调模型
            chatrat = get_model_from_checkpoint()
        elif model_type == 2:
            del chatrat
            # 加载预训练模型
            chatrat = get_pretrained_model()
        elif now_type == 1:
            del chatrat
            # 加载预训练模型
            chatrat = get_pretrained_model()
            
        now_type = int(model_type)
    
    print('请求类型: ', now_type)
    
    if now_type == 3:
        query_text = prompt.get_context(query_text)
        print('添加后: ', query_text)

    op = chatrat.predict(input_str=query_text, max_length=2048*16,top_p=0.7,temperature=0.95)
    
    print('回答: ', op)
    
    result = {
        "status": 1,
        "answer": op,
    }

    return jsonify(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-port")
    args = parser.parse_args()
    
    port = 6006
    if args.port is not None:
        port = args.port

    app.run(port=port)
