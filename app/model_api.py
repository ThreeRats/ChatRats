import os
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)


class ChatRats:
    def __init__(self, model_name_or_path, ptuning_checkpoint=None, pre_seq_len=128, prefix_projection=None, quantization_bit=None) -> None:
        """_summary_

        Args:
            model_name_or_path (_type_): /root/autodl-tmp/model/chatglm-6b-int4
            ptuning_checkpoint (_type_): /opt/nlpchat/ChatGLM-6B/ptuning/output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-1000
        """
        # 声明两个全局变量model和tokenizer，分别用来存储模型和分词器
        self.model = None
        self.tokenizer = None

        self.history = []

        # 调用AutoTokenizer类的from_pretrained方法，传入模型的名称或路径，以及trust_remote_code参数为True，表示信任远程代码，返回一个分词器对象，并赋值给tokenizer变量。
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True)
        # 调用AutoConfig类的from_pretrained方法，传入模型的名称或路径，以及trust_remote_code参数为True，表示信任远程代码，返回一个配置对象，并赋值给config变量。
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True)

        #  将model_args对象中的pre_seq_len属性赋值给config对象中的pre_seq_len属性，表示模型的前缀序列长度。
        config.pre_seq_len = pre_seq_len
        # 将model_args对象中的prefix_projection属性赋值给config对象中的prefix_projection属性，表示模型是否使用前缀投影。
        config.prefix_projection = prefix_projection

        # 判断model_args对象中的ptuning_checkpoint属性是否不为None。如果是，说明传入了一个预训练的检查点的路径，需要从中加载模型权重。
        if ptuning_checkpoint is not None:
            #  打印一条信息，显示检查点的路径。
            print(
                f"Loading prefix_encoder weight from {ptuning_checkpoint}")

            #  调用AutoModel类的from_pretrained方法，传入模型的名称或路径，配置对象，以及trust_remote_code参数为True，表示信任远程代码，返回一个模型对象，并赋值给model变量。
            self.model = AutoModel.from_pretrained(
                model_name_or_path, config=config, trust_remote_code=True)

            # 使用torch库的load方法，加载检查点路径下的pytorch_model.bin文件，并返回一个包含模型权重的字典，并赋值给prefix_state_dict变量。
            prefix_state_dict = torch.load(os.path.join(
                ptuning_checkpoint, "pytorch_model.bin"))

            # 创建一个空字典，并赋值给new_prefix_state_dict变量。
            new_prefix_state_dict = {}
            # 遍历prefix_state_dict字典中的键值对。
            for k, v in prefix_state_dict.items():
                # 判断键是否以"transformer.prefix_encoder."开头。如果是，说明是前缀编码器相关的权重。
                if k.startswith("transformer.prefix_encoder."):
                    # 将键去掉"transformer.prefix_encoder."前缀，并将其作为新字典new_prefix_state_dict中的键，将值作为新字典new_prefix_state_dict中对应键的值。这样就得到了只包含前缀编码器权重的新字典。
                    new_prefix_state_dict[k[len(
                        "transformer.prefix_encoder."):]] = v
            # 调用模型对象中transformer属性中prefix_encoder属性中load_state_dict方法，传入新字典new_prefix_state_dict，将其加载到前缀编码器中。这样就完成了预训练权重的加载。
            self.model.transformer.prefix_encoder.load_state_dict(
                new_prefix_state_dict)
        else:
            self.model = AutoModel.from_pretrained(
                model_name_or_path, config=config, trust_remote_code=True)

        # 判断model_args对象中quantization_bit属性是否不为None。如果是，说明需要对模型进行量化操作。
        if quantization_bit is not None:
            print(f"Quantized to {quantization_bit} bit")
            # 打印一条信息，显示量化后的位数
            self.model = self.model.quantize(quantization_bit)

        if pre_seq_len is not None:
            # P-tuning v2
            self.model = self.model.half().cuda()
            self.model.transformer.prefix_encoder.float().cuda()

        self.model = self.model.eval()

    def parse_text(self, text):
        """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
        lines = text.split("\n")
        lines = [line for line in lines if line != ""]
        count = 0
        for i, line in enumerate(lines):
            if "```" in line:
                count += 1
                items = line.split('`')
                if count % 2 == 1:
                    lines[i] = f'<pre><code class="language-{items[-1]}">'
                else:
                    lines[i] = f'<br></code></pre>'
            else:
                if i > 0:
                    if count % 2 == 1:
                        line = line.replace("`", "\`")
                        line = line.replace("<", "&lt;")
                        line = line.replace(">", "&gt;")
                        line = line.replace(" ", "&nbsp;")
                        line = line.replace("*", "&ast;")
                        line = line.replace("_", "&lowbar;")
                        line = line.replace("-", "&#45;")
                        line = line.replace(".", "&#46;")
                        line = line.replace("!", "&#33;")
                        line = line.replace("(", "&#40;")
                        line = line.replace(")", "&#41;")
                        line = line.replace("$", "&#36;")
                    lines[i] = "<br>"+line
        text = "".join(lines)
        return text

    def predict(self, input_str, max_length, top_p, temperature):
        for response, self.history in self.model.stream_chat(self.tokenizer, input_str, self.history, max_length=max_length, top_p=top_p, temperature=temperature):
            continue
        return response

    def clear(self):
        """
        清除历史信息
        """
        self.history = []

    def change_history(self, new_history):
        """
        更新历史信息
        """
        self.history = new_history


if __name__ == '__main__':
    chatrat = ChatRats(model_name_or_path='../chatglm-6b-int4',
                       ptuning_checkpoint='../ptuning/output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-6000')
    i = 0
    while i <= 8:
        ip = input("说点什么8：:")
        op = chatrat.predict(input_str=ip, max_length=2048*16,
                             top_p=0.7, temperature=0.95)
        print('RAT说：', op)
        print('chatRats.history', chatrat.history)
