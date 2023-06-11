from langchain.text_splitter import CharacterTextSplitter
import re
from typing import Any, List
# from text_split import ChineseTextSplitter
from langchain.document_loaders import TextLoader
import pickle

class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf=False, sentence_size=100, **kwargs: Any):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size
    
    def split_text(self, text: str) -> List[str]:   ##此处需要进一步优化逻辑
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        ls = [i for i in text.split("\n") if i]
        
        return ls

if __name__ == '__main__':

    loader = TextLoader('dataset/medi/knowledge_base.txt', encoding='utf-8')
    textsplitter = ChineseTextSplitter(pdf=False, sentence_size=500)
    print('开始构建docs ...')
    docs = loader.load_and_split(textsplitter)

    with open('docs.pickle', 'wb') as f:
        pickle.dump(docs, f)

    print('构建docs完成!')