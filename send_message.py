from langchain.document_loaders import TextLoader
import re
from typing import Any, List
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


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

class BuildVecStore():
    def __init__(self, embeddings, docs, vector_store_path) -> None:
        self.embeddings = embeddings
        self.demo_sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
        self.docs = docs
        self.vector_store_path = vector_store_path
    
    def save(self):
        vector_store = FAISS.from_documents(self.docs, self.embeddings)
        vector_store.save_local(self.vector_store_path)

    def demo(self):
        vectors = self.embeddings.embed_documents(self.demo_sentences)
        print(f'the length of vectors[0] is {len(vectors[0])}')



if __name__ == '__main__':

    
    # loader = TextLoader('dataset/medi/knowledge_base.txt',autodetect_encoding=True)
    # loader = TextLoader('dataset/medi/test_base.txt',autodetect_encoding=True)
    loader = TextLoader('dataset/medi/test_base.txt', encoding='utf-8')
    sentence_size = 500
    textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
    docs = loader.load_and_split(textsplitter)
    print(docs)


    embeddings = HuggingFaceEmbeddings(model_name='GanymedeNil/text2vec-large-chinese')
    BuildVecStore(embeddings, docs, 'dataset/medi/vector_store').save()

    local_vector_store = FAISS.load_local('dataset/medi/vector_store', embeddings)
    local_vector_store.score_threshold = 499
    query = '肚子痛'
    related_docs_with_score = local_vector_store.similarity_search_with_score(query, k=5)

    context = "\n".join([doc[0].page_content for doc in related_docs_with_score])


    prompt_template = """已知信息：
    {context} 

    根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

 
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)

    print(prompt)