from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pickle

class BuildVecStore():
    def __init__(self, embeddings, docs, vector_store_path) -> None:
        self.embeddings = embeddings
        self.demo_sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
        self.docs = docs
        self.vector_store_path = vector_store_path
        self.has_build_store = False
    
    def save(self):
        if self.has_build_store == False:
            print(f'没有建立过vec store，开始建立...')
            vector_store = FAISS.from_documents(self.docs, self.embeddings)
            vector_store.save_local(self.vector_store_path)
            self.has_build_store = True

    def demo(self):
        vectors = self.embeddings.embed_documents(self.demo_sentences)
        print(f'the length of vectors[0] is {len(vectors[0])}')

