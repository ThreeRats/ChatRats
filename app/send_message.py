from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from build_vec_store import BuildVecStore
import os
import pickle

class Message:
    def __init__(self) -> None:

        self.embeddings = None
        if not os.path.exists("embeddings.txt"):
            print('embeddings文件不存在')
            self.embeddings = HuggingFaceEmbeddings(model_name='GanymedeNil/text2vec-large-chinese')
            with open('embeddings.txt', 'wb') as f:
                pickle.dump(self.embeddings, f)
        else:
            print('embeddings文件存在')
            with open('embeddings.txt', 'rb') as f:
                self.embeddings = pickle.load(f)
        
        self.prompt_template =  """知识库信息：
    {context} 

    根据从知识库中获取的已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据知识库中的信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

            
    def get_context(self, query) -> str:
        
        '''
        根据输入的查询问题, 输出含有上下文的模板
        '''
        with open('embeddings.txt', 'rb') as f:
            embeddings = pickle.load(f)
            
        if os.path.exists('./vector_store'):
            print('vector_store文件夹存在')
            
        else:
            print('vector_store文件夹不存在')

            with open('./docs.pkl', 'rb') as f:
                docs = pickle.load(f)
            BuildVecStore(embeddings, docs, 'vector_store').save()
            print(f'vec_store 构建完成...')

        local_vector_store = FAISS.load_local('vector_store', embeddings)
        local_vector_store.score_threshold = 499
        related_docs_with_score = local_vector_store.similarity_search_with_score(query, k=5)
        
        context = "\n".join([doc[0].page_content for doc in related_docs_with_score])

        prompt = self.prompt_template.replace("{question}", query).replace("{context}", context)
        # qas = [doc[0].page_content for doc in related_docs_with_score]
        # ans = []

        # for q in qas:
        #     question = q.split(' ')[0]
        #     answer = q.split(' ')[-1]

        #     ans.append((question, answer))
        return prompt


if __name__ == '__main__':
    query = '肚子痛'
    prompt = Message().get_context(query)
    print(prompt)

