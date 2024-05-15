import os
from typing import List

import nltk
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypinyin import pinyin, Style

from LlmClient import LlmClient
from RerankModel import RerankerModel
from Configs import rerank_model_path, embedding_path, filepath
from splitter.chinese_text_splitter import ChineseTextSplitter

nltk_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'zhipu_chat/nltk_data')
nltk.data.path.insert(0, nltk_data_path)



# 创建嵌入模型
embeddings = HuggingFaceEmbeddings(model_name=embedding_path)

# 获取文件名
file_name = os.path.basename(filepath)
# 将文件名转换为拼音
pinyin_names = pinyin(file_name, style=Style.NORMAL)
# 生成数据库id
kb_id = ''.join([item[0] for item in pinyin_names]).replace('.', '_')

faiss_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), kb_id, 'faiss_index')


def merge_splits(docs) -> List:
    new_docs = []
    for doc in docs:
        if not new_docs:
            new_docs.append(doc)
        else:
            last_doc = new_docs[-1]
            if len(last_doc.page_content) + len(doc.page_content) < 200:
                last_doc.page_content += '\n' + doc.page_content
            else:
                new_docs.append(doc)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
        chunk_size=400,
        chunk_overlap=100,
    )
    end_docs = splitter.split_documents(new_docs)
    return end_docs


if os.path.exists(faiss_index_path):
    print("Index loaded from：", faiss_index_path)
    index = FAISS.load_local(folder_path=faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
else:
    loader = UnstructuredWordDocumentLoader(filepath)

    # 创建文本分割器
    text_splitter = ChineseTextSplitter()
    # 分割文本
    splits = loader.load_and_split(text_splitter)
    # 再次分割处理
    splits = merge_splits(splits)
    # 创建索引
    index = FAISS.from_texts(
        texts=[doc.page_content for doc in splits],
        embedding=embeddings
    )
    # 保存索引
    index.save_local(folder_path=faiss_index_path)
    print("Index saved to：", faiss_index_path)

# 基于问题检索出类似的文档段落，喂给llm，llm经过推理后获取答案
llm_client = LlmClient()
while True:
    user_input = input("请输入文字，按回车键确认：")
    # 检查用户是否想要退出
    if user_input.lower() == 'exit':
        print("程序退出。")
        break
    # 执行相似性搜索，并返回与给定查询最相似的前k个结果。
    doc_score = index.similarity_search_with_score(user_input, k=30)

    for doc, score in doc_score:
        doc.metadata['score'] = score
    docs = [doc for doc, score in doc_score]
    retrieval_documents = sorted(docs, key=lambda x: x.metadata['score'], reverse=True)

    reranker_model = RerankerModel(rerank_model_path)

    scores = reranker_model.score_pairs([(user_input, doc.page_content) for doc in retrieval_documents])
    for doc, score in zip(retrieval_documents, scores):
        doc.metadata['reranker_score'] = score.tolist()
    # 排序
    rerank_documents = sorted(retrieval_documents, key=lambda x: x.metadata['reranker_score'], reverse=True)
    # 删除分数小于0.35的文档
    rerank_documents = [doc for doc in rerank_documents if doc.metadata['reranker_score'] > 0.35]

    # 只拿前面7个
    rerank_documents = retrieval_documents[: 7]

    # 调用llm回答
    llm_client.query(prompt=';'.join(doc.page_content for doc in rerank_documents),
                     user_input=user_input)
