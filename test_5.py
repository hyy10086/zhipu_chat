import os

from docx import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypinyin import pinyin, Style

from LlmClient import LlmClient
from Configs import filepath, embedding_path

# 第一步：割分文档
# 打开Word文档
doc = Document(filepath)

# 读取文档内容
content = ' '.join([paragraph.text for paragraph in doc.paragraphs])

# 创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
# 分割文本
splits = text_splitter.split_text(content)

# 第二步：文档嵌入
# 获取文件名
file_name = os.path.basename(filepath)
# 将文件名转换为拼音
pinyin_names = pinyin(file_name, style=Style.NORMAL)
# 生成数据库id
kb_id = ''.join([item[0] for item in pinyin_names]).replace('.', '_')

faiss_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), kb_id, 'faiss_index')

# 创建嵌入模型
embeddings = HuggingFaceEmbeddings(model_name=embedding_path)

if os.path.exists(faiss_index_path):
    print("Index loaded from：", faiss_index_path)
    index = FAISS.load_local(folder_path=faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
else:
    # 创建索引
    index = FAISS.from_texts(
        texts=[doc for doc in splits],
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
    result_list = index.similarity_search(user_input, k=3)
    llm_client.query(prompt=';'.join(doc.page_content for doc in result_list),
                     user_input=user_input)
