from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import text, PyPDFLoader, WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings


class KnowledgeBase:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=50
        )
        self.retriever = None

    def load_txt(self, path):
        loader = text.TextLoader(file_path=path)
        documents = loader.load()
        chunked_docs = self.text_splitter.split_documents(documents=documents)
        print(chunked_docs.__len__())
        db = FAISS.from_documents(embedding=self.embeddings, documents=chunked_docs)
        self.retriever = db.as_retriever()

    def load_pdf(self, path):
        loader = PyPDFLoader(file_path=path)
        documents = loader.load()
        chunked_docs = self.text_splitter.split_documents(documents=documents)
        db = FAISS.from_documents(embedding=self.embeddings, documents=chunked_docs)
        self.retriever = db.as_retriever()

    def load_url(self, path):
        loader = WebBaseLoader(web_path=path)
        docs = loader.load()
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        chunked_docs = self.text_splitter.split_documents(docs_transformed)
        db = FAISS.from_documents(embedding=self.embeddings, documents=chunked_docs)
        self.retriever = db.as_retriever()

    def invoke(self, query):
        return self.retriever.invoke(query)
