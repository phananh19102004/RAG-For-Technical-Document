
from dotenv import load_dotenv
load_dotenv()


import os
from concurrent import futures
import grpc

from shared import rag_pb2, rag_pb2_grpc

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = os.getenv("PDF_PATH", "data/sample.pdf")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "4"))

class RetrievalServicer(rag_pb2_grpc.RetrievalServiceServicer):
    def __init__(self):
        
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"[Retrieval] Loaded {len(documents)} pages from PDF: {PDF_PATH}")

        # CHUNKING
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        docs = splitter.split_documents(documents)
        print(f"[Retrieval] Total chunks: {len(docs)}")

        # EMBEDDINGS + VECTOR DB 
        embeddings = OpenAIEmbeddings()  
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K_DEFAULT})
        print(f"[Retrieval] Chroma persisted at: {CHROMA_DIR}")

    def Retrieve(self, request, context):
        query = (request.query or "").strip()
        if not query:
            return rag_pb2.RetrieveResponse(chunks=[])

        top_k = request.top_k if request.top_k > 0 else TOP_K_DEFAULT
        
        docs = self.vectorstore.as_retriever(search_kwargs={"k": top_k}).invoke(query)

        chunks = []
        for d in docs:
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", None)
            if page is not None:
                src = f"{src}#page={page}"
            chunks.append(rag_pb2.Chunk(source=str(src), text=d.page_content))

        return rag_pb2.RetrieveResponse(chunks=chunks)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rag_pb2_grpc.add_RetrievalServiceServicer_to_server(RetrievalServicer(), server)
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    print("gRPC listening on :50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
