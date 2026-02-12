from dotenv import load_dotenv
load_dotenv()

import os
import json
from concurrent import futures
import grpc
from urllib import request as urlrequest

from shared import rag_pb2, rag_pb2_grpc

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Provider: "openai" or "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_0")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", str(TEMPERATURE)))

prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.
Answer the question using ONLY the information provided in the context.
If the answer cannot be found in the context, say exactly:
"I cannot find the answer in the document."

Context:
{context}

Question:
{question}
""".strip())

def format_chunks(chunks):
    parts = []
    for c in chunks:
        parts.append(f"SOURCE: {c.source}\n{c.text}")
    return "\n\n---\n\n".join(parts)

def call_ollama_chat(full_prompt: str) -> str:
    """
    Call Ollama Chat API: POST {OLLAMA_HOST}/api/chat
    No extra dependency (urllib).
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": False,
        "options": {"temperature": OLLAMA_TEMPERATURE},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlrequest.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
    obj = json.loads(raw)
    # Ollama returns: {"message": {"role": "...", "content": "..."} , ...}
    return (obj.get("message") or {}).get("content", "").strip()

class GenerationServicer(rag_pb2_grpc.GenerationServiceServicer):
    def __init__(self):
        self.provider = LLM_PROVIDER

        if self.provider == "openai":
            self.llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)
            self.chain = prompt | self.llm | StrOutputParser()
            print(f"[Generation] Provider=OPENAI model={MODEL}, temperature={TEMPERATURE}")
        elif self.provider == "ollama":
            self.chain = None
            print(f"[Generation] Provider=OLLAMA model={OLLAMA_MODEL}, host={OLLAMA_HOST}, temperature={OLLAMA_TEMPERATURE}")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER={self.provider}. Use openai|ollama")

    def Generate(self, request, context):
        question = (request.question or "").strip()
        if not question:
            return rag_pb2.GenerateResponse(answer="You haven't entered a question yet.")

        ctx_text = format_chunks(request.chunks)

        if self.provider == "openai":
            answer = self.chain.invoke({"context": ctx_text, "question": question})
            return rag_pb2.GenerateResponse(answer=answer)

        # provider == "ollama"
        full_prompt = prompt.format(context=ctx_text, question=question)
        answer = call_ollama_chat(full_prompt)
        if not answer:
            answer = "(ollama returned empty response)"
        return rag_pb2.GenerateResponse(answer=answer)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rag_pb2_grpc.add_GenerationServiceServicer_to_server(GenerationServicer(), server)
    server.add_insecure_port("0.0.0.0:50052")
    server.start()
    print("gRPC listening on :50052")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
