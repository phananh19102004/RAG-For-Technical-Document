from dotenv import load_dotenv
load_dotenv()

import os
import grpc
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from shared import rag_pb2, rag_pb2_grpc

RETRIEVAL_ADDR = os.getenv("RETRIEVAL_ADDR", "127.0.0.1:50051")
GENERATION_ADDR = os.getenv("GENERATION_ADDR", "127.0.0.1:50052")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask(payload: dict):
    question = (payload.get("question") or "").strip()
    if not question:
        return {"answer": "Don't have question"}

    # call Retrieval gRPC
    r_channel = grpc.insecure_channel(RETRIEVAL_ADDR)
    r_stub = rag_pb2_grpc.RetrievalServiceStub(r_channel)
    r_resp = r_stub.Retrieve(rag_pb2.RetrieveRequest(query=question, top_k=4), timeout=20)
    r_channel.close()

    # call Generation gRPC
    g_channel = grpc.insecure_channel(GENERATION_ADDR)
    g_stub = rag_pb2_grpc.GenerationServiceStub(g_channel)
    g_resp = g_stub.Generate(
        rag_pb2.GenerateRequest(question=question, chunks=r_resp.chunks),
        timeout=60,
    )
    g_channel.close()

    return {"answer": g_resp.answer}
