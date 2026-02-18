## Live Demo : (https://phananh1910.xyz/)
The Qwen model is not included in the live demo. If you can't ask questions, it probably means the API key has run out of budget :).
The reference document can be found at " data/sample.pdf ". 
Some recommended questions based on the sample document :
- What is the main motivation behind Retrieval-Augmented Generation?
- What problem does RAG aim to solve in large language models?
- How does RAG reduce hallucination in LLMs?

![Demo Result](https://drive.google.com/uc?id=1CPwJFBkv5soQXqbw-IjJFkI2hhCgHixn)

## Project Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system using LLM GPT and Qwen2.5 to generate accurate, context-aware answers based on custom knowledge sources.
Instead of relying solely on a pretrained LLM, the system retrieves relevant documents from a vector database and injects them into the prompt to improve factual accuracy and domain-specific reasoning.

## Author  
**Phan Nguyễn Phan Anh**  
Final Year Student – USTH  

## Technologies Used :
- LLM: / Qwen / OpenAI API / Ollama
- FastAPI
- ChromaDB
- Langchain

To run local LLM Qwen2.5 . Please install Ollama ( https://docs.ollama.com/quickstart )
  
## System Architecture
![RAG System](https://drive.google.com/uc?id=1tOu0wm9IhF8O4oqK7LMGvaz4DUTy9krI)


