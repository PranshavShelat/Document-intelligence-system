# Document-intelligence-system

A full-stack Retrieval-Augmented Generation (RAG) application built to parse, embed, and query complex academic PDFs. I developed this to solve the friction of navigating dense research papers. 

The system uses a custom asynchronous backend to handle document ingestion without hitting API rate limits, and allows users to dynamically scale the number of retrieved chunks for more accurate synthesis.

## Live Demo
**Frontend:** https://699c7d633c50691ef85059b8--frolicking-melomakarona-e3a30c.netlify.app/
**Backend API:** Hosted on Hugging Face Spaces (Dockerized FastAPI)

## Tech Stack
* **Backend:** FastAPI, Python, Docker
* **AI/ML:** LlamaIndex, Gemini 2.5 Flash, BAAI/bge-small-en-v1.5 (Hugging Face)
* **Document Parsing:** LlamaParse (with custom rate-limit handling)
* **Frontend:** HTML5, JavaScript, Tailwind CSS

## Key Features
* **Hybrid RAG Architecture:** Combines fast Hugging Face vector embeddings with Gemini's reasoning capabilities.
* **Fault-Tolerant Ingestion:** Built-in async cooldowns and error handling to ensure large batches of PDFs process without crashing.
* **Dynamic Context Scaling:** Users can adjust the `top_k` retrieval parameter directly from the UI to control how many references the AI reads before answering.
