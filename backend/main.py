import os
import shutil
import time
import nest_asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse

# Apply async fixes
nest_asyncio.apply()

# Initialize App
app = FastAPI()

# Allow the frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Storage 
index = None
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

# Setup Models
Settings.llm = Gemini(model_name="models/gemini-2.5-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 15

@app.get("/")
def home():
    return {"status": "Active", "message": "RAG Backend is running"}

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    global index
    
    # Save uploaded files
    saved_files = []
    for file in files:
        file_path = os.path.join(data_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)

    # Trigger Parsing WITH COLAB COOLDOWN LOGIC
    try:
        documents = []
        for i, file_name in enumerate(saved_files):
            file_path = os.path.join(data_dir, file_name)
            
            # Give background system a fresh start for every file
            current_parser = LlamaParse(result_type="markdown", verbose=True, language="en")
            file_extractor = {".pdf": current_parser}

            try:
                single_doc = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
                documents.extend(single_doc)
                
                # 10-second delay to avoid unwanted pulling data issues
                time.sleep(10)
            except Exception as parse_error:
                print(f"Error on {file_name}: {parse_error}")
                time.sleep(5)
                continue
        
        # Build/Update Index
        index = VectorStoreIndex.from_documents(documents)
        return {"message": f"Successfully processed {len(saved_files)} files.", "total_chunks": len(documents)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_index(request: QueryRequest):
    global index
    if index is None:
        raise HTTPException(status_code=400, detail="Index not built. Upload documents first.")
    
    # Utilize the dynamic top_k passed from the frontend
    chat_engine = index.as_chat_engine(chat_mode="condense_question", similarity_top_k=request.top_k)
    response = chat_engine.chat(request.query)
    
    # Extract sources
    sources = []
    for node in response.source_nodes:
        sources.append({
            "file": node.metadata.get("file_name", "Unknown"),
            "page": node.metadata.get("page_label", "N/A"),
            "text": node.node.text[:200] + "..."
        })
        
    return {"response": response.response, "sources": sources}