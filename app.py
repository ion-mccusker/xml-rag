from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from rag_pipeline import RAGPipeline
from hf_rag_pipeline import HuggingFaceRAGPipeline
import tempfile
import os

app = FastAPI(title="Document RAG System", description="RAG system for XML, JSON, and text documents")

templates = Jinja2Templates(directory="templates")

# Initialize both pipelines
openai_rag = RAGPipeline()
try:
    hf_rag = HuggingFaceRAGPipeline(model_name="microsoft/DialoGPT-medium")
    hf_available = True
except Exception as e:
    print(f"HuggingFace pipeline unavailable: {e}")
    hf_rag = None
    hf_available = False

# Default to OpenAI pipeline
current_rag = openai_rag

class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5
    pipeline: Optional[str] = "openai"  # "openai" or "huggingface"

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    documents = current_rag.list_documents()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "documents": documents,
        "document_count": len(documents),
        "hf_available": hf_available
    })

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = ['.xml', '.json', '.txt', '.md', '.py', '.js', '.html', '.css', '.log', '.csv']

    if not any(file.filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Only these file types are allowed: {', '.join(allowed_extensions)}")

    try:
        content = await file.read()
        text_content = content.decode('utf-8')

        if file.filename.endswith('.xml'):
            document_id = current_rag.add_xml_document(text_content, file.filename)
            doc_type = "XML"
        elif file.filename.endswith('.json'):
            document_id = current_rag.add_json_document(text_content, file.filename)
            doc_type = "JSON"
        else:
            document_id = current_rag.add_text_document(text_content, file.filename)
            doc_type = "TEXT"

        return {
            "message": f"{doc_type} document uploaded successfully",
            "document_id": document_id,
            "filename": file.filename,
            "document_type": doc_type.lower()
        }

    except Exception as e:
        if file.filename.endswith('.xml'):
            file_type = "XML"
        elif file.filename.endswith('.json'):
            file_type = "JSON"
        else:
            file_type = "TEXT"
        raise HTTPException(status_code=500, detail=f"Error processing {file_type}: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(query_request: QueryRequest):
    try:
        # Select pipeline based on request
        if query_request.pipeline == "huggingface" and hf_available:
            selected_rag = hf_rag
        else:
            selected_rag = openai_rag

        result = selected_rag.query(
            question=query_request.question,
            n_results=query_request.max_results
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents")
async def list_documents():
    return {
        "documents": current_rag.list_documents(),
        "total_count": current_rag.get_document_count()
    }

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    try:
        document = current_rag.get_document_content(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        current_rag.delete_document(document_id)
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/pipelines/info")
async def get_pipeline_info():
    openai_info = {"name": "OpenAI", "type": "API", "available": True}
    hf_info = {"name": "HuggingFace", "type": "Local", "available": hf_available}
    if hf_available:
        hf_info.update(hf_rag.get_model_info())

    return {
        "pipelines": {
            "openai": openai_info,
            "huggingface": hf_info
        },
        "current_default": "openai"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)