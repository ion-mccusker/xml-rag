from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from rag_pipeline import RAGPipeline
import tempfile
import os

app = FastAPI(title="XML RAG System", description="RAG system for XML documents")

templates = Jinja2Templates(directory="templates")

rag = RAGPipeline()

class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    documents = rag.list_documents()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "documents": documents,
        "document_count": len(documents)
    })

@app.post("/upload")
async def upload_xml(file: UploadFile = File(...)):
    if not file.filename.endswith('.xml'):
        raise HTTPException(status_code=400, detail="Only XML files are allowed")

    try:
        content = await file.read()
        xml_content = content.decode('utf-8')

        document_id = rag.add_xml_document(xml_content, file.filename)

        return {
            "message": "XML document uploaded successfully",
            "document_id": document_id,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing XML: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(query_request: QueryRequest):
    try:
        result = rag.query(
            question=query_request.question,
            n_results=query_request.max_results
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents")
async def list_documents():
    return {
        "documents": rag.list_documents(),
        "total_count": rag.get_document_count()
    }

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        rag.delete_document(document_id)
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)