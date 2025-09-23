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
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

class SearchRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str

class SearchResponse(BaseModel):
    results: List[dict]
    query: str
    total_results: int

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

@app.post("/upload-batch")
async def upload_batch_documents(files: List[UploadFile] = File(...)):
    allowed_extensions = ['.xml', '.json', '.txt', '.md', '.py', '.js', '.html', '.css', '.log', '.csv']
    results = []
    successful_uploads = 0
    failed_uploads = 0

    for file in files:
        result = {"filename": file.filename}

        try:
            # Validate file extension
            if not any(file.filename.endswith(ext) for ext in allowed_extensions):
                result["success"] = False
                result["error"] = f"File type not allowed. Only these file types are supported: {', '.join(allowed_extensions)}"
                failed_uploads += 1
                results.append(result)
                continue

            # Read and process file
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

            result["success"] = True
            result["document_id"] = document_id
            result["document_type"] = doc_type.lower()
            result["message"] = f"{doc_type} document uploaded successfully"
            successful_uploads += 1

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            failed_uploads += 1

        results.append(result)

    return {
        "total_files": len(files),
        "successful_uploads": successful_uploads,
        "failed_uploads": failed_uploads,
        "results": results,
        "message": f"Batch upload completed: {successful_uploads} successful, {failed_uploads} failed"
    }

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

@app.post("/search", response_model=SearchResponse)
async def search_documents(search_request: SearchRequest):
    try:
        # Use current_rag for searching (both pipelines use same vector store)
        search_results = current_rag.search_documents(
            query=search_request.question,
            n_results=search_request.max_results
        )

        # Format results for response
        formatted_results = [
            {
                "filename": result["metadata"].get("filename", "unknown"),
                "document_type": result["metadata"].get("document_type", "unknown"),
                "chunk_index": result["metadata"].get("chunk_index", 0),
                "distance": result["distance"],
                "relevance_score": round(1 - result["distance"], 3),
                "content": result["document"],
                "content_preview": result["document"][:200] + "..." if len(result["document"]) > 200 else result["document"],
                "document_id": result["metadata"].get("document_id", ""),
                "metadata": result["metadata"]
            }
            for result in search_results
        ]

        return SearchResponse(
            results=formatted_results,
            query=search_request.question,
            total_results=len(formatted_results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")

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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # reload=reload,
        log_level="info",
        access_log=True
    )