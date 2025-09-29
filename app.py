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

# Initialize pipelines with fallback logic
openai_rag = None
hf_rag = None
openai_available = False
hf_available = False
current_rag = None

# Try to initialize OpenAI pipeline first
try:
    openai_rag = RAGPipeline()
    openai_available = True
    current_rag = openai_rag
    logger.info("OpenAI RAG pipeline initialized successfully")
except Exception as e:
    logger.error(f"OpenAI pipeline initialization failed: {e}")
    print(f"OpenAI pipeline unavailable: {e}")

# Try to initialize HuggingFace pipeline
try:
    hf_rag = HuggingFaceRAGPipeline(model_name="microsoft/DialoGPT-medium")
    hf_available = True
    # If OpenAI failed, use HuggingFace as default
    if not openai_available:
        current_rag = hf_rag
        logger.info("Using HuggingFace pipeline as fallback")
    logger.info("HuggingFace RAG pipeline initialized successfully")
except Exception as e:
    logger.error(f"HuggingFace pipeline initialization failed: {e}")
    print(f"HuggingFace pipeline unavailable: {e}")

# Check if any pipeline is available
pipelines_available = openai_available or hf_available
if not pipelines_available:
    logger.error("CRITICAL: No RAG pipelines available - search and query functionality will be disabled")
    print("WARNING: No RAG pipelines available - search and query functionality will be disabled")

class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5
    pipeline: Optional[str] = "openai"  # "openai" or "huggingface"
    collection_name: Optional[str] = "documents"

class SearchRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5
    collection_name: Optional[str] = "documents"

class CollectionRequest(BaseModel):
    collection_name: str

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
    documents = current_rag.list_documents() if current_rag else []
    return templates.TemplateResponse("index.html", {
        "request": request,
        "documents": documents,
        "document_count": len(documents),
        "openai_available": openai_available,
        "hf_available": hf_available,
        "pipelines_available": pipelines_available
    })

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form("documents")
):
    if not current_rag:
        raise HTTPException(status_code=503, detail="No RAG pipeline available - upload functionality disabled")

    allowed_extensions = ['.xml', '.json', '.txt', '.md', '.py', '.js', '.html', '.css', '.log', '.csv']

    if not any(file.filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Only these file types are allowed: {', '.join(allowed_extensions)}")

    try:
        content = await file.read()
        text_content = content.decode('utf-8')

        if file.filename.endswith('.xml'):
            document_id = current_rag.add_xml_document(text_content, file.filename, collection_name)
            doc_type = "XML"
        elif file.filename.endswith('.json'):
            document_id = current_rag.add_json_document(text_content, file.filename, collection_name)
            doc_type = "JSON"
        else:
            document_id = current_rag.add_text_document(text_content, file.filename, collection_name)
            doc_type = "TEXT"

        return {
            "message": f"{doc_type} document uploaded successfully to collection '{collection_name}'",
            "document_id": document_id,
            "filename": file.filename,
            "document_type": doc_type.lower(),
            "collection_name": collection_name
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
    if not current_rag:
        raise HTTPException(status_code=503, detail="No RAG pipeline available - upload functionality disabled")

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
    if not pipelines_available:
        raise HTTPException(status_code=503, detail="No RAG pipelines available - query functionality disabled")

    try:
        # Select pipeline based on request
        if query_request.pipeline == "huggingface" and hf_available:
            selected_rag = hf_rag
        elif query_request.pipeline == "openai" and openai_available:
            selected_rag = openai_rag
        else:
            # Fallback to any available pipeline
            selected_rag = current_rag

        if not selected_rag:
            raise HTTPException(status_code=503, detail="Requested pipeline not available")

        result = selected_rag.query(
            question=query_request.question,
            n_results=query_request.max_results,
            collection_name=query_request.collection_name
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(search_request: SearchRequest):
    if not pipelines_available:
        raise HTTPException(status_code=503, detail="No RAG pipelines available - search functionality disabled")

    try:
        # Use current_rag for searching (both pipelines use same vector store)
        search_results = current_rag.search_documents(
            query=search_request.question,
            n_results=search_request.max_results,
            collection_name=search_request.collection_name
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
async def list_documents(
    page: int = 1,
    per_page: int = 10,
    search: Optional[str] = None,
    document_type: Optional[str] = None,
    collection_name: Optional[str] = None
):
    if not current_rag:
        return {
            "documents": [],
            "total_count": 0,
            "page": 1,
            "per_page": per_page,
            "total_pages": 0,
            "has_next": False,
            "has_prev": False
        }

    # Get all documents from specified collection (or default collection if None)
    all_documents = current_rag.list_documents(collection_name)

    # Apply filters
    filtered_documents = all_documents

    if search:
        search_lower = search.lower()
        filtered_documents = [
            doc for doc in filtered_documents
            if (search_lower in doc.get('filename', '').lower() or
                search_lower in doc.get('title', '').lower() or
                search_lower in str(doc.get('content', '')).lower())
        ]

    if document_type:
        filtered_documents = [
            doc for doc in filtered_documents
            if doc.get('document_type', '').lower() == document_type.lower()
        ]

    # Calculate pagination
    total_count = len(filtered_documents)
    total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_documents = filtered_documents[start_idx:end_idx]

    return {
        "documents": paginated_documents,
        "total_count": total_count,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
        "start_index": start_idx + 1 if paginated_documents else 0,
        "end_index": min(end_idx, total_count)
    }

@app.get("/documents/{document_id}")
async def get_document(document_id: str, collection_name: Optional[str] = None):
    if not current_rag:
        raise HTTPException(status_code=503, detail="No RAG pipeline available")

    try:
        document = current_rag.get_document_content(document_id, collection_name)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    if not current_rag:
        raise HTTPException(status_code=503, detail="No RAG pipeline available")

    try:
        current_rag.delete_document(document_id)
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/collections")
async def list_collections():
    if not current_rag:
        raise HTTPException(status_code=503, detail="No RAG pipeline available")

    try:
        collections = current_rag.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@app.post("/collections")
async def create_collection(collection_request: CollectionRequest):
    if not current_rag:
        raise HTTPException(status_code=503, detail="No RAG pipeline available")

    try:
        success = current_rag.create_collection(collection_request.collection_name)
        if success:
            return {
                "message": f"Collection '{collection_request.collection_name}' created successfully",
                "collection_name": collection_request.collection_name
            }
        else:
            raise HTTPException(status_code=400, detail=f"Collection '{collection_request.collection_name}' already exists")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")

@app.get("/pipelines/info")
async def get_pipeline_info():
    openai_info = {"name": "OpenAI", "type": "API", "available": openai_available}
    hf_info = {"name": "HuggingFace", "type": "Local", "available": hf_available}
    if hf_available and hf_rag:
        hf_info.update(hf_rag.get_model_info())

    current_default = "none"
    if openai_available:
        current_default = "openai"
    elif hf_available:
        current_default = "huggingface"

    return {
        "pipelines": {
            "openai": openai_info,
            "huggingface": hf_info
        },
        "current_default": current_default,
        "pipelines_available": pipelines_available
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