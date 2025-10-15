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
from query_export import QueryExporter
import tempfile
import os
import logging
import time


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI(title="Document RAG System", description="RAG system for XML, JSON, and text documents")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Initialize pipelines with fallback logic
openai_rag = None
hf_rag = None
openai_available = False
hf_available = False
current_rag = None

# Initialize query exporter
query_exporter = QueryExporter()

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
    use_reranker: Optional[bool] = False

class SearchRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5
    collection_name: Optional[str] = "documents"
    use_reranker: Optional[bool] = False

class CollectionRequest(BaseModel):
    collection_name: str
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str

class SearchResponse(BaseModel):
    results: List[dict]
    query: str
    total_results: int

class ExportRequest(BaseModel):
    query: str
    search_results: List[dict]
    ai_answer: Optional[str] = None
    collection_name: Optional[str] = None
    chunking_config: Optional[dict] = None
    search_params: Optional[dict] = None
    user_notes: Optional[str] = ""
    model_used: Optional[str] = None
    response_time_ms: Optional[int] = None

class ComparisonRequest(BaseModel):
    export_ids: List[str]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if current_rag:
        # Get first page of documents for display
        document_data = current_rag.list_documents(collection_name=None, page=1, per_page=10)
        documents = document_data.get("documents", [])
        total_document_count = document_data.get("total_count", 0)
    else:
        documents = []
        total_document_count = 0

    return templates.TemplateResponse("index.html", {
        "request": request,
        "documents": documents,
        "document_count": total_document_count,
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
        # Get collection-specific chunking configuration
        collection_info = current_rag.get_collection_info(collection_name)
        chunk_size = collection_info.get("chunk_size", 1000)
        chunk_overlap = collection_info.get("chunk_overlap", 200)

        content = await file.read()
        text_content = content.decode('utf-8')

        if file.filename.endswith('.xml'):
            document_id = current_rag.add_xml_document(text_content, file.filename, collection_name, chunk_size, chunk_overlap)
            doc_type = "XML"
        elif file.filename.endswith('.json'):
            document_id = current_rag.add_json_document(text_content, file.filename, collection_name, chunk_size, chunk_overlap)
            doc_type = "JSON"
        else:
            document_id = current_rag.add_text_document(text_content, file.filename, collection_name, chunk_size, chunk_overlap)
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
async def upload_batch_documents(
    files: List[UploadFile] = File(...),
    collection_name: Optional[str] = Form("documents")
):
    if not current_rag:
        raise HTTPException(status_code=503, detail="No RAG pipeline available - upload functionality disabled")

    # Get collection-specific chunking configuration
    collection_info = current_rag.get_collection_info(collection_name)
    chunk_size = collection_info.get("chunk_size", 1000)
    chunk_overlap = collection_info.get("chunk_overlap", 200)

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
                document_id = current_rag.add_xml_document(text_content, file.filename, collection_name, chunk_size, chunk_overlap)
                doc_type = "XML"
            elif file.filename.endswith('.json'):
                document_id = current_rag.add_json_document(text_content, file.filename, collection_name, chunk_size, chunk_overlap)
                doc_type = "JSON"
            else:
                document_id = current_rag.add_text_document(text_content, file.filename, collection_name, chunk_size, chunk_overlap)
                doc_type = "TEXT"

            result["success"] = True
            result["document_id"] = document_id
            result["document_type"] = doc_type.lower()
            result["message"] = f"{doc_type} document uploaded successfully"
            result["collection_name"] = collection_name
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
        start_time = time.time()

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

        # Get collection info for export metadata
        collection_info = selected_rag.get_collection_info(query_request.collection_name)

        # Perform the query
        result = selected_rag.query(
            question=query_request.question,
            n_results=query_request.max_results,
            collection_name=query_request.collection_name,
            use_reranker=query_request.use_reranker
        )

        # Calculate response time
        end_time = time.time()
        response_time_ms = int((end_time - start_time) * 1000)

        # Add export metadata to result
        result["export_metadata"] = {
            "collection_info": collection_info,
            "search_params": {
                "max_results": query_request.max_results,
                "pipeline_used": query_request.pipeline,
                "collection_name": query_request.collection_name
            },
            "response_time_ms": response_time_ms,
            "timestamp": time.time()
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(search_request: SearchRequest):
    if not pipelines_available:
        raise HTTPException(status_code=503, detail="No RAG pipelines available - search functionality disabled")

    try:
        start_time = time.time()

        # Get collection info for export metadata
        collection_info = current_rag.get_collection_info(search_request.collection_name)

        # Use current_rag for searching (both pipelines use same vector store)
        search_results = current_rag.search_documents(
            query=search_request.question,
            n_results=search_request.max_results,
            collection_name=search_request.collection_name,
            use_reranker=search_request.use_reranker
        )

        # Calculate response time
        end_time = time.time()
        response_time_ms = int((end_time - start_time) * 1000)

        # Format results for response
        formatted_results = []
        for result in search_results:
            formatted_result = {
                "filename": result["metadata"].get("filename", "unknown"),
                "document_type": result["metadata"].get("document_type", "unknown"),
                "chunk_index": result["metadata"].get("chunk_index", 0),
                "distance": result["distance"],
                "relevance_score": round(1 - (result["distance"] / 2), 3),
                "content": result["document"],
                "full_content": result["document"],
                "content_preview": result["document"][:200] + "..." if len(result["document"]) > 200 else result["document"],
                "document_id": result["metadata"].get("document_id", ""),
                "collection_name": result["metadata"].get("collection_name", search_request.collection_name),
                "chunk_length": result["metadata"].get("chunk_length", len(result["document"])),
                "metadata": result["metadata"]
            }

            # Add all available metadata fields to top level for easy access
            # Exclude core search result fields to avoid conflicts
            core_result_fields = {
                "filename", "document_type", "chunk_index", "distance", "relevance_score",
                "content", "full_content", "content_preview", "document_id", "collection_name", "chunk_length", "metadata"
            }

            for field, value in result["metadata"].items():
                if field not in core_result_fields:
                    formatted_result[field] = value

            formatted_results.append(formatted_result)

        # Create response with export metadata
        response = SearchResponse(
            results=formatted_results,
            query=search_request.question,
            total_results=len(formatted_results)
        )

        # Add export metadata (convert to dict to add extra fields)
        response_dict = response.dict()
        response_dict["export_metadata"] = {
            "collection_info": collection_info,
            "search_params": {
                "max_results": search_request.max_results,
                "collection_name": search_request.collection_name
            },
            "response_time_ms": response_time_ms,
            "timestamp": time.time()
        }

        return response_dict

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

    # Use the new filesystem-based document listing with pagination
    return current_rag.list_documents(
        collection_name=collection_name,
        page=page,
        per_page=per_page,
        search=search,
        document_type=document_type
    )

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
        success = current_rag.create_collection(
            collection_request.collection_name,
            collection_request.embedding_model,
            collection_request.chunk_size,
            collection_request.chunk_overlap
        )
        if success:
            return {
                "message": f"Collection '{collection_request.collection_name}' created successfully",
                "collection_name": collection_request.collection_name,
                "embedding_model": collection_request.embedding_model,
                "chunk_size": collection_request.chunk_size,
                "chunk_overlap": collection_request.chunk_overlap
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

@app.get("/embedding-models")
async def get_embedding_models():
    if not current_rag:
        raise HTTPException(status_code=503, detail="No RAG pipeline available")

    try:
        models = current_rag.get_available_embedding_models()
        return {"embedding_models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embedding models: {str(e)}")

@app.get("/collections/{collection_name}")
async def get_collection_info(collection_name: str):
    if not current_rag:
        raise HTTPException(status_code=503, detail="No RAG pipeline available")

    try:
        info = current_rag.get_collection_info(collection_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")

@app.get("/collections/{collection_name}/info")
async def get_collection_info_legacy(collection_name: str):
    """Legacy endpoint for backward compatibility"""
    return await get_collection_info(collection_name)

@app.post("/export-query")
async def export_query_results(export_request: ExportRequest):
    if not current_rag:
        raise HTTPException(status_code=503, detail="No RAG pipeline available")

    try:
        # Get collection info and chunking config
        collection_info = current_rag.get_collection_info(export_request.collection_name)

        # Build chunking config from request or defaults
        chunking_config = export_request.chunking_config or {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "text_splitter": "RecursiveCharacterTextSplitter"
        }

        # Export the query results
        file_path = query_exporter.export_query_results(
            query=export_request.query,
            search_results=export_request.search_results,
            ai_answer=export_request.ai_answer,
            collection_info=collection_info,
            chunking_config=chunking_config,
            search_params=export_request.search_params,
            user_notes=export_request.user_notes,
            response_time_ms=export_request.response_time_ms,
            model_used=export_request.model_used
        )

        return {
            "message": "Query results exported successfully",
            "file_path": file_path,
            "export_id": file_path.split("_")[-1].replace(".json", "")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting query results: {str(e)}")

@app.get("/exports")
async def list_exports(limit: int = 50):
    try:
        exports = query_exporter.list_exports(limit)
        return {"exports": exports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing exports: {str(e)}")

@app.get("/exports/{export_id}")
async def get_export(export_id: str):
    try:
        export_data = query_exporter.load_export(export_id)
        if not export_data:
            raise HTTPException(status_code=404, detail="Export not found")
        return export_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading export: {str(e)}")

@app.delete("/exports/{export_id}")
async def delete_export(export_id: str):
    try:
        success = query_exporter.delete_export(export_id)
        if not success:
            raise HTTPException(status_code=404, detail="Export not found")
        return {"message": "Export deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting export: {str(e)}")

@app.post("/exports/compare")
async def compare_exports(comparison_request: ComparisonRequest):
    try:
        if len(comparison_request.export_ids) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 export IDs for comparison")

        comparison = query_exporter.generate_comparison_report(comparison_request.export_ids)

        if "error" in comparison:
            raise HTTPException(status_code=400, detail=comparison["error"])

        return comparison
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing exports: {str(e)}")

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    if not current_rag:
        raise HTTPException(status_code=503, detail="No RAG pipeline available")

    try:
        success = current_rag.delete_collection(collection_name)
        if success:
            return {"message": f"Collection '{collection_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # reload=reload,
        log_level="info",
        access_log=True
    )