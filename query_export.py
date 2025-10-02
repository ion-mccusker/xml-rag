import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import os


class QueryExporter:
    def __init__(self, export_directory: str = "./query_exports"):
        self.export_directory = Path(export_directory)
        self.export_directory.mkdir(exist_ok=True)

    def export_query_results(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        ai_answer: Optional[str] = None,
        collection_info: Dict[str, Any] = None,
        chunking_config: Dict[str, Any] = None,
        search_params: Dict[str, Any] = None,
        performance_metrics: Dict[str, Any] = None,
        user_notes: str = "",
        response_time_ms: Optional[int] = None,
        model_used: Optional[str] = None
    ) -> str:
        """
        Export query results with comprehensive metadata for comparison analysis

        Returns the export file path
        """
        export_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Build export data structure
        export_data = {
            "export_metadata": {
                "timestamp": timestamp,
                "query": query,
                "export_id": export_id,
                "user_notes": user_notes,
                "export_version": "1.0"
            },
            "search_configuration": {
                "collection_name": collection_info.get("collection_name", "unknown") if collection_info else "unknown",
                "embedding_model": collection_info.get("embedding_model", "unknown") if collection_info else "unknown",
                "embedding_model_name": collection_info.get("embedding_model_name", "Unknown") if collection_info else "Unknown",
                "embedding_model_description": collection_info.get("embedding_model_description", "") if collection_info else "",
                "chunking_config": chunking_config or {},
                "search_params": search_params or {}
            },
            "results": {
                "total_results": len(search_results),
                "search_results": self._format_search_results(search_results),
                "ai_answer": ai_answer,
                "model_used": model_used,
                "response_time_ms": response_time_ms
            },
            "performance_metrics": performance_metrics or {}
        }

        # Generate filename with timestamp and query snippet
        query_snippet = "".join(c for c in query[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_snippet = query_snippet.replace(' ', '_')
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"query_export_{timestamp_str}_{query_snippet}_{export_id[:8]}.json"

        file_path = self.export_directory / filename

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return str(file_path)

    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format search results for export with consistent structure"""
        formatted_results = []

        for i, result in enumerate(search_results):
            formatted_result = {
                "rank": i + 1,
                "filename": result.get("filename", "unknown"),
                "document_type": result.get("document_type", "unknown"),
                "chunk_index": result.get("chunk_index", 0),
                "relevance_score": result.get("relevance_score", 0.0),
                "distance": result.get("distance", 1.0),
                "content_preview": result.get("content_preview", ""),
                "full_content": result.get("content", result.get("full_content", "")),
                "document_id": result.get("document_id", ""),
                "metadata": result.get("metadata", {})
            }
            formatted_results.append(formatted_result)

        return formatted_results

    def list_exports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent exports with metadata"""
        exports = []

        for file_path in sorted(self.export_directory.glob("query_export_*.json"), reverse=True):
            if len(exports) >= limit:
                break

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                export_info = {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "export_id": data.get("export_metadata", {}).get("export_id", ""),
                    "timestamp": data.get("export_metadata", {}).get("timestamp", ""),
                    "query": data.get("export_metadata", {}).get("query", "")[:100],
                    "collection": data.get("search_configuration", {}).get("collection_name", "unknown"),
                    "embedding_model": data.get("search_configuration", {}).get("embedding_model_name", "Unknown"),
                    "total_results": data.get("results", {}).get("total_results", 0),
                    "user_notes": data.get("export_metadata", {}).get("user_notes", ""),
                    "file_size_kb": round(file_path.stat().st_size / 1024, 2)
                }
                exports.append(export_info)

            except Exception as e:
                print(f"Error reading export file {file_path}: {e}")
                continue

        return exports

    def load_export(self, export_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific export by ID"""
        for file_path in self.export_directory.glob("query_export_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if data.get("export_metadata", {}).get("export_id") == export_id:
                    return data

            except Exception as e:
                continue

        return None

    def delete_export(self, export_id: str) -> bool:
        """Delete an export by ID"""
        for file_path in self.export_directory.glob("query_export_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if data.get("export_metadata", {}).get("export_id") == export_id:
                    file_path.unlink()
                    return True

            except Exception as e:
                continue

        return False

    def generate_comparison_report(self, export_ids: List[str]) -> Dict[str, Any]:
        """Generate a comparison report between multiple exports"""
        exports = []
        for export_id in export_ids:
            export_data = self.load_export(export_id)
            if export_data:
                exports.append(export_data)

        if len(exports) < 2:
            return {"error": "Need at least 2 exports for comparison"}

        comparison = {
            "comparison_metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "exports_compared": len(exports),
                "comparison_id": str(uuid.uuid4())
            },
            "query_comparison": {
                "base_query": exports[0]["export_metadata"]["query"],
                "queries_match": all(exp["export_metadata"]["query"] == exports[0]["export_metadata"]["query"] for exp in exports)
            },
            "configuration_comparison": [
                {
                    "export_id": exp["export_metadata"]["export_id"],
                    "embedding_model": exp["search_configuration"]["embedding_model_name"],
                    "chunk_size": exp["search_configuration"]["chunking_config"].get("chunk_size", "unknown"),
                    "chunk_overlap": exp["search_configuration"]["chunking_config"].get("chunk_overlap", "unknown"),
                    "collection": exp["search_configuration"]["collection_name"],
                    "total_results": exp["results"]["total_results"],
                    "response_time_ms": exp["results"].get("response_time_ms", "unknown")
                }
                for exp in exports
            ],
            "result_analysis": self._analyze_result_differences(exports)
        }

        return comparison

    def _analyze_result_differences(self, exports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze differences in search results between exports"""
        analysis = {
            "relevance_score_comparison": [],
            "result_overlap": [],
            "unique_results_per_export": [],
            "detailed_results_comparison": []
        }

        # Compare relevance scores for top results
        for i, export in enumerate(exports):
            results = export["results"]["search_results"]
            if results:
                avg_relevance = sum(r.get("relevance_score", 0) for r in results) / len(results)
                top_relevance = max(r.get("relevance_score", 0) for r in results)

                analysis["relevance_score_comparison"].append({
                    "export_id": export["export_metadata"]["export_id"],
                    "embedding_model": export["search_configuration"]["embedding_model_name"],
                    "average_relevance": round(avg_relevance, 3),
                    "top_relevance": round(top_relevance, 3),
                    "total_results": len(results)
                })

        # Detailed results comparison
        if len(exports) >= 2:
            export1, export2 = exports[0], exports[1]
            results1 = export1["results"]["search_results"]
            results2 = export2["results"]["search_results"]

            # Create detailed comparison
            detailed_comparison = {
                "export1": {
                    "export_id": export1["export_metadata"]["export_id"],
                    "model": export1["search_configuration"]["embedding_model_name"],
                    "results": []
                },
                "export2": {
                    "export_id": export2["export_metadata"]["export_id"],
                    "model": export2["search_configuration"]["embedding_model_name"],
                    "results": []
                },
                "common_results": [],
                "unique_to_export1": [],
                "unique_to_export2": []
            }

            # Process results for detailed comparison
            results1_lookup = {f"{r['filename']}_{r['chunk_index']}": r for r in results1}
            results2_lookup = {f"{r['filename']}_{r['chunk_index']}": r for r in results2}

            # Find common and unique results
            keys1 = set(results1_lookup.keys())
            keys2 = set(results2_lookup.keys())

            common_keys = keys1.intersection(keys2)
            unique_keys1 = keys1 - keys2
            unique_keys2 = keys2 - keys1

            # Common results with score comparison
            for key in common_keys:
                r1, r2 = results1_lookup[key], results2_lookup[key]
                detailed_comparison["common_results"].append({
                    "filename": r1["filename"],
                    "chunk_index": r1["chunk_index"],
                    "export1_relevance": r1.get("relevance_score", 0),
                    "export2_relevance": r2.get("relevance_score", 0),
                    "relevance_diff": round(abs(r1.get("relevance_score", 0) - r2.get("relevance_score", 0)), 3),
                    "export1_rank": r1["rank"],
                    "export2_rank": r2["rank"],
                    "content_preview": r1.get("content_preview", r1.get("full_content", ""))[:150] + "..."
                })

            # Unique to export1
            for key in unique_keys1:
                r = results1_lookup[key]
                detailed_comparison["unique_to_export1"].append({
                    "filename": r["filename"],
                    "chunk_index": r["chunk_index"],
                    "relevance_score": r.get("relevance_score", 0),
                    "rank": r["rank"],
                    "content_preview": r.get("content_preview", r.get("full_content", ""))[:150] + "..."
                })

            # Unique to export2
            for key in unique_keys2:
                r = results2_lookup[key]
                detailed_comparison["unique_to_export2"].append({
                    "filename": r["filename"],
                    "chunk_index": r["chunk_index"],
                    "relevance_score": r.get("relevance_score", 0),
                    "rank": r["rank"],
                    "content_preview": r.get("content_preview", r.get("full_content", ""))[:150] + "..."
                })

            # Sort results by relevance score
            detailed_comparison["common_results"].sort(key=lambda x: max(x["export1_relevance"], x["export2_relevance"]), reverse=True)
            detailed_comparison["unique_to_export1"].sort(key=lambda x: x["relevance_score"], reverse=True)
            detailed_comparison["unique_to_export2"].sort(key=lambda x: x["relevance_score"], reverse=True)

            analysis["detailed_results_comparison"] = detailed_comparison

        # Analyze result overlap between exports
        if len(exports) >= 2:
            for i in range(len(exports)):
                for j in range(i + 1, len(exports)):
                    export1 = exports[i]
                    export2 = exports[j]

                    results1 = {r["filename"] + "_" + str(r["chunk_index"]) for r in export1["results"]["search_results"]}
                    results2 = {r["filename"] + "_" + str(r["chunk_index"]) for r in export2["results"]["search_results"]}

                    overlap = len(results1.intersection(results2))
                    total_unique = len(results1.union(results2))
                    overlap_percentage = (overlap / total_unique * 100) if total_unique > 0 else 0

                    analysis["result_overlap"].append({
                        "export1_id": export1["export_metadata"]["export_id"],
                        "export2_id": export2["export_metadata"]["export_id"],
                        "model1": export1["search_configuration"]["embedding_model_name"],
                        "model2": export2["search_configuration"]["embedding_model_name"],
                        "overlapping_results": overlap,
                        "total_unique_results": total_unique,
                        "overlap_percentage": round(overlap_percentage, 1)
                    })

        return analysis