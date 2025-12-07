from typing import List, Dict
from logger_config import get_logger

class RetrieverTool:
    """Simple retriever tool wrapper with logging"""
    def __init__(self, retriever):
        self.retriever = retriever
        self.name = "medical_knowledge_retriever"
        self.description = "Retrieves relevant medical information from the knowledge base"
        self.logger = get_logger('retriever')
    
    def run(self, query: str) -> List[Dict]:
        """Execute retriever and return documents"""
        self.logger.info(f"Retrieval query: '{query}'")
        
        try:
            docs = self.retriever.invoke(query)
            result = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
            
            self.logger.info(f"Retrieved {len(result)} documents")
            self.logger.debug(f"Document IDs: {[doc.metadata.get('source', 'unknown') for doc in docs]}")
            return result
        except Exception as e:
            self.logger.error(f"Retrieval failed: {str(e)}", exc_info=True)
            raise