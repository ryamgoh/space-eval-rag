class RAGManager:
    def __init__(self, vector_store_path: str, embedding_model: str):
        self.vector_store = FAISS.load_local(...)
        self.retriever = LangChainRetriever(...)
    
    def augment_prompt(prompt: str, context_k: int = 3) -> str
    def add_to_knowledge_base(documents: List[str])