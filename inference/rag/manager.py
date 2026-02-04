class RAGManager:
    """Placeholder for a retrieval-augmented generation manager."""
    def __init__(self, *args, **kwargs):
        """Initialize RAG components (disabled)."""
        raise NotImplementedError("RAG is currently disabled in this pipeline.")

    def augment_prompt(self, prompt: str, context_k: int = 3) -> str:
        """Augment a prompt with retrieved context (disabled)."""
        raise NotImplementedError

    def add_to_knowledge_base(self, documents: list[str]) -> None:
        """Add documents to the knowledge base (disabled)."""
        raise NotImplementedError
