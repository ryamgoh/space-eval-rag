class RAGManager:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RAG is currently disabled in this pipeline.")

    def augment_prompt(self, prompt: str, context_k: int = 3) -> str:
        raise NotImplementedError

    def add_to_knowledge_base(self, documents: list[str]) -> None:
        raise NotImplementedError
