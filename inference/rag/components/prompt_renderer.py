from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from inference.task.processor import TaskProcessor


class PromptRenderer:
    """Render prompts with retrieved context."""
    def __init__(self, task_cfg: Mapping[str, Any], rag_cfg: Mapping[str, Any]):
        self._task_cfg = task_cfg
        self._rag_cfg = rag_cfg

    def render_prompt(
        self,
        row: Mapping[str, Any],
        context: str,
        query: str,
        retrieved: Iterable[Mapping[str, Any]],
    ) -> tuple[str, Mapping[str, Any]]:
        """Render a prompt and its associated RAG metadata."""
        prompt = TaskProcessor.render_template(
            row,
            self._task_cfg["prompt_template"],
            self._task_cfg["input_mappings"],
            extras={"context": context},
        )
        extra = {
            "rag": {
                "query": query,
                "context": context,
                "results": list(retrieved),
            }
        }
        return prompt, extra

    @staticmethod
    def format_context(
        retrieved: Iterable[Mapping[str, Any]],
        template: str,
        separator: str,
    ) -> str:
        """Format retrieved items into a single context string."""
        rendered: List[str] = []
        for item in retrieved:
            rendered.append(template.format_map(item))
        return separator.join(rendered)
