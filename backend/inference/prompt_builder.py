from typing import Optional, List, Dict


class PromptBuilder:
    """
    Builds structured prompts aligned with training format.
    Supports instruction, context, and conversation history.
    """

    def __init__(self):
        self.template = (
            "Instruction: {instruction}\n"
            "Context: {context}\n"
            "Output:"
        )

    def build(
        self,
        instruction: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        context_parts = []

        # Include conversation history (limited window)
        if history:
            for turn in history[-5:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                context_parts.append(f"{role.capitalize()}: {content}")

        # Include external context
        if context:
            context_parts.append(context.strip())

        full_context = "\n".join(context_parts).strip()

        return self.template.format(
            instruction=instruction.strip(),
            context=full_context
        ).strip()