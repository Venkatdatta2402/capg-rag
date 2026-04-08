"""Tool definition for structured MCQ presentation.

When the RAG agent calls this tool, the LLM outputs quiz questions as a
structured JSON tool call instead of inline text. The client receives a
typed QuizForm object it can render as interactive clickable options
rather than text the student has to read and retype.
"""

# OpenAI / Gemini tool definition (JSON schema format)
PRESENT_MCQ_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "present_mcq",
        "description": (
            "Present adversarial MCQ comprehension questions to the student. "
            "Call this after completing your explanation to embed an interactive quiz."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {
                                "type": "string",
                                "description": "Identifier: q1, q2, or q3",
                            },
                            "question": {
                                "type": "string",
                                "description": "The question text",
                            },
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": (
                                    "Four options as flat strings: "
                                    "[\"A) 100 cm\", \"B) 10 cm\", \"C) 1000 cm\", \"D) 1 cm\"]"
                                ),
                            },
                            "correct_answer": {
                                "type": "string",
                                "enum": ["A", "B", "C", "D"],
                                "description": "Letter of the correct option",
                            },
                        },
                        "required": ["question_id", "question", "options", "correct_answer"],
                    },
                }
            },
            "required": ["questions"],
        },
    },
}
