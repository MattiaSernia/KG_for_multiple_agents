import json
import re
from CoreferenceResolver import CoreferenceResolver
from utils import ollama_chat, sentence_split


class Phi4ConstraintsExtractor:

    _SYSTEM = (
        "You are a helpful AI assistant specializing in Information Extraction tasks "
        "such as Named Entity Recognition and Relation Extraction. "
        "Follow the instructions given by the user."
    )

    _SCHEMA = json.dumps({
        "entities": [
            "System", "Data", "Provider", "Infrastructure", "Contract",
            "Candidate", "Solution", "Platform", "Supplier"
        ],
        "relations": [
            "must comply with", "shall not exceed", "must be located in",
            "must achieve", "is limited to", "must not transfer",
            "must be encrypted", "must integrate with", "must guarantee",
            "shall not exceed", "must hold", "must demonstrate"
        ],
        "constraintType_values": [
            "Budgetary", "Temporal", "Technical", "Regulatory",
            "Environmental", "Sovereignty"
        ]
    })

    _EXAMPLE = (
        'Text: "The total contract value shall not exceed 2.5 million euros over three years. '
        'All data must be processed exclusively on infrastructure located within the national territory. '
        'The platform must achieve a minimum uptime of 99.5%. '
        'The framework agreement will run for an initial period of two years, with the possibility of two one-year extensions. '
        'All subcontractors must hold a valid operating licence issued by the competent national authority and comply with applicable labour law. '
        'The equipment deployed under this contract must carry an EU Energy Label rating of A or above and must not exceed a power consumption of 500 watts per unit."\n'
        'Output: ['
        '{"subject": "total contract value", "predicate": "shall not exceed", "object": "2.5 million euros over three years", "constraintType": "Budgetary"}, '
        '{"subject": "data", "predicate": "must be processed on", "object": "infrastructure located within the national territory", "constraintType": "Sovereignty"}, '
        '{"subject": "platform", "predicate": "must achieve", "object": "minimum uptime of 99.5%", "constraintType": "Technical"}, '
        '{"subject": "framework agreement", "predicate": "will run for", "object": "initial period of two years with two possible one-year extensions", "constraintType": "Temporal"}, '
        '{"subject": "subcontractors", "predicate": "must hold", "object": "valid operating licence issued by the competent national authority", "constraintType": "Regulatory"}, '
        '{"subject": "subcontractors", "predicate": "must comply with", "object": "applicable labour law", "constraintType": "Regulatory"}, '
        '{"subject": "equipment", "predicate": "must carry", "object": "EU Energy Label rating of A or above", "constraintType": "Environmental"}, '
        '{"subject": "equipment", "predicate": "must not exceed", "object": "power consumption of 500 watts per unit", "constraintType": "Environmental"}'
        ']'
    )

    _OUTPUT_FORMAT = '[{"subject": "...", "predicate": "...", "object": "...", "constraintType": "Budgetary|Temporal|Technical|Regulatory|Environmental|Sovereignty or null"}]'

    def __init__(self, model: str, coref=None, chunk_dim: int = 0, no_schema: bool = False):
        self.model = model
        self.coref = coref if coref is not None else CoreferenceResolver()
        self._chunk_dim = chunk_dim
        self._no_schema = no_schema

    def _build_prompt(self, text: str) -> str:
        return (
            "Information Extraction is the process of automatically identifying and "
            "extracting structured information from unstructured text data.\n"
            "Always extract numbers, dates, and currency values regardless of the specific task.\n\n"
            "The task at hand is Constraints Extraction: extract CONSTRAINTS from the text. "
            "A constraint expresses HOW or UNDER WHAT CONDITIONS the solution must operate. "
            "It is non-negotiable and restricts the space of acceptable solutions. "
            "Ignore business goals and desired outcomes.\n\n"
            "Here is an example of task execution:\n"
            f"{self._EXAMPLE}\n\n"
            "Analyze the text carefully. Extract only conditions, limits, rules, and mandatory requirements.\n"
            f"Extract the information in the following format: `{self._OUTPUT_FORMAT}`.\n"
            "If no constraints are found, return an empty list: [].\n"
            "Please provide only the extracted information without any explanations.\n\n"
            f"Schema: {self._SCHEMA}\n"
            f"Text: {text}"
        )

    def _parse(self, raw: str) -> list:
        raw = raw.strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [
                        {
                            "subject":        str(item.get("subject", "")).strip(),
                            "predicate":      str(item.get("predicate", "")).strip(),
                            "object":         str(item.get("object", "")).strip(),
                            "constraintType": item.get("constraintType") or None,
                        }
                        for item in data
                        if isinstance(item, dict)
                        and item.get("subject") and item.get("predicate") and item.get("object")
                    ]
            except json.JSONDecodeError:
                pass
        return []

    def answer(self, text: str) -> list:
        response = ollama_chat(
            self.model,
            [
                {"role": "system", "content": self._SYSTEM},
                {"role": "user",   "content": self._build_prompt(text)},
            ],
        )
        return self._parse(response["message"]["content"])

    def pipe(self, text: str) -> dict:
        text = self.coref.resolve(text)
        phrases = sentence_split(text, self._chunk_dim)
        chunk_constraints = {}
        for phrase in phrases:
            chunk_constraints[phrase] = self.answer(phrase)
        return chunk_constraints
