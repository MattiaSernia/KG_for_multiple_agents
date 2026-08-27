import json
import re
from CoreferenceResolver import CoreferenceResolver
from utils import ollama_chat, sentence_split


class Phi4RequirementsExtractor:

    _SYSTEM = (
        "You are a helpful AI assistant specializing in Information Extraction tasks "
        "such as Named Entity Recognition and Relation Extraction. "
        "Follow the instructions given by the user."
    )

    _SCHEMA = json.dumps({
        "entities": [
            "Organization", "User", "Agent", "Service", "System",
            "Platform", "Document", "Capability"
        ],
        "relations": [
            "must improve", "must enable", "must provide", "must facilitate",
            "must assist", "must support", "must reduce", "must automate",
            "must centralise", "must accelerate", "must help", "must deliver",
            "must simplify", "must ensure"
        ],
        "priority_values": ["MUST", "SHOULD", "MAY"],
        "category_values": [
            "Staff Assistance", "Information Access", "Drafting", "File Handling",
            "Collaboration", "Integration", "Explainability", "Service Quality",
            "Efficiency", "Usability", "Deployment", "Change Management"
        ]
    })

    _EXAMPLE = (
        'Text: "The organisation aims to reduce document processing times and provide staff with quick access to regulatory information from their workstation. '
        'The solution should allow teams to share file context seamlessly when handing over work between colleagues."\n'
        'Output: ['
        '{"subject": "organisation", "predicate": "reduce", "object": "document processing times", "priority": "MUST", "category": "Efficiency"}, '
        '{"subject": "staff", "predicate": "access", "object": "regulatory information from their workstation", "priority": "MUST", "category": "Information Access"}, '
        '{"subject": "teams", "predicate": "share", "object": "file context when handing over work between colleagues", "priority": "SHOULD", "category": "Collaboration"}'
        ']'
    )

    _OUTPUT_FORMAT = '[{"subject": "...", "predicate": "...", "object": "...", "priority": "MUST|SHOULD|MAY or null", "category": "... or null"}]'

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
            "The task at hand is Requirements Extraction: extract high-level BUSINESS NEEDS and GOALS "
            "from the text. Ignore constraints, technical specifications, budget limits, and administrative requirements. "
            "A requirement expresses WHAT the client wants to achieve and WHY.\n\n"
            "Here is an example of task execution:\n"
            f"{self._EXAMPLE}\n\n"
            "Analyze the text carefully. Extract only business goals and desired outcomes.\n"
            f"Extract the information in the following format: `{self._OUTPUT_FORMAT}`.\n"
            "If no requirements are found, return an empty list: [].\n"
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
                            "subject":   str(item.get("subject", "")).strip(),
                            "predicate": str(item.get("predicate", "")).strip(),
                            "object":    str(item.get("object", "")).strip(),
                            "priority":  item.get("priority") or None,
                            "category":  item.get("category") or None,
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
        chunk_requirements = {}
        for phrase in phrases:
            chunk_requirements[phrase] = self.answer(phrase)
        return chunk_requirements
