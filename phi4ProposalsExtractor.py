import json
import re
from CoreferenceResolver import CoreferenceResolver
from utils import ollama_chat, sentence_split


class Phi4ProposalsExtractor:

    _SYSTEM = (
        "You are a helpful AI assistant specializing in Information Extraction tasks "
        "such as Named Entity Recognition and Relation Extraction. "
        "Follow the instructions given by the user."
    )

    _SCHEMA = json.dumps({
        "entities": [
            "Consortium", "Solution", "Platform", "Infrastructure",
            "Team", "Module", "System", "Service"
        ],
        "relations": [
            "will deliver", "provides", "guarantees", "integrates with",
            "complies with", "will deploy", "hosts", "supports",
            "commits to deliver", "will provide", "ensures", "certifies",
            "will implement", "offers", "covers"
        ]
    })

    _EXAMPLE = (
        'Text: "Our platform guarantees 99.95% uptime and includes automatic failover. '
        'We will deploy a containerised architecture on ISO 27001-certified cloud infrastructure. '
        'The consortium commits to completing all on-site maintenance interventions within 48 hours of notification."\n'
        'Output: ['
        '{"subject": "our platform", "predicate": "guarantees", "object": "99.95% uptime"}, '
        '{"subject": "our platform", "predicate": "includes", "object": "automatic failover"}, '
        '{"subject": "the consortium", "predicate": "will deploy", "object": "a containerised architecture on ISO 27001-certified cloud infrastructure"}, '
        '{"subject": "the consortium", "predicate": "commits to complete", "object": "all on-site maintenance interventions within 48 hours of notification"}'
        ']'
    )

    _OUTPUT_FORMAT = '[{"subject": "...", "predicate": "...", "object": "..."}]'

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
            "The task at hand is Proposals Extraction: extract concrete COMMITMENTS made by a bidding "
            "consortium in response to a call for tenders. A proposal expresses something the consortium "
            "WILL DO, WILL DELIVER, GUARANTEES, or OFFERS. "
            "Ignore rephrasings of the client's needs and generic marketing claims.\n\n"
            "Here is an example of task execution:\n"
            f"{self._EXAMPLE}\n\n"
            "Analyze the text carefully. Extract only concrete bidder commitments and deliverables.\n"
            f"Extract the information in the following format: `{self._OUTPUT_FORMAT}`.\n"
            "If no proposals are found, return an empty list: [].\n"
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
        chunk_proposals = {}
        for phrase in phrases:
            chunk_proposals[phrase] = self.answer(phrase)
        return chunk_proposals
