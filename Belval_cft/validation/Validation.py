import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import ollama_chat
class Validation:
    def __init__(self, model, temperature):
        self.model=model
        self.temperature=temperature
        _here = os.path.dirname(os.path.abspath(__file__))
        self._requirements=self._load(os.path.join(_here, "requirements.json"), "requirements")
        self._constraints=self._load(os.path.join(_here, "constraints.json"), "constraints")
        
    def validate_requirements(self, proposal: str) -> list[str]:
        prompt = (
            "You are an expert requirements analyst.\n\n"
            "Below is a list of requirements in JSON format:\n"
            f"{self._requirements}\n\n"
            "Below is a proposal:\n"
            f"{proposal}\n\n"
            "Identify which requirements are satisfied by this proposal.\n"
            'Return a JSON object with a single key "satisfied" whose value is an array of the matching requirement IDs.\n'
            'Example: {"satisfied": ["REQ-01", "REQ-03"]}\n'
            "If none are satisfied return: {\"satisfied\": []}"
        )

        response = ollama_chat(
            self.model,
            [{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
            format="json",
        )
        data = json.loads(response["message"]["content"])
        return data.get("satisfied", [])

    def validate_constraints(self, proposal: str) -> list[str]:
        prompt = (
            "You are an expert compliance analyst.\n\n"
            "Below is a list of constraints in JSON format:\n"
            f"{self._constraints}\n\n"
            "Below is a proposal:\n"
            f"{proposal}\n\n"
            "Identify which constraints are satisfied (respected) by this proposal.\n"
            'Return a JSON object with a single key "satisfied" whose value is an array of the matching constraint IDs.\n'
            'Example: {"satisfied": ["CON-01", "CON-03"]}\n'
            "If none are satisfied return: {\"satisfied\": []}"
        )

        response = ollama_chat(
            self.model,
            [{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
            format="json",
        )
        data = json.loads(response["message"]["content"])
        return data.get("satisfied", [])

    def _validate_requirement(self, proposal: str, requirement: dict, key: str) -> tuple:
        text = f"{key}: {requirement.get('subject', '')} {requirement.get('predicate', '')} {requirement.get('object', '')}"
        if requirement.get("priority"):
            text += f" (priority: {requirement['priority']})"
        if requirement.get("category"):
            text += f" (category: {requirement['category']})"

        prompt = (
            "You are an expert requirements analyst.\n\n"
            "Below is a single requirement:\n"
            f"{text}\n\n"
            "Below is a proposal:\n"
            f"{proposal}\n\n"
            "Is this requirement satisfied by the proposal?\n"
            'Return a JSON object with a single key "satisfied" whose value is true or false.\n'
            'Example: {"satisfied": true}'
        )

        response = ollama_chat(
            self.model,
            [{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
            format="json",
        )
        data = json.loads(response["message"]["content"])
        satisfied = bool(data.get("satisfied", False))
        return (True, text) if satisfied else (False, None)

    def _validate_constraint(self, proposal: str, constraint: dict, key:str) -> tuple:
        text = f"{key}: {constraint.get('subject', '')} {constraint.get('predicate', '')} {constraint.get('object', '')}"
        if constraint.get("constraintType"):
            text += f" ({constraint['constraintType']})"

        prompt = (
            "You are an expert compliance analyst.\n\n"
            "Below is a single constraint:\n"
            f"{text}\n\n"
            "Below is a proposal:\n"
            f"{proposal}\n\n"
            "Is this constraint satisfied (respected) by the proposal?\n"
            'Return a JSON object with a single key "satisfied" whose value is true or false.\n'
            'Example: {"satisfied": true}'
        )

        response = ollama_chat(
            self.model,
            [{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
            format="json",
        )
        data = json.loads(response["message"]["content"])
        satisfied = bool(data.get("satisfied", False))
        return (True, text) if satisfied else (False, None)

    def validate(self, proposal: str, title:str):
        with open(title, 'w') as f:
            f.write("\nSATISFIED REQUIREMENTS\n")
            requirements = json.loads(self._requirements)
            for key,element in requirements.items():
                answer=self._validate_requirement(proposal, element, key)
                if answer[0]:
                    f.write(f"{answer[1]}\n")
            f.write("\nSATISFIED CONSTRAINTS\n")
            constraints = json.loads(self._constraints)
            for key,element in constraints.items():
                answer=self._validate_constraint(proposal, element, key)
                if answer[0]:
                    f.write(f"{answer[1]}\n")
                



    def _load(self, file, doc):
        with open(file, "r", encoding="utf-8") as f:
            file=json.load(f)
        struct={}
        for f in file:
            key = f["id"]
            entry = {"subject":f["subject"], "predicate": f["predicate"], "object": f["object"]}
            if doc=="requirements":
                if f["priority"]:
                    entry["priority"] = f["priority"]
                if f["category"]:
                    entry["category"] = f["category"]
            else:
                if f["constraintType"]:
                    entry["constraintType"] = f["constraintType"]
            struct[key]=entry
        return json.dumps(struct, indent=2, ensure_ascii=False)
