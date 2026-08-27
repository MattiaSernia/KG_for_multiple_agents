import re
from CoreferenceResolver import CoreferenceResolver
from utils import sentence_split, ollama_chat

class RequirementsExtractor:
    def __init__(self, model, temperature, coref=None, chunk_dim:int=0):
        self.model=model
        self.temperature=temperature
        self.coref = coref if coref is not None else CoreferenceResolver()
        self._chunk_dim=chunk_dim
        self.context=("""You are an expert in Open Information Extraction (OIE) specialised in public procurement documents (Calls for Tender, CFT).

            Your SOLE task is to extract high-level BUSINESS NEEDS and GOALS from a given sentence.
            You must completely IGNORE constraints, technical specifications, and administrative requirements.

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            WHAT IS A NEED?
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            A NEED expresses WHAT the client wants to achieve and WHY.
            It describes a desired capability, an improvement, or a value to be delivered.

            Ask yourself: "Is this sentence describing a goal or an outcome the client wants?"
            → YES → it is a NEED → extract it.
            → NO  → it is a constraint or administrative requirement → ignore it.

            NEEDS typically contain verbs such as:
            improve, enable, provide, facilitate, assist, support, reduce, simplify,
            deliver, ensure (an outcome), automate, centralise, accelerate, help.

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            WHAT IS NOT A NEED? (IGNORE THESE)
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            Ignore anything that describes HOW or UNDER WHAT CONDITIONS the solution must operate:
            - Compliance requirements   (GDPR, ISO norms, national regulations)
            - Technical bounds          (response time, uptime, storage capacity)
            - Budget or schedule limits (maximum cost, contract duration)
            - Infrastructure rules      (hosting location, cloud certification)
            - Integration mandates      (must connect to system X via API)
            - Security requirements     (encryption, access control)
            - Candidate qualifications  (certifications, turnover thresholds)

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            OUTPUT FORMAT
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            For each need found, output exactly one line in this format:
            [category | priority | subject | predicate | object]

            Field definitions:
            1. category  (OPTIONAL) — High-level functional domain of the need.
                                        Leave blank if not clearly inferable.
            2. priority  (OPTIONAL) — Infer from modal verbs in the source sentence:
                                        must / shall           → MUST
                                        should / is expected   → SHOULD
                                        may / can / could      → MAY
                                        Leave blank if not determinable.
            3. subject   (MANDATORY) — Who achieves or benefits from the goal.
            4. predicate (MANDATORY) — The core action, written as an active infinitive.
            5. object    (MANDATORY) — The target outcome or capability.

            Output ONLY the tuple lines. No numbering, no prose, no explanation.
            If a sentence contains no need at all, output nothing."""
        )

        
        self.examples= """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            EXAMPLES
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            Sentence: The hospital seeks to reduce waiting times for patients in emergency departments.
            Tuples:
            [Healthcare | MUST | hospital | reduce | waiting times for patients in emergency departments]

            Sentence: All suppliers must hold a valid ISO 27001 certificate and have a minimum annual turnover of 5 million euros.
            Tuples:

            Sentence: The goal of the project is to automate invoice processing and free up staff for higher-value tasks.
            Tuples:
            [Finance | | organisation | automate | invoice processing]
            [Workforce | | organisation | free up | staff for higher-value tasks]

            Sentence: The platform should allow field inspectors to submit reports directly from mobile devices.
            Tuples:
            [Field Operations | SHOULD | field inspectors | submit | reports directly from mobile devices]

            Sentence: The system must respond within 3 seconds under peak load and guarantee 99.5% uptime.
            Tuples:

            Sentence: The authority wants to centralise citizen requests across all service channels into a single dashboard, and to provide supervisors with real-time visibility into workload distribution.
            Tuples:
            [Service Management | MUST | authority | centralise | citizen requests across all service channels into a single dashboard]
            [Service Management | MUST | supervisors | obtain | real-time visibility into workload distribution]

            Sentence: Data must be stored on servers located within the national territory and encrypted at rest using AES-256.
            Tuples:

            Sentence: The contract aims to improve the onboarding experience for newly hired employees by providing guided training pathways.
            Tuples:
            [HR | | contract | improve | onboarding experience for newly hired employees]

            Sentence: The solution shall help procurement officers identify the most cost-effective vendors based on historical performance data. The maximum contract value is 800,000 euros over three years.
            Tuples:
            [Procurement | MUST | procurement officers | identify | most cost-effective vendors based on historical performance data]

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Now extract ONLY needs from:
            Sentence: {sentence}
            Tuples:"""

    def parse_tuples(self, text:str)->list:
        pattern=r"\[([^\]|]*)\|([^\]|]*)\|([^\]|]+)\|([^\]|]+)\|([^\]|]+)\]" 
        results = re.findall(pattern, text)
        return [[elem.strip() for elem in match] for match in results]
    
    def answer(self, text:str)->list:
        prompt = self.examples.format(sentence=text)
        response = ollama_chat(
            self.model,
            [{'role': 'system', 'content': self.context}, {'role': 'user', 'content': prompt}],
            options={"temperature": self.temperature},
        )
        return self.parse_tuples(response['message']['content'])

    def pipe(self, text:str) -> list:
        text=self.coref.resolve(text)
        phrases=sentence_split(text, self._chunk_dim)
        chunk_requirements={}
        for phrase in phrases:
            text_requirements=[]
            sentence_tuples=self.answer(phrase)
            for req in sentence_tuples:
                requirement_data = {
                    "category": req[0] if req[0] else None,
                    "priority": req[1] if req[1] else None,
                    "subject": req[2],
                    "predicate": req[3],
                    "object": req[4]
                }
                text_requirements.append(requirement_data)
            chunk_requirements[phrase]=text_requirements
        return chunk_requirements