import re
from CoreferenceResolver import CoreferenceResolver
from utils import sentence_split, ollama_chat

class ConstraintsExtractor:
    def __init__(self, model, temperature, coref=None, chunk_dim:int=0):
        self.model=model
        self.temperature=temperature
        self.coref = coref if coref is not None else CoreferenceResolver()
        self._chunk_dim=chunk_dim
        self.context=("""You are an expert in Open Information Extraction (OIE) specialised in public procurement documents (Calls for Tender, CFT).

            Your SOLE task is to extract CONSTRAINTS from a given sentence.
            You must completely IGNORE business needs, goals, and desired outcomes.

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            WHAT IS A CONSTRAINT?
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            A CONSTRAINT expresses HOW or UNDER WHAT CONDITIONS the solution must operate.
            It is non-negotiable and restricts the space of acceptable solutions.

            Ask yourself: "Is this sentence imposing a limit, a rule, or a mandatory condition?"
            → YES → it is a CONSTRAINT → extract it.
            → NO  → it is a goal or desired outcome → ignore it.

            CONSTRAINTS typically concern:
            - Budget limits          (maximum cost, price ceiling, annual expenditure)
            - Time or duration       (contract length, deadlines, delivery dates)
            - Technical requirements (performance thresholds, uptime, response time, formats)
            - Regulatory compliance  (laws, norms, standards: GDPR, ISO, WCAG, NIS2...)
            - Infrastructure rules   (hosting location, cloud certification, on-premise)
            - Sovereignty / data     (no transfer outside a territory, national hosting)
            - Environmental rules    (emission standards, energy consumption limits)
            - Candidate qualifications (certifications, turnover, insurance)
            - Security requirements  (encryption, access control, audit trails)

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            WHAT IS NOT A CONSTRAINT? (IGNORE THESE)
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            Ignore anything that describes a desired capability, improvement, or outcome:
            - "The system should help users find information faster"
            - "The goal is to reduce processing time"
            - "The solution must provide an intelligent assistant to staff"
            - "The authority wants to improve service quality"

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            OUTPUT FORMAT
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            For each constraint found, output exactly one line in this format:
            [constraintType | subject | predicate | object]

            Field definitions:
            1. constraintType (OPTIONAL) — You MUST use ONLY one of these exact values if applicable:
                                            "Budgetary"    — cost, price, financial limits
                                            "Temporal"     — duration, deadlines, schedule
                                            "Technical"    — performance, formats, uptime, standards
                                            "Regulatory"   — legal norms, compliance obligations
                                            "Environmental"— energy, emissions, carbon footprint
                                            "Sovereignty"  — data location, territorial restrictions
                                            Leave blank if none apply, but KEEP the pipe.
            2. subject   (MANDATORY) — The entity being constrained.
            3. predicate (MANDATORY) — The limiting verb or state (e.g. "must be", "is limited to",
                                        "must comply with", "shall not exceed", "is required to have").
            4. object    (MANDATORY) — The exact limit, value, standard, or condition imposed.

            Output ONLY the tuple lines. No numbering, no prose, no explanation.
            If a sentence contains no constraint at all, output nothing.""")
        
        self.examples= """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            EXAMPLES
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            Sentence: The total contract value shall not exceed 2.5 million dollars over five years.
            Tuples:
            [Budgetary | total contract value | shall not exceed | 2.5 million dollars over five years]

            Sentence: The agency wants to modernise its document management system and make information easier to find for staff.
            Tuples:

            Sentence: All servers hosting citizen data must be located within the national territory and must not be operated by non-national entities.
            Tuples:
            [Sovereignty | servers hosting citizen data | must be located | within the national territory]
            [Sovereignty | servers hosting citizen data | must not be operated by | non-national entities]

            Sentence: The framework agreement will run for an initial period of two years, with the possibility of two one-year extensions.
            Tuples:
            [Temporal | framework agreement | will run for | initial period of two years with two possible one-year extensions]

            Sentence: The platform must achieve a minimum uptime of 99.5% and return search results in under 4 seconds.
            Tuples:
            [Technical | platform | must achieve | minimum uptime of 99.5%]
            [Technical | platform | must return search results in | under 4 seconds]

            Sentence: The solution should allow inspectors to file reports remotely and reduce duplicated data entry.
            Tuples:

            Sentence: Tenderers must demonstrate compliance with ISO 27001 and hold valid professional liability insurance of at least 1 million euros.
            Tuples:
            [Regulatory | tenderers | must demonstrate compliance with | ISO 27001]
            [Budgetary | professional liability insurance | must be of at least | 1 million euros]

            Sentence: The fleet of vehicles deployed under this contract must meet Euro 6 emission standards.
            Tuples:
            [Environmental | fleet of vehicles | must meet | Euro 6 emission standards]

            Sentence: The system shall integrate with the existing HR software via a REST API. The goal of the project is to reduce administrative workload for HR staff.
            Tuples:
            [Technical | system | shall integrate with | existing HR software via a REST API]

            Sentence: Personal data processed under this contract must be encrypted at rest and in transit using AES-256, and must never be transferred outside the European Economic Area.
            Tuples:
            [Technical | personal data | must be encrypted at rest and in transit using | AES-256]
            [Sovereignty | personal data | must never be transferred outside | the European Economic Area]

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Now extract ONLY constraints from:
            Sentence: {sentence}
            Tuples:"""

    def parse_tuples(self, text:str)->list:
        pattern=r"\[([^\]|]*)\|([^\]|]+)\|([^\]|]+)\|([^\]|]+)\]" 
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

    def pipe(self, text:str) -> dict:
        text=self.coref.resolve(text)
        phrases=sentence_split(text, self._chunk_dim)
        chunk_constraints={}
        for phrase in phrases:
            text_constraints=[]
            sentence_tuples=self.answer(phrase)
            for const in sentence_tuples:
                constraint_data = {
                    "constraintType": const[0] if const[0] else None,
                    "subject": const[1],
                    "predicate": const[2],
                    "object": const[3]
                }
                text_constraints.append(constraint_data)
            chunk_constraints[phrase]=text_constraints
        return chunk_constraints