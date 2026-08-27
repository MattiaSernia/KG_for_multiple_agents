import re
from CoreferenceResolver import CoreferenceResolver
from utils import sentence_split, ollama_chat

class ProposalsExtractor:
    def __init__(self, model, temperature, coref=None, chunk_dim:int=0):
        self.model=model
        self.temperature=temperature
        self.coref = coref if coref is not None else CoreferenceResolver()
        self._chunk_dim=chunk_dim
        self.context=("""You are an expert in Open Information Extraction (OIE) specialised in tender PROPOSALS
            (the documents written BY a bidding consortium IN RESPONSE to a public Call for Tenders).

            Your SOLE task is to extract PROPOSALS (commitments made by the bidder) from a given sentence.
            You must completely IGNORE rephrasings of the client's needs, background context, and
            generic marketing claims that do not commit the bidder to anything concrete.

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            WHAT IS A PROPOSAL?
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            A PROPOSAL expresses something the consortium (or the solution it proposes)
            WILL DO, WILL DELIVER, GUARANTEES, COMPLIES WITH, OFFERS or INCLUDES.
            It is a concrete commitment of the bidder towards the contracting authority.

            Ask yourself: "Is the bidder taking responsibility for something concrete here?"
            → YES → it is a PROPOSAL → extract it.
            → NO  → it is context, a restatement of the client's need, or filler → ignore it.

            PROPOSALS typically contain verbs such as:
            propose, commit to, will deliver, will provide, provide, deliver, include,
            guarantee, comply with, host, integrate, support, ensure, certify,
            cover, address, implement, offer, use, deploy, build.

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            WHAT IS NOT A PROPOSAL? (IGNORE THESE)
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            Ignore anything that does not commit the bidder to something concrete:
            - Rephrasings of the client's need
              ("The City wants to improve administrative services")
            - Generic marketing claims with no deliverable
              ("We are leaders in AI for the public sector")
            - Background descriptions of the consortium
              ("Our company was founded in 2005 and has 120 employees")
            - Pure transitions / section introductions
              ("In this section we describe our technical approach")
            - Tender requirements quoted verbatim without a corresponding commitment
              ("The tender requires 99.9% availability")

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            OUTPUT FORMAT
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            For each proposal found, output exactly one line in this format:
            [subject | predicate | object]

            Field definitions:
            1. subject   (MANDATORY) — Who commits. Typically "the consortium",
                                        "the proposed solution", or one of its components
                                        (e.g. "our hosting infrastructure", "the chatbot module").
                                        MUST NOT be the contracting authority.
            2. predicate (MANDATORY) — The commitment verb, written as an active form
                                        (e.g. "commits to host", "will deliver", "provides",
                                        "guarantees", "complies with", "integrates with").
            3. object    (MANDATORY) — The concrete deliverable, figure, certification,
                                        capability or scope being committed.

            Output ONLY the tuple lines. No numbering, no prose, no explanation.
            If a sentence contains no proposal at all, output nothing.""")

        self.examples= """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            EXAMPLES
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            Sentence: Our platform will provide real-time translation in 24 official EU languages for all citizen-facing interactions.
            Tuples:
            [our platform | will provide | real-time translation in 24 official EU languages for all citizen-facing interactions]

            Sentence: The hospital authority seeks to modernise its patient intake process and reduce waiting times.
            Tuples:

            Sentence: The proposed system guarantees 99.95% uptime and includes automatic failover to a secondary data centre located in Frankfurt.
            Tuples:
            [the proposed system | guarantees | 99.95% uptime]
            [the proposed system | includes | automatic failover to a secondary data centre located in Frankfurt]

            Sentence: With over 15 years of experience in the public sector, our team has delivered more than 40 digital transformation projects.
            Tuples:

            Sentence: We will deploy a containerised microservices architecture on ISO 27001-certified cloud infrastructure, ensuring full data residency within national borders.
            Tuples:
            [the consortium | will deploy | a containerised microservices architecture on ISO 27001-certified cloud infrastructure]
            [the consortium | ensures | full data residency within national borders]

            Sentence: The contracting authority requires all data to be encrypted at rest using AES-256.
            Tuples:

            Sentence: Our solution integrates with the existing SAP ERP system via a bidirectional REST API and provides a single sign-on module compatible with Active Directory.
            Tuples:
            [our solution | integrates with | the existing SAP ERP system via a bidirectional REST API]
            [our solution | provides | a single sign-on module compatible with Active Directory]

            Sentence: The consortium commits to delivering a fully operational pilot within 4 months of contract signature, covering three regional offices.
            Tuples:
            [the consortium | commits to deliver | a fully operational pilot within 4 months of contract signature covering three regional offices]

            Sentence: In the following section we present the staffing plan for the project.
            Tuples:

            Sentence: All training data will be anonymised using differential privacy techniques before any model training takes place, and no personal data will leave the client's infrastructure.
            Tuples:
            [the consortium | will anonymise | all training data using differential privacy techniques before model training]
            [the proposed solution | guarantees | no personal data will leave the client's infrastructure]

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Now extract ONLY proposals from:
            Sentence: {sentence}
            Tuples:"""

    def parse_tuples(self, text:str)->list:
        pattern=r"\[([^\]|]+)\|([^\]|]+)\|([^\]|]+)\]"
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
        chunk_proposals={}
        for phrase in phrases:
            text_proposals=[]
            sentence_tuples=self.answer(phrase)
            for prop in sentence_tuples:
                proposal_data = {
                    "subject": prop[0],
                    "predicate": prop[1],
                    "object": prop[2]
                }
                text_proposals.append(proposal_data)
            chunk_proposals[phrase]=text_proposals
        return chunk_proposals