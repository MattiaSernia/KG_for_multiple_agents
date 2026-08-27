import re
from CoreferenceResolver import CoreferenceResolver
from utils import ollama_chat, sentence_split
class TripletExtractor:
    def __init__(self, model, temperature, coref=None, chunk_dim:int=0):
        self.model=model
        self.temperature=temperature
        self.coref = coref if coref is not None else CoreferenceResolver()
        self._chunk_dim=chunk_dim
        self.context=("""You are an expert in Open Information Extraction (OIE) for business intelligence.
You extract factual subject-relation-object triplets from messages produced by AI agents 
operating in the context of company management and public tender evaluation.
 
The domain includes: financial data, HR resources, project portfolios, 
compliance certifications, contracts, and tender bids.
 
RULES:
- Extract only concrete, factual triplets (numbers, statuses, names, dates, amounts)
- Use the company or agent name as subject when it is the source of the fact
- Keep relations short and descriptive (e.g. "has_cash_position", "has_risk_level", "available_FTE")
- Keep objects specific and literal (e.g. "€520,000", "low", "38%", "2", "up-to-date")
- DO NOT extract generic or encyclopedic facts (e.g. "Italy has capital Rome")
- DO NOT extract meta-facts about the conversation itself
- If no concrete factual triplets can be extracted, output nothing
- Output ONLY the triplets, one per line, using this exact format: [subject | relation | object]
- No numbering, no explanation, no other text""")
        self.examples= """EXAMPLES:
 
Sentence: Your current cash position is €520,000, while the available credit line is €300,000, for a combined liquidity of €820,000.
Triplets:
[Nexus Engineering | has_cash_position | €520,000]
[Nexus Engineering | has_credit_line | €300,000]
[Nexus Engineering | has_liquidity | €820,000]
 
Sentence: There are a total of two FTEs available for new projects in Q2 2025.
Triplets:
[Nexus Engineering | available_FTE_Q2_2025 | 2]
 
Sentence: Two out of three projects are assessed as having low risk (CTR001 and CTR002), while one project has been evaluated as medium-risk (CTR003).
Triplets:
[CTR001 | has_risk_level | low]
[CTR002 | has_risk_level | low]
[CTR003 | has_risk_level | medium]
 
Sentence: The total completion percentage of ongoing projects is approximately 38%.
Triplets:
[Nexus Engineering portfolio | completion_percentage | 38%]
 
Sentence: Yes, all mandatory certifications are up-to-date as of April 1st 2025.
Triplets:
[Nexus Engineering | certifications_status | up-to-date]
[Nexus Engineering | certifications_valid_as_of | 2025-04-01]
 
Sentence: Are we ready to bid for the Regione Lombardia €2M infrastructure tender, submission deadline May 10th?
Triplets:
[Regione Lombardia tender | value | €2,000,000]
[Regione Lombardia tender | submission_deadline | 2025-05-10]
[Regione Lombardia tender | type | infrastructure]
 
Now extract triplets from:
Sentence: {sentence}
Triplets:"""

    def parse_triplets(self, text:str)->list:
        pattern=r"\[([^\]|]+)\|([^\]|]+)\|([^\]|]+)\]" 
        #pattern = r"\[([^\]|]+)\|([^\]|]+)\|([^\]|]+)\|([^\]|]+)\]"
        results = re.findall(pattern, text)
        return [[elem.strip() for elem in match] for match in results]
    
    def answer(self, text:str)->list:
        prompt = self.examples.format(sentence=text)
        response = ollama_chat(
            self.model,
            [{'role': 'system', 'content': self.context}, {'role': 'user', 'content': prompt}],
            options={"temperature": self.temperature},
        )
        return self.parse_triplets(response['message']['content'])
    
    def pipe(self, text:str) -> dict:
        text=self.coref.resolve(text)
        phrases=sentence_split(text, self._chunk_dim)
        chunk_triplets={}
        for phrase in phrases:
            text_triplets=[]
            sentence_tuples=self.answer(phrase)
            for prop in sentence_tuples:
                triplets_data = {
                    "subject": prop[0],
                    "predicate": prop[1],
                    "object": prop[2]
                }
                text_triplets.append(triplets_data)
            chunk_triplets[phrase]=text_triplets
        return chunk_triplets
