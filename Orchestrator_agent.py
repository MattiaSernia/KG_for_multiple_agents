import logging
import json
import yaml
from agent import Agent
from custom_graph import Custom_Graph
from rdflib import Namespace
from rdflib.namespace import RDF, PROV
from utils import uri_to_label, ollama_chat
from mxg import Message


class Orchestrator_Agent:
    def __init__(self, agents:list[Agent], model:str, graph_name:str, chunk_dimension:int, extractor_type:str="llama", kg_format:str="turtle-light", no_schema:bool=False):
        self.agents=agents
        self.model=model
        self.agent_answer=[]
        self._graph_name=graph_name
        self._kg_format=kg_format
        self._cgraph=Custom_Graph(agents, graph_name, model, chunk_dimension, extractor_type, no_schema)

    def _agent_registry(self) -> str:
        lines = []
        for agent in self.agents:
            lines.append(f'- "{agent.name}": {agent.description}')
        return "\n".join(lines)

    def _get_requirements_text(self) -> str:
        EX   = Namespace("http://example.org/ontologia#")
        EDGE = "http://example.org/edge/"
        ds   = self._cgraph._ds
        struct: dict = {}
        for subj in ds.subjects(RDF.type, EX.Requirement):
            s_uri = next(ds.objects(subj, RDF.subject), None)
            s = uri_to_label(s_uri) if s_uri else "?"
            pred, obj = "", ""
            for p, o in ds.predicate_objects(subj):
                if str(p).startswith(EDGE):
                    pred = uri_to_label(p)
                    obj  = uri_to_label(o)
                    break
            pri_uri = next(ds.objects(subj, EX.priority), None)
            cat_uri = next(ds.objects(subj, EX.category), None)
            if s not in struct:
                struct[s] = []
            entry = {"predicate": pred, "object": obj}
            if pri_uri:
                entry["priority"] = uri_to_label(pri_uri)
            if cat_uri:
                entry["category"] = uri_to_label(cat_uri)
            struct[s].append(entry)
        return json.dumps(struct, indent=2, ensure_ascii=False)

    def _get_constraints_text(self) -> str:
        EX   = Namespace("http://example.org/ontologia#")
        EDGE = "http://example.org/edge/"
        ds   = self._cgraph._ds
        struct: dict = {}
        for subj in ds.subjects(RDF.type, EX.Constraint):
            s_uri = next(ds.objects(subj, RDF.subject), None)
            s = uri_to_label(s_uri) if s_uri else "?"
            pred, obj = "", ""
            for p, o in ds.predicate_objects(subj):
                if str(p).startswith(EDGE):
                    pred = uri_to_label(p)
                    obj  = uri_to_label(o)
                    break
            ct_uri = next(ds.objects(subj, EX.constraintType), None)
            if s not in struct:
                struct[s] = []
            entry = {"predicate": pred, "object": obj}
            if ct_uri:
                entry["constraintType"] = uri_to_label(ct_uri)
            struct[s].append(entry)
        return json.dumps(struct, indent=2, ensure_ascii=False)
    
    def _get_triplets_text(self) -> str:
        EX   = Namespace("http://example.org/ontologia#")
        EDGE = "http://example.org/edge/"
        ds   = self._cgraph._ds

        def local(uri):
            s = str(uri)
            return s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]

        lines = []
        for subj in ds.subjects(RDF.type, EX.Triplet):
            parts = ["a ex:Triplet"]
            s_uri = next(ds.objects(subj, RDF.subject), None)
            if s_uri:
                parts.append(f"rdf:subject node:{local(s_uri)}")
            for p, o in ds.predicate_objects(subj):
                if str(p).startswith(EDGE):
                    parts.append(f"edge:{local(p)} node:{local(o)}")
            for _, _, _, ctx in ds.quads((subj, RDF.type, EX.Triplet, None)):
                chunk_uri = next(ds.objects(ctx.identifier, PROV.wasDerivedFrom), None)
                if chunk_uri:
                    msg_uri = next(ds.objects(chunk_uri, PROV.wasDerivedFrom), None)
                    if msg_uri:
                        agent_uri = next(ds.objects(msg_uri, PROV.wasAttributedTo), None)
                        if agent_uri:
                            parts.append(f"ex:extractedBy node:{local(agent_uri)}")
                break
            lines.append(f"tri:{local(subj)} " + " ; ".join(parts) + " .")
        return "\n".join(lines)

    def _get_triplets_json(self) -> str:
        EX   = Namespace("http://example.org/ontologia#")
        EDGE = "http://example.org/edge/"
        ds   = self._cgraph._ds
        struct: dict = {}
        for subj in ds.subjects(RDF.type, EX.Triplet):
            s_uri = next(ds.objects(subj, RDF.subject), None)
            s = uri_to_label(s_uri) if s_uri else "?"
            pred, obj = "", ""
            for p, o in ds.predicate_objects(subj):
                if str(p).startswith(EDGE):
                    pred = uri_to_label(p)
                    obj  = uri_to_label(o)
                    break
            agent_name = "unknown"
            for _, _, _, ctx in ds.quads((subj, RDF.type, EX.Triplet, None)):
                chunk_uri = next(ds.objects(ctx.identifier, PROV.wasDerivedFrom), None)
                if chunk_uri:
                    msg_uri = next(ds.objects(chunk_uri, PROV.wasDerivedFrom), None)
                    if msg_uri:
                        agent_uri = next(ds.objects(msg_uri, PROV.wasAttributedTo), None)
                        if agent_uri:
                            agent_name = uri_to_label(agent_uri).replace("_", " ").title()
                break
            if agent_name not in struct:
                struct[agent_name] = []
            struct[agent_name].append({"subject": s, "predicate": pred, "object": obj})
        return json.dumps(struct, indent=2, ensure_ascii=False)

    def _get_kg_turtle_light(self) -> str:
        EX   = Namespace("http://example.org/ontologia#")
        EDGE = "http://example.org/edge/"
        ds   = self._cgraph._ds

        def local(uri):
            s = str(uri)
            return s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]

        lines = []
        for subj in ds.subjects(RDF.type, EX.Requirement):
            parts = ["a ex:Requirement"]
            s_uri = next(ds.objects(subj, RDF.subject), None)
            if s_uri:
                parts.append(f"rdf:subject node:{local(s_uri)}")
            for p, o in ds.predicate_objects(subj):
                if str(p).startswith(EDGE):
                    parts.append(f"edge:{local(p)} node:{local(o)}")
            pri_uri = next(ds.objects(subj, EX.priority), None)
            if pri_uri:
                parts.append(f"ex:priority ex:{local(pri_uri)}")
            cat_uri = next(ds.objects(subj, EX.category), None)
            if cat_uri:
                parts.append(f"ex:category ex:{local(cat_uri)}")
            lines.append(f"req:{local(subj)} " + " ; ".join(parts) + " .")

        for subj in ds.subjects(RDF.type, EX.Constraint):
            parts = ["a ex:Constraint"]
            s_uri = next(ds.objects(subj, RDF.subject), None)
            if s_uri:
                parts.append(f"rdf:subject node:{local(s_uri)}")
            for p, o in ds.predicate_objects(subj):
                if str(p).startswith(EDGE):
                    parts.append(f"edge:{local(p)} node:{local(o)}")
            ct_uri = next(ds.objects(subj, EX.constraintType), None)
            if ct_uri:
                parts.append(f"ex:constraintType ex:{local(ct_uri)}")
            lines.append(f"con:{local(subj)} " + " ; ".join(parts) + " .")

        return "\n".join(lines)

    def _get_kg_yaml_ld(self) -> str:
        EX   = Namespace("http://example.org/ontologia#")
        EDGE = "http://example.org/edge/"
        ds   = self._cgraph._ds

        def local(uri):
            s = str(uri)
            return s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]

        context = {
            "ex":   "http://example.org/ontologia#",
            "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "node": "http://example.org/node/",
            "edge": "http://example.org/edge/",
            "req":  "http://example.org/requirement/",
            "con":  "http://example.org/constraint/",
        }
        graph = []

        for subj in ds.subjects(RDF.type, EX.Requirement):
            node = {"@id": f"req:{local(subj)}", "@type": "ex:Requirement"}
            s_uri = next(ds.objects(subj, RDF.subject), None)
            if s_uri:
                node["rdf:subject"] = {"@id": f"node:{local(s_uri)}"}
            for p, o in ds.predicate_objects(subj):
                if str(p).startswith(EDGE):
                    node[f"edge:{local(p)}"] = {"@id": f"node:{local(o)}"}
            pri_uri = next(ds.objects(subj, EX.priority), None)
            if pri_uri:
                node["ex:priority"] = {"@id": f"ex:{local(pri_uri)}"}
            cat_uri = next(ds.objects(subj, EX.category), None)
            if cat_uri:
                node["ex:category"] = {"@id": f"ex:{local(cat_uri)}"}
            graph.append(node)

        for subj in ds.subjects(RDF.type, EX.Constraint):
            node = {"@id": f"con:{local(subj)}", "@type": "ex:Constraint"}
            s_uri = next(ds.objects(subj, RDF.subject), None)
            if s_uri:
                node["rdf:subject"] = {"@id": f"node:{local(s_uri)}"}
            for p, o in ds.predicate_objects(subj):
                if str(p).startswith(EDGE):
                    node[f"edge:{local(p)}"] = {"@id": f"node:{local(o)}"}
            ct_uri = next(ds.objects(subj, EX.constraintType), None)
            if ct_uri:
                node["ex:constraintType"] = {"@id": f"ex:{local(ct_uri)}"}
            graph.append(node)

        doc = {"@context": context, "@graph": graph}
        return yaml.dump(doc, allow_unicode=True, default_flow_style=False, sort_keys=True)

    def _get_triplets_yaml_ld(self) -> str:
        EX   = Namespace("http://example.org/ontologia#")
        EDGE = "http://example.org/edge/"
        ds   = self._cgraph._ds

        def local(uri):
            s = str(uri)
            return s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]

        context = {
            "ex":   "http://example.org/ontologia#",
            "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "node": "http://example.org/node/",
            "edge": "http://example.org/edge/",
            "tri":  "http://example.org/triplet/",
        }
        graph = []

        for subj in ds.subjects(RDF.type, EX.Triplet):
            node = {"@id": f"tri:{local(subj)}", "@type": "ex:Triplet"}
            s_uri = next(ds.objects(subj, RDF.subject), None)
            if s_uri:
                node["rdf:subject"] = {"@id": f"node:{local(s_uri)}"}
            for p, o in ds.predicate_objects(subj):
                if str(p).startswith(EDGE):
                    node[f"edge:{local(p)}"] = {"@id": f"node:{local(o)}"}
            for _, _, _, ctx in ds.quads((subj, RDF.type, EX.Triplet, None)):
                chunk_uri = next(ds.objects(ctx.identifier, PROV.wasDerivedFrom), None)
                if chunk_uri:
                    msg_uri = next(ds.objects(chunk_uri, PROV.wasDerivedFrom), None)
                    if msg_uri:
                        agent_uri = next(ds.objects(msg_uri, PROV.wasAttributedTo), None)
                        if agent_uri:
                            node["ex:extractedBy"] = {"@id": f"node:{local(agent_uri)}"}
                break
            graph.append(node)

        doc = {"@context": context, "@graph": graph}
        return yaml.dump(doc, allow_unicode=True, default_flow_style=False, sort_keys=True)

    def get_kg_context(self) -> str:
        if self._kg_format == "json":
            return (
                "=== REQUIREMENTS (JSON) ===\n"
                f"{self._get_requirements_text()}\n"
                "=== END REQUIREMENTS ===\n\n"
                "=== CONSTRAINTS (JSON) ===\n"
                f"{self._get_constraints_text()}\n"
                "=== END CONSTRAINTS ==="
            )
        if self._kg_format == "yaml-ld":
            return (
                "=== KNOWLEDGE GRAPH (YAML-LD) ===\n"
                f"{self._get_kg_yaml_ld()}\n"
                "=== END KNOWLEDGE GRAPH ==="
            )
        return (
            "=== KNOWLEDGE GRAPH (Turtle Light) ===\n"
            f"{self._get_kg_turtle_light()}\n"
            "=== END KNOWLEDGE GRAPH ==="
        )

    def add_message(self, mxg: Message):
        self._cgraph.add_message(mxg)

    def plan(self, task: str="", attempt:int=0, graph_in_prompt:bool=True, no_text:bool=False) -> dict:
        if attempt==0:
            logging.log(25, f"User asked: {task}")
            self._cgraph.add_message(Message.now(task, "User", "question", "default"))

        if graph_in_prompt:
            kg_context = self.get_kg_context()

            system = f"""You are an orchestrator managing a consortium responding to a public call for tenders.
                You have access to these specialized agents:
                {self._agent_registry()}

                ### CRITICAL CONSTRAINT — READ THIS FIRST:
                Each agent operates in COMPLETE ISOLATION. They have NO access to the Call for Tenders document.
                They can only answer based on their own internal knowledge (their company role and data).
                If your question does not contain the relevant facts from the tender, the agent will answer
                in a vacuum and produce a useless generic response.
                YOUR JOB IS TO BE THEIR EYES: copy every relevant requirement, figure, constraint and
                deadline from both the knowledge graph and the original tender text directly into the
                question you write for that agent.

                ### Your role:
                The user will provide you with two complementary sources:
                1. A structured Knowledge Graph (KG) extracted from the Call for Tenders, serialized in
                   {"Turtle Light format (simplified Turtle: no prefix declarations, subject-factorised, one line per subject)" if self._kg_format == "turtle-light" else "YAML-LD format (JSON-LD expressed in YAML syntax, using @context for namespace prefixes and @graph for nodes)" if self._kg_format == "yaml-ld" else "JSON format"}.
                   It contains two types of nodes:
                   - ex:Requirement — high-level business needs and goals the client wants to achieve.
                   - ex:Constraint  — conditions, limits and rules under which the solution must operate
                   (technical bounds, budget limits, regulatory requirements, infrastructure rules, etc.).
                2. The full original Call for Tenders text.
                Use the KG as the primary structured reference and the tender text to fill in any detail,
                context or nuance that the KG may not have captured.
                Your job is to dispatch targeted, self-contained questions to the relevant agents so that
                together they can produce a complete bid response.

                ### How to build each question:
                1. List every requirement (ex:Requirement) and every constraint (ex:Constraint) in the KG.
                2. Assign each one to the most appropriate agent based on its domain.
                3. For each agent, write a self-contained question that quotes verbatim every requirement
                and constraint assigned to it — copy names, standards, source types, figures and deadlines
                exactly as they appear in the KG and the tender text, without paraphrasing them away.
                4. End with a precise, answerable question about our company's capabilities or risks
                relative to those specific items.

                ### Rules:
                - Respond ONLY with a valid JSON object.
                - Keys must be agent names from the list above (use only agents relevant to this tender).
                - Values must be specific, self-contained questions derived from both sources.
                - COVERAGE: every requirement and every constraint from the KG must appear verbatim in
                at least one agent's question. Do not silently drop or summarise away any KG node.
                - When a node names a specific standard, source type, domain obligation, certification or
                technical specification, copy it exactly into the question — do not replace it with a
                generic description.
                - Each question MUST quote the exact figures and constraints
                (budget amounts, SLA targets, regulatory frameworks, technical specs, deadlines).
                - Do not include any explanation, markdown, or extra text — raw JSON only.
                - Questions will be asked in parallel, so each must be fully self-contained.
                - USE the right name for the Agents, do not modify them

                ### BAD example (never do this — agent has no context to answer):
                {{
                    "Budget Agent": "Given our budget constraints, can we deliver this project?"
                }}

                ### GOOD example (agent has everything it needs to answer):
                {{
                    "Budget Agent": "The client's global budget is [EXACT AMOUNT AND DURATION FROM TENDER]. Annual operating costs must remain controlled and compatible with the client's budgetary capacity. The tender requires a pilot phase followed by progressive rollout. Given our pricing model and current financial position, what is our projected margin on this contract, and are there cost-optimisation strategies we can propose?"
                }}

                Full example format:
                {{
                    "TechnicalArchitect Agent": "The client requires: response time below 2 seconds, [EXACT AVAILABILITY TARGET, including any 24/7 or off-hours constraint stated in the tender], API integration with the client's existing information systems and user directories, hosting on sovereign infrastructure certified to the level required by the tender within the EU, and no dependency on non-European suppliers. For each requirement state whether it is FULLY COVERED, PARTIALLY COVERED or NOT COVERED by our current stack, and describe the proposed architecture.",
                    "Budget Agent": "The client's global budget is [EXACT AMOUNT AND DURATION FROM TENDER]. Annual operating costs must remain controlled and compatible with the client's capacity. The tender requires [DEPLOYMENT STRUCTURE FROM TENDER]. Given our pricing model and current financial position, what is our projected margin, and what cost-optimisation strategies can we propose?",
                    "Legal Agent": "The tender requires strict GDPR compliance (EU-only data hosting, no transfer outside EU, encryption and pseudonymisation of sensitive data where applicable). All AI outputs must be explainable and auditable a posteriori. No decision may be automated without explicit validation by an authorised professional. [COPY HERE EVERY SECTOR-SPECIFIC REGULATORY OBLIGATION NAMED IN THE KG OR THE TENDER, VERBATIM.] What legal risks should we flag, and are we compliant?"
                }}"""
            if no_text:
                user_content = f"=== CALL FOR TENDERS KNOWLEDGE GRAPH ===\n\n{kg_context}"
            else:
                user_content = (
                    f"=== CALL FOR TENDERS KNOWLEDGE GRAPH ===\n\n{kg_context}\n\n"
                    f"=== ORIGINAL CALL FOR TENDERS TEXT ===\n\n{task}"
                )
        else:
            system =  f"""You are an orchestrator managing a consortium responding to a public call for tenders.
                You have access to these specialized agents:
                {self._agent_registry()}

                ### CRITICAL CONSTRAINT — READ THIS FIRST:
                Each agent operates in COMPLETE ISOLATION. They have NO access to the Call for Tenders document.
                They can only answer based on their own internal knowledge (their company role and data).
                If your question does not contain the relevant facts from the tender, the agent will answer
                in a vacuum and produce a useless generic response.
                YOUR JOB IS TO BE THEIR EYES: copy every relevant requirement, figure, constraint and
                deadline from the tender directly into the question you write for that agent.

                ### Your role:
                The user will provide you with the full text of a Call for Tenders document.
                Your job is to read it carefully and dispatch targeted, self-contained questions
                to the relevant agents so that together they can produce a complete bid response.

                ### How to build each question:
                1. Read the entire tender and list every distinct requirement and constraint it contains.
                2. Assign each one to the most appropriate agent based on its domain.
                3. For each agent, write a self-contained question that quotes verbatim every requirement
                and constraint assigned to it — copy names, standards, source types, figures and deadlines
                exactly as they appear in the tender, without paraphrasing them away.
                4. End with a precise, answerable question about our company's capabilities or risks
                relative to those specific items.

                ### Rules:
                - Respond ONLY with a valid JSON object.
                - Keys must be agent names from the list above (use only agents relevant to this tender).
                - Values must be specific, self-contained questions derived from the Call for Tenders text.
                - COVERAGE: every requirement and every constraint from the tender must appear verbatim in
                at least one agent's question. Do not silently drop or summarise away any item.
                - When an item names a specific standard, source type, domain obligation, certification or
                technical specification, copy it exactly into the question — do not replace it with a
                generic description.
                - Each question MUST quote the exact figures and constraints from the tender
                (budget amounts, SLA targets, regulatory frameworks, technical specs, deadlines).
                - Do not include any explanation, markdown, or extra text — raw JSON only.
                - Questions will be asked in parallel, so each must be fully self-contained.
                - USE the right name for the Agents, do not modify them

                ### BAD example (never do this — agent has no context to answer):
                {{
                    "Budget Agent": "Given our budget constraints, can we deliver this project?"
                }}

                ### GOOD example (agent has everything it needs to answer):
                {{
                    "Budget Agent": "The client's global budget is [EXACT AMOUNT AND DURATION FROM TENDER]. Annual operating costs must remain controlled and compatible with the client's budgetary capacity. The tender requires a pilot phase followed by progressive rollout. Given our pricing model and current financial position, what is our projected margin on this contract, and are there cost-optimisation strategies we can propose?"
                }}

                Full example format:
                {{
                    "TechnicalArchitect Agent": "The client requires: response time below 2 seconds, [EXACT AVAILABILITY TARGET, including any 24/7 or off-hours constraint stated in the tender], API integration with the client's existing information systems and user directories, hosting on sovereign infrastructure certified to the level required by the tender within the EU, and no dependency on non-European suppliers. For each requirement state whether it is FULLY COVERED, PARTIALLY COVERED or NOT COVERED by our current stack, and describe the proposed architecture.",
                    "Budget Agent": "The client's global budget is [EXACT AMOUNT AND DURATION FROM TENDER]. Annual operating costs must remain controlled and compatible with the client's capacity. The tender requires [DEPLOYMENT STRUCTURE FROM TENDER]. Given our pricing model and current financial position, what is our projected margin, and what cost-optimisation strategies can we propose?",
                    "Legal Agent": "The tender requires strict GDPR compliance (EU-only data hosting, no transfer outside EU, encryption and pseudonymisation of sensitive data where applicable). All AI outputs must be explainable and auditable a posteriori. No decision may be automated without explicit validation by an authorised professional. [COPY HERE EVERY SECTOR-SPECIFIC REGULATORY OBLIGATION NAMED IN THE TENDER, VERBATIM.] What legal risks should we flag, and are we compliant?"
                }}"""
            user_content = f"Call for Tenders:\n\n{task}"

        response = ollama_chat(
            self.model,
            [{"role": "system", "content": system}, {"role": "user", "content": user_content}],
        )

        raw = response["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            plan = json.loads(raw)
            logging.log(25, f"Orchestrator Answer: {raw}")
        except json.JSONDecodeError:
            logging.warning(f"[orchestrator] Could not parse plan JSON (attempt {attempt}). Raw: {raw}")
            plan = {}

        self._cgraph.add_message(Message.now(raw, "Orchestrator", "answer", "default"))
        return plan


    def correct_answer(self, name, answer, question):
        text = f"""### Task: Evaluation
            Does the provided Answer satisfy the original Question asked to the agent "{name}"?

            **Question:** {question}
            **Answer:** {answer}

            ### Instructions:
            - Respond with "TRUE" if the answer is accurate and complete.
            - Respond with "FALSE" if the answer is incorrect, incomplete, or irrelevant.
            - Provide ONLY the word "TRUE" or "FALSE". No other text.

            Result:"""
        logging.log(25,f"Orchestrator correct received: {answer}")
        response = ollama_chat(self.model, [{'role': 'user', 'content': text}])
        self.agent_answer.append(f"=== {name} ===\n{answer}")
        textual_answer= response['message']['content']
        cleaned=textual_answer.lower().replace(".","").strip()
        if cleaned== "false":
            logging.log(25,"Orchestrator correct answered: FALSE")
            return False
        elif cleaned== "true":
            logging.log(25,"Orchestrator correct answered: TRUE")
            return True
        return False

    def propose(self, task: str, use_triplets: bool = False, no_text: bool = False) -> str:
        # no_text only applies where the KG is actually injected into the proposal, i.e.
        # when triplets/KG are added (use_triplets). Otherwise treat it as the text option.
        no_text = no_text and use_triplets
        agents_context = "" if no_text else "\n\n".join(self.agent_answer)

        if no_text:
            source = "the extracted triplets and the requirements/constraints of the knowledge graph"
            system = f"""You are writing a bid response on behalf of a consortium.

Write a single, coherent proposal text that integrates all the information from {source} into a flowing, professional response to the call for tenders.

Rules (no exceptions):
- Use ONLY information explicitly stated in {source}. Do not invent anything.
- Copy exact figures, technologies, costs, regulations, and deadlines from {source}.
- No filler sentences ("we are pleased to", "our team is committed to", etc.).
- No bullet-point lists of requirements. Write flowing prose.
- Output only the proposal text."""
        else:
            system = """You are writing a bid response on behalf of a consortium.

Write a single, coherent proposal text that integrates all the information from the agents' answers into a flowing, professional response to the call for tenders.

Rules (no exceptions):
- Use ONLY information explicitly stated in the agents' answers. Do not invent anything.
- Copy exact figures, technologies, costs, regulations, and deadlines from the agents' answers.
- No filler sentences ("we are pleased to", "our team is committed to", etc.).
- No bullet-point lists of requirements. Write flowing prose.
- Output only the proposal text."""

        triplets_section = ""
        kg_section = ""
        if use_triplets:
            if self._kg_format == "json":
                triplets_content = self._get_triplets_json()
            elif self._kg_format == "yaml-ld":
                triplets_content = self._get_triplets_yaml_ld()
            else:
                triplets_content = self._get_triplets_text()
            triplets_section = f"\n\n=== TRIPLETS EXTRACTED FROM AGENT CONVERSATIONS ===\n{triplets_content}\n=== END TRIPLETS ==="
            kg_section = (
                f"\n\n=== REQUIREMENTS AND CONSTRAINTS — address and respect each one ===\n"
                f"{self.get_kg_context()}\n"
                f"=== END REQUIREMENTS AND CONSTRAINTS ==="
            )

        cft_section = "" if no_text else f"=== CALL FOR TENDERS ===\n{task}\n\n"
        agents_section = "" if not agents_context else f"=== AGENTS' ANSWERS ===\n{agents_context}"

        final_instruction = (
            "Write the proposal now. Use the extracted triplets and the knowledge-graph "
            "requirements and constraints as your only source. "
            if no_text else
            "Write the proposal now. Use the agents' answers as your only source. "
        )
        user_message = (
            f"{cft_section}"
            f"{agents_section}"
            f"{triplets_section}"
            f"{kg_section}\n\n"
            f"{final_instruction}"
            "Produce a coherent, flowing text that responds to the call for tenders."
        )

        response = ollama_chat(
            self.model,
            [{"role": "system", "content": system}, {"role": "user", "content": user_message}],
        )

        proposal = response["message"]["content"]
        logging.log(25, f"Final proposal generated: {proposal}")
        self.agent_answer = []
        return proposal
