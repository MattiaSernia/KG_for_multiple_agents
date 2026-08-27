import logging
from rdflib import ConjunctiveGraph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD, PROV
from constraintsExtractor import ConstraintsExtractor
from requirementsExtractor import RequirementsExtractor
from proposalsExtractor import ProposalsExtractor
from tripletExtractorClaude import TripletExtractor
from phi4TripletExtractor import Phi4TripletExtractor
from phi4RequirementsExtractor import Phi4RequirementsExtractor
from phi4ConstraintsExtractor import Phi4ConstraintsExtractor
from phi4ProposalsExtractor import Phi4ProposalsExtractor
from CoreferenceResolver import CoreferenceResolver
from mxg import Message






import os
import re

class Custom_Graph:

    def __init__(self, agent_list:list, name:str, model:str="llama3.3:70b", chunk_dim:int=0, extractor_type:str="llama", no_schema:bool=False):
        self._EX  = Namespace("http://example.org/ontologia#")
        self._REQ = Namespace("http://example.org/requirement/")
        self._EXT = Namespace("http://example.org/extraction/")
        self._CON = Namespace("http://example.org/constraint/")
        self._PRO = Namespace("http://example.org/proposal/")
        self._TRI = Namespace("http://example.org/triplet/")

        self._ds = ConjunctiveGraph()
        self._ds.bind("ex",  self._EX)
        self._ds.bind("req", self._REQ)
        self._ds.bind("ext", self._EXT)
        self._ds.bind("prov",PROV)
        self._ds.bind("con", self._CON)
        self._ds.bind("pro", self._PRO)
        self._ds.bind("tri", self._TRI)

        self._agent_list=agent_list

        self._nodeUri="http://example.org/node/"
        self._activityUri = "http://example.org/activity/"
        self._extractionUri= "http://example.org/extraction/"
        self._edgeUri="http://example.org/edge/"

        self._dict={}

        self._activity_counter=0
        self._extraction_counter=0
        self._requirement_counter=0
        self._constraint_counter=0
        self._proposal_counter=0
        self._triplet_counter=0
        self._chunk_counter=0

        self._prev_message = None
        self._prev_message_uri = None
        self._first_message_uri = None
        self._mex = 0

        _coref = CoreferenceResolver()
        _phi4  = "hf.co/FinaPolat/phi4_adaptableIE_v2-gguf:Q4_K_M"
        if extractor_type == "phi4":
            self._req_extr  = Phi4RequirementsExtractor(_phi4, _coref, chunk_dim, no_schema)
            self._con_extr  = Phi4ConstraintsExtractor(_phi4, _coref, chunk_dim, no_schema)
            self._pro_extr  = Phi4ProposalsExtractor(_phi4, _coref, chunk_dim, no_schema)
            self._extractor = Phi4TripletExtractor(_phi4, _coref, chunk_dim, no_schema)
        else:
            self._req_extr  = RequirementsExtractor(model, 0, _coref, chunk_dim)
            self._con_extr  = ConstraintsExtractor(model, 0, _coref, chunk_dim)
            self._pro_extr  = ProposalsExtractor(model, 0, _coref, chunk_dim)
            self._extractor = TripletExtractor(model, 0, _coref, chunk_dim)

        self._name=name

    def clear(self):
        self._ds = ConjunctiveGraph()
        self._ds.bind("ex",  self._EX)
        self._ds.bind("req", self._REQ)
        self._ds.bind("ext", self._EXT)
        self._ds.bind("prov", PROV)
        self._ds.bind("con", self._CON)
        self._ds.bind("pro", self._PRO)
        self._ds.bind("tri", self._TRI)
        self._dict = {}
        self._mex = 0
        self._activity_counter    = 0
        self._extraction_counter  = 0
        self._requirement_counter = 0
        self._constraint_counter  = 0
        self._proposal_counter    = 0
        self._triplet_counter     = 0
        self._chunk_counter       = 0
        self._prev_message        = None
        self._prev_message_uri    = None
        self._first_message_uri   = None

    def add_message(self, mxg: Message):
        """Add a single message to the graph in real-time, then save."""
        if mxg.role == "default":
            URIagent = URIRef(self._nodeUri + self._clean_uri(mxg.node))
            if (URIagent, RDF.type, PROV.Agent, self._ds.default_context) not in self._ds:
                self._ds.add((URIagent, RDF.type, PROV.Agent, self._ds.default_context))

            if mxg.text in self._dict:
                URImxg = self._dict[mxg.text]
                if (URImxg, self._EX.is_coherent, Literal('FALSE'), self._ds.default_context) in self._ds \
                   or (URImxg, self._EX.does_answer, Literal('FALSE'), self._ds.default_context) in self._ds:
                    self._remove_derived_graphs(URImxg)
                    self._ds.remove((URImxg, None, None, self._ds.default_context))
                    self._ds.remove((None, None, URImxg, self._ds.default_context))
                else:
                
                    self._prev_message = mxg
                    self._prev_message_uri = URImxg
                    self._mex += 1
                    return
            else:
                URImxg = URIRef(self._nodeUri + f"message{len(self._dict.keys())}")
                self._dict[mxg.text] = URImxg

            self._ds.add((URImxg, RDF.type, PROV.Entity, self._ds.default_context))

            URIactivity = self._new_activity_uri()
            self._ds.add((URIactivity, RDF.type, PROV.Activity, self._ds.default_context))
            self._ds.add((URIactivity, PROV.wasAssociatedWith, URIagent, self._ds.default_context))
            self._ds.add((URIactivity, PROV.startedAtTime, Literal(mxg.timestamp, datatype=XSD.dateTime), self._ds.default_context))
            self._ds.add((URImxg, PROV.wasGeneratedBy, URIactivity, self._ds.default_context))
            self._ds.add((URImxg, PROV.wasAttributedTo, URIagent, self._ds.default_context))
            self._ds.add((URImxg, PROV.generatedAtTime, Literal(mxg.timestamp, datatype=XSD.dateTime), self._ds.default_context))

            text = mxg.text

            if mxg.convPart == "question":
                
                pass
            elif mxg.convPart == "answer":
                if self._prev_message is not None:
                    URIprev = URIRef(self._nodeUri + self._clean_uri(self._prev_message.node))
                    self._ds.add((URImxg, self._EX.sent_To, URIprev, self._ds.default_context))
                    if self._prev_message_uri is not None:
                        URIanswerer = URIRef(self._nodeUri + self._clean_uri(mxg.node))
                        self._ds.add((self._prev_message_uri, self._EX.sent_To, URIanswerer, self._ds.default_context))
                    if self._prev_message.text in self._dict:
                        URIprevMsg = self._dict[self._prev_message.text]
                        self._ds.add((URImxg, PROV.wasDerivedFrom, URIprevMsg, self._ds.default_context))
                        self._ds.add((URIactivity, PROV.used, URIprevMsg, self._ds.default_context))
                    text = self._prev_message.text + "\n" + text
            elif mxg.convPart == "proposal":
                if self._prev_message is not None:
                    URIprev = URIRef(self._nodeUri + self._clean_uri(self._prev_message.node))
                    self._ds.add((URImxg, self._EX.sent_To, URIprev, self._ds.default_context))
                    if self._prev_message_uri is not None and self._first_message_uri is not None:
                        self._ds.add((URImxg, PROV.wasDerivedFrom, self._first_message_uri, self._ds.default_context))
                        self._ds.add((URIactivity, PROV.used, self._first_message_uri, self._ds.default_context))

            if mxg.node == "User":
                self._userExtraction(text, URImxg)
            elif mxg.convPart == "proposal":
                self._propExtraction(text, URImxg)
            elif mxg.node != "Orchestrator":
                self._normalExtraction(text, URImxg)

            if self._first_message_uri is None:
                self._first_message_uri = URImxg
            self._prev_message = mxg
            self._prev_message_uri = URImxg

        elif mxg.role == "coherency":
            if mxg.convPart == "answer" and self._prev_message_uri is not None:
                self._ds.add((self._prev_message_uri, self._EX.is_coherent, Literal(mxg.text), self._ds.default_context))

        elif mxg.role == "correction":
            if mxg.convPart == "answer" and self._prev_message_uri is not None:
                self._ds.add((self._prev_message_uri, self._EX.does_answer, Literal(mxg.text.lstrip().rstrip()), self._ds.default_context))

        self._mex += 1
        self._saveGraph()

    def _userExtraction(self, text: str, URImxg):
        chunks=self._req_extr.pipe(text)
        for chunk in chunks.keys():
            ChunkURI=self._new_chunk_uri()
            ReqURI=self._new_extraction_uri("req/")
            ng1=self._ds.get_context(ReqURI)
            for requirement in chunks[chunk]:
                node=self._new_requirement_uri()
                ng1.add((node, RDF.type, self._EX.Requirement))
                ng1.add((node, RDF.subject,    URIRef(self._nodeUri + self._clean_uri(requirement["subject"]))))
                ng1.add((node, URIRef(self._edgeUri + self._clean_uri(requirement["predicate"])),    URIRef(self._nodeUri + self._clean_uri(requirement["object"]))))
                if requirement["priority"]:
                    ng1.add((node, self._EX.priority,   self._EX[self._clean_uri(requirement["priority"])]))
                if requirement["category"]:
                    ng1.add((node, self._EX.category,   self._EX[self._clean_uri(requirement["category"])]))
            self._ds.add((ReqURI, RDF.type, self._EX.Extraction,   self._ds.default_context))
            self._ds.add((ReqURI, PROV.wasDerivedFrom, ChunkURI,   self._ds.default_context))
            self._ds.add((ChunkURI, PROV.wasDerivedFrom,URImxg ,   self._ds.default_context))
            self._ds.add((ChunkURI, RDF.type ,PROV.Entity ,   self._ds.default_context))
            self._ds.add((ChunkURI, self._EX.hasText ,Literal(chunk) ,   self._ds.default_context))




        chunks_con=self._con_extr.pipe(text)
        for chunk in chunks_con.keys():
            ChunkURI=self._new_chunk_uri()
            ConURI=self._new_extraction_uri("con/")
            ng2=self._ds.get_context(ConURI)
            for constraint in chunks_con[chunk]:
                node=self._new_constraint_uri()
                ng2.add((node, RDF.type, self._EX.Constraint))
                ng2.add((node, RDF.subject,    URIRef(self._nodeUri + self._clean_uri(constraint["subject"]))))
                ng2.add((node, URIRef(self._edgeUri + self._clean_uri(constraint["predicate"])),    URIRef(self._nodeUri + self._clean_uri(constraint["object"]))))
                if constraint["constraintType"]:
                    ng2.add((node, self._EX.constraintType,   self._EX[self._clean_uri(constraint["constraintType"])]))
            self._ds.add((ConURI, RDF.type, self._EX.Extraction,   self._ds.default_context))
            self._ds.add((ConURI, PROV.wasDerivedFrom, ChunkURI,   self._ds.default_context))
            self._ds.add((ChunkURI, PROV.wasDerivedFrom, URImxg,   self._ds.default_context))
            self._ds.add((ChunkURI, RDF.type, PROV.Entity,   self._ds.default_context))
            self._ds.add((ChunkURI, self._EX.hasText, Literal(chunk),   self._ds.default_context))

    def _clean_uri(self, label: str) -> str:
        label = label.strip().lower()
        label = re.sub(r'[^a-zA-Z0-9]', '_', label)
        label = re.sub(r'_+', '_', label)
        return label.strip('_')     

    def _new_activity_uri(self) -> URIRef:
        self._activity_counter += 1
        return URIRef(self._activityUri + f"activity{self._activity_counter}")

    def _new_extraction_uri(self, string:str="") -> URIRef:
        self._extraction_counter += 1
        return URIRef(self._extractionUri + f"{string}extraction{self._extraction_counter}")
    
    def _new_requirement_uri(self) -> URIRef:
        self._requirement_counter += 1
        return self._REQ[f"req{self._requirement_counter}"]

    def _new_constraint_uri(self) -> URIRef:
        self._constraint_counter += 1
        return self._CON[f"con{self._constraint_counter}"]
  
    def _new_chunk_uri(self) -> URIRef:
        self._chunk_counter += 1
        return URIRef(self._nodeUri + f"chunk{self._chunk_counter}")

    def _propExtraction(self, text:str, URImxg):
        chunks_pro=self._pro_extr.pipe(text)
        for chunk in chunks_pro.keys():
            ChunkURI=self._new_chunk_uri()
            ProURI=self._new_extraction_uri("pro/")
            ng=self._ds.get_context(ProURI)
            for proposal in chunks_pro[chunk]:
                node=self._new_proposal_uri()
                ng.add((node, RDF.type, self._EX.Proposal))
                ng.add((node, RDF.subject,    URIRef(self._nodeUri + self._clean_uri(proposal["subject"]))))
                ng.add((node, URIRef(self._edgeUri + self._clean_uri(proposal["predicate"])),    URIRef(self._nodeUri + self._clean_uri(proposal["object"]))))
            self._ds.add((ProURI, RDF.type, self._EX.Extraction,   self._ds.default_context))
            self._ds.add((ProURI, PROV.wasDerivedFrom, ChunkURI,   self._ds.default_context))
            self._ds.add((ChunkURI, PROV.wasDerivedFrom, URImxg,   self._ds.default_context))
            self._ds.add((ChunkURI, RDF.type, PROV.Entity,   self._ds.default_context))
            self._ds.add((ChunkURI, self._EX.hasText, Literal(chunk),   self._ds.default_context))

    def _normalExtraction(self, text:str, URImxg):
        chunks_tri=self._extractor.pipe(text)
        for chunk in chunks_tri.keys():
            ChunkURI=self._new_chunk_uri()
            TriURI=self._new_extraction_uri("tri/")
            ng=self._ds.get_context(TriURI)
            for triplet in chunks_tri[chunk]:
                node=self._new_triplet_uri()
                ng.add((node, RDF.type, self._EX.Triplet))
                ng.add((node, RDF.subject,    URIRef(self._nodeUri + self._clean_uri(triplet["subject"]))))
                ng.add((node, URIRef(self._edgeUri + self._clean_uri(triplet["predicate"])),    URIRef(self._nodeUri + self._clean_uri(triplet["object"]))))
            self._ds.add((TriURI, RDF.type, self._EX.Extraction,   self._ds.default_context))
            self._ds.add((TriURI, PROV.wasDerivedFrom, ChunkURI,   self._ds.default_context))
            self._ds.add((ChunkURI, PROV.wasDerivedFrom, URImxg,   self._ds.default_context))
            self._ds.add((ChunkURI, RDF.type, PROV.Entity,   self._ds.default_context))
            self._ds.add((ChunkURI, self._EX.hasText, Literal(chunk),   self._ds.default_context))
    
    def _new_proposal_uri(self)->URIRef:
        self._proposal_counter += 1
        return self._PRO[f"pro{self._proposal_counter}"]
    
    def _new_triplet_uri(self) -> URIRef:
        self._triplet_counter += 1
        return self._TRI[f"tri{self._triplet_counter}"]

    def _saveGraph(self):
        filepath=f"{self._name}.trig"
        trig_str = self._ds.serialize(format="trig")
        if os.path.exists(filepath):
            os.remove(filepath)
        trig_str = re.sub(
            r'(?m)^{$',
            'GRAPH <urn:x-arq:DefaultGraph> {',
            trig_str
        )
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(trig_str)
        logging.debug(f"Graph saved to {filepath}")

    def _remove_derived_graphs(self, URImxg):
        """Remove all chunks and their derived extraction graphs linked to this message."""
        chunk_uris = list(self._ds.subjects(PROV.wasDerivedFrom, URImxg))
        for chunk_uri in chunk_uris:
            extraction_uris = list(self._ds.subjects(PROV.wasDerivedFrom, chunk_uri))
            for ext_uri in extraction_uris:
                self._ds.remove_context(self._ds.get_context(ext_uri))
                self._ds.remove((ext_uri, None, None, self._ds.default_context))
            self._ds.remove((chunk_uri, None, None, self._ds.default_context))

    def mxgnr(self):
        return self._mex

    def rename(self, name:str):
        self._name=name