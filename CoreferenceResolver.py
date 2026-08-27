from fastcoref import FCoref
class CoreferenceResolver:
    def __init__(self, device:str="cpu"):
        self.model=FCoref(device=device)
        self.pronouns={"he", "she", "it", "they", "him", "her", "them", 
                       "his", "hers", "its", "their", "theirs",
                        "himself", "herself", "itself", "themselves", "this","that"}
    def clean(self,text:str)->str:
        text=text.replace("\n\n\n", " ").replace("\n\n", " ").replace("\n", " ")
        return text
    
    def resolve(self, text:str)->str:
        text=self.clean(text)
        predictions=self.model.predict(text) 
        clusters_str=predictions.get_clusters(as_strings=True)
        clusters_span= predictions.get_clusters(as_strings=False)
        cleaned_text=self._replace_pronouns(text, clusters_span, clusters_str)
        return cleaned_text
    
    def _replace_pronouns(self, text, clusters_span, cluster_str):
        replacements={} 

        for spans, strings in zip(clusters_span,cluster_str):
            if len(spans)<2: 
                continue

            head=strings[0] 
            for span, string in zip(spans[1:], strings[1:]):
                replacements[span[0]]=[span[1],head]
        
        if not replacements:
            return text
        
        chars=list(text) 
        for start in sorted(replacements.keys(), reverse=True): 
            chars[start:replacements[start][0]]=list(replacements[start][1])
        
        return "".join(chars)


