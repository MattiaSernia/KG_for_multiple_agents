from datetime import datetime
class Message:
    def __init__(self, text:str, timestamp:str, node:str, convPart:str, role:str):
        self.text = text
        self.timestamp = datetime.strptime(timestamp.split(',')[0], "%Y-%m-%d %H:%M:%S")
        self.node = node
        self.convPart = convPart
        self.role = role

    @classmethod
    def now(cls, text:str, node:str, convPart:str, role:str) -> "Message":
        return cls(text, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), node, convPart, role)

    def toString(self)->str:
        return f"Corp: {self.text}\nTimestamp: {self.timestamp}\nNode: {self.node}\nconvPart: {self.convPart}\nRole: {self.role}\n\n"