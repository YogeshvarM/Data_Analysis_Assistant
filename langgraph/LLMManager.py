from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class LLMManager:
    def __init__(self, api_key: str = None):
        if not api_key:
            raise ValueError(
                "Groq API key is required. Please provide your API key when initializing LLMManager."
            )
        
        self.llm = ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile", 
            temperature=0
        )

    def invoke(self, prompt: ChatPromptTemplate, **kwargs) -> str:
        messages = prompt.format_messages(**kwargs)
        response = self.llm.invoke(messages)
        return response.content