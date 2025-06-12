from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from estudante import DadosDeEstudantes, PerfilAcademico
import os

load_dotenv()
class AgentGeminiFunctions:
    def __init__(self):
        LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
        
        dados_de_estudante = DadosDeEstudantes()
        perfil_academico = PerfilAcademico()
        self.tools = [
            Tool(name = dados_de_estudante.name,
                func = dados_de_estudante._run,
                description = dados_de_estudante.description),
            Tool(name= perfil_academico.name,
                func=perfil_academico._run,
                description=perfil_academico.description)
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.agent  = create_tool_calling_agent(LLM, self.tools, prompt)