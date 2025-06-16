from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from estudante import DadosDeEstudantes, PerfilAcademico
from university import DadosDeUniversidade, TodasUniversidades
from langchain import hub
import os

load_dotenv()
class AgentGeminiFunctions:
    def __init__(self):
        LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
        
        dados_de_estudante = DadosDeEstudantes()
        perfil_academico = PerfilAcademico()
        dados_de_universidade = DadosDeUniversidade()
        todas_universidades = TodasUniversidades()
        self.tools = [
            Tool(name = dados_de_estudante.name,
                func = dados_de_estudante._run,
                description = dados_de_estudante.description),
            Tool(name= perfil_academico.name,
                func=perfil_academico._run,
                description=perfil_academico.description),
            Tool(name= dados_de_universidade.name,
                func=dados_de_universidade._run,
                description=dados_de_universidade.description),
            Tool(name= todas_universidades.name,
                func=todas_universidades._run,
                description=todas_universidades.description)
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Obtém um prompt pré-definido do LangChain Hub para o agente ReAct
        prompt = hub.pull("hwchase17/react")
        
        # Cria o agente usando o modelo LLM, as ferramentas e o prompt.
        # create_react_agent cria um agente que segue o padrão ReAct (Reasoning and Acting).
        self.agent  = create_react_agent(LLM, self.tools, prompt)

        # Alternativamente, poderia ser usado create_tool_calling_agent para um agente que chama ferramentas via funções.
        # self.agent  = create_tool_calling_agent(LLM, self.tools, prompt)