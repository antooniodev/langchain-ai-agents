from langchain.tools import BaseTool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain.agents import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
import os
import json
load_dotenv()

def busca_dados_estudante(estudante):
    dados = pd.read_csv("documents/estudantes.csv")
    dados_com_esse_estudante = dados[dados['USUARIO'] == estudante]
    if dados_com_esse_estudante.empty:
        return {}
    return dados_com_esse_estudante.iloc[:1].to_dict()
class ExtratorDeEstudante(BaseModel):
    estudante: str = Field("Nome do estudante informado, sempre em letras minusculas. Exemplo: ana, joão, carlos.")

class DadosDeEstudantes(BaseTool):
    """
    Classe que representa uma ferramenta para extrair informações do histórico e preferências de um estudante.
    Atributos:
        name (str): Nome da ferramenta.
        description (str): Descrição da funcionalidade da ferramenta.
    Métodos:
        _run(input: str) -> str:
            Executa o fluxo de extração de dados do estudante a partir de um texto de entrada.
            Utiliza um modelo de linguagem para analisar o texto, extrair o nome do estudante e retornar o resultado no formato especificado.
    """
    # O método _run executa o fluxo principal de extração de dados do estudante.
    name: str = "DadosDeEstudante"
    description: str = """Essa ferramenta extrai o histórico e preferências  de um estudante de acordo com seu histórico."""
    
    def _run(self, input: str) -> str:
        LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)
        template = PromptTemplate(
            template="""Você deve analisar a {input} e extrair o nome do usuário informado.
                       Formato de saída:
                       {formato_saida}""",
            input_variables=["input"],
            partial_variables={"formato_saida": parser.get_format_instructions()}
        )
        chain = template | LLM | parser
        result = chain.invoke({"input": input})
        estudante = result['estudante']
        dados = busca_dados_estudante(estudante)
        return json.dumps(dados)
        
    
pergunta = "Quais os dados da Bianca? Busque pelo nome."

LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
dados_de_estudante = DadosDeEstudantes()
tools = [
    Tool(name = dados_de_estudante.name,
         func = dados_de_estudante._run,
         description = dados_de_estudante.description) 
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
print(prompt)
agent  = create_tool_calling_agent(LLM, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": pergunta})
print(result)
