from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import pandas as pd
import os
import json
from typing import List

# Função para buscar dados do estudante no arquivo CSV
def busca_dados_estudante(estudante):
    dados = pd.read_csv("documents/estudantes.csv")  # Lê o arquivo CSV com os dados dos estudantes
    dados_com_esse_estudante = dados[dados['USUARIO'] == estudante]  # Filtra pelo nome do estudante
    if dados_com_esse_estudante.empty:
        return {}  # Retorna dicionário vazio se não encontrar
    return dados_com_esse_estudante.iloc[:1].to_dict()  # Retorna os dados do primeiro registro encontrado

# Modelo Pydantic para extrair o nome do estudante
class ExtratorDeEstudante(BaseModel):
    estudante: str = Field("Nome do estudante informado, sempre em letras minusculas.")

# Classe principal da ferramenta
class DadosDeEstudantes(BaseTool):
    """
    Classe que representa uma ferramenta para extrair informações do histórico e preferências de um estudante.
    """
    name: str = "DadosDeEstudante"
    description: str = """Essa ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico.
    Passe para essa ferramenta como argumento o primeiro nome do estudante."""
    
    # Método principal que executa o fluxo de extração
    def _run(self, input: str) -> str:
        # Inicializa o modelo de linguagem da Google com a chave de API
        LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
        
        # Define o parser para extrair o nome do estudante no formato JSON
        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)
        
        # Cria o prompt para o modelo de linguagem, instruindo como extrair o nome
        template = PromptTemplate(
            template="""Você deve analisar a entrada a seguir e extrair o nome informado em minúsculo.
            Entrada:
            -----------------
            {input}
            -----------------
                       Formato de saída:
                       {formato_saida}""",
            input_variables=["input"],
            partial_variables={"formato_saida": parser.get_format_instructions()}
        )
        
        # Monta a cadeia de execução: prompt -> LLM -> parser
        chain = template | LLM | parser
        
        # Executa a cadeia com o texto de entrada e obtém o resultado
        result = chain.invoke({"input": input})
        
        # Extrai o nome do estudante do resultado e converte para minúsculas
        estudante = result['estudante']
        estudante = estudante.lower().strip()
        # estudante = input.lower().strip()
        # Busca os dados do estudante no CSV
        dados = busca_dados_estudante(estudante)
        
        # Retorna os dados encontrados em formato JSON
        return json.dumps(dados)
    
class Nota(BaseModel):
    area: str = Field("Nome da área de conhecimento")
    nota: float = Field("Nota na área de conhecimento")
    
class PerfilAcademicoDeEstudante(BaseModel):
    nome: str = Field("Nome do estudante")
    ano_de_conclusao: str = Field("Ano de conclusão")
    notas:List[Nota] = Field("Lista de notas da disciplina do estudante e áreas de conhecimento.")
    resumo: str = Field("Resumo das principais características desse estudante de forma a torná-lo único e com um ótimo potencial estudante para faculdades. Exemplo: Só esse estudante tem tal coisa.")
class PerfilAcademico(BaseTool):
    name: str = "PerfilAcademico"
    description: str = """Criar um perfil acadêmico de um estudante. Esta ferramenta requer como entrada todos os dados do estudante.
    Eu sou incapaz de buscar os dados do estudante.
    você tem que buscar os dados do estudante antes de me invocar.
    """
    def _run(self, input: str) -> str:
        LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
        
        # Define o parser para extrair o nome do estudante no formato JSON
        parser = JsonOutputParser(pydantic_object=PerfilAcademicoDeEstudante)
        template = PromptTemplate(template="""
            - Formate o estudante para seu perfil acadêmico.
            - Com os dados, identifique as opções de universidades sugeridas e cursos compativéis com o interesse do aluno.
            - Destaque o perfil do aluno dando ênfase principalmente naquilo que faz sentido para as instituições de interesse.
            
            Persona: Você é uma consultora de carreiras e precisa indicar com detalhes, riqueza, mas direta ao ponto, para o estudante as opções e consequências possiveis.
            Informações atuais:
            
            {dados_do_estudante}
            {formato_de_saida}
            """,
            input_variables=["dados_do_estudante"],
            partial_variables={"formato_de_saida": parser.get_format_instructions()})
        chain = template | LLM | parser
        result = chain.invoke({"dados_do_estudante": input})
        return result