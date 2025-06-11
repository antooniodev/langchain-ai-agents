from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import pandas as pd
import os
import json

# Função para buscar dados do estudante no arquivo CSV
def busca_dados_estudante(estudante):
    dados = pd.read_csv("documents/estudantes.csv")  # Lê o arquivo CSV com os dados dos estudantes
    dados_com_esse_estudante = dados[dados['USUARIO'] == estudante]  # Filtra pelo nome do estudante
    if dados_com_esse_estudante.empty:
        return {}  # Retorna dicionário vazio se não encontrar
    return dados_com_esse_estudante.iloc[:1].to_dict()  # Retorna os dados do primeiro registro encontrado

# Modelo Pydantic para extrair o nome do estudante
class ExtratorDeEstudante(BaseModel):
    estudante: str = Field("Nome do estudante informado, sempre em letras minusculas. Exemplo: ana, joão, carlos.")

# Classe principal da ferramenta
class DadosDeEstudantes(BaseTool):
    """
    Classe que representa uma ferramenta para extrair informações do histórico e preferências de um estudante.
    """
    name: str = "DadosDeEstudante"
    description: str = """Essa ferramenta extrai o histórico e preferências  de um estudante de acordo com seu histórico."""
    
    # Método principal que executa o fluxo de extração
    def _run(self, input: str) -> str:
        # Inicializa o modelo de linguagem da Google com a chave de API
        LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
        
        # Define o parser para extrair o nome do estudante no formato JSON
        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)
        
        # Cria o prompt para o modelo de linguagem, instruindo como extrair o nome
        template = PromptTemplate(
            template="""Você deve analisar a {input} e extrair o nome do usuário informado.
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
        estudante = estudante.lower()
        
        # Busca os dados do estudante no CSV
        dados = busca_dados_estudante(estudante)
        
        # Retorna os dados encontrados em formato JSON
        return json.dumps(dados)