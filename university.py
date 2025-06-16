from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import pandas as pd
import os
import json
from typing import List

# Função para buscar dados de uma universidade específica no arquivo CSV
def busca_dados_da_universidade(universidade: str):
    """
    Busca os dados de uma universidade específica no arquivo CSV.

    Args:
        universidade (str): Nome da universidade em minúsculo.

    Returns:
        dict: Dicionário com os dados da universidade encontrada ou vazio se não encontrar.
    """
    dados = pd.read_csv("documents/universidades.csv")  # Lê o arquivo CSV com os dados das universidades
    dados["NOME_FACULDADE"] = dados["NOME_FACULDADE"].str.lower()  # Converte os nomes para minúsculas
    dados_com_essa_universidade = dados[dados['NOME_FACULDADE'] == universidade]  # Filtra pelo nome da universidade
    if dados_com_essa_universidade.empty:
        return {}  # Retorna dicionário vazio se não encontrar
    return dados_com_essa_universidade.iloc[:1].to_dict()  # Retorna os dados do primeiro registro encontrado

# Função para buscar dados de todas as universidades no arquivo CSV
def busca_dados_das_universidades():
    """
    Busca os dados de todas as universidades no arquivo CSV.

    Returns:
        dict: Dicionário com os dados de todas as universidades.
    """
    dados = pd.read_csv("documents/universidades.csv")  # Lê o arquivo CSV com os dados das universidades
    return dados.to_dict()

# Modelo Pydantic para extrair o nome da universidade
class ExtratorDeUniversidade(BaseModel):
    """
    Modelo para extração do nome da universidade.
    """
    universidade: str = Field("O nome da universidade em minúsculo.")

# Ferramenta para extrair dados de uma universidade específica
class DadosDeUniversidade(BaseTool):
    name: str = "DadosDeUniversidade"
    description: str = """Essa ferramenta extrai os dados de uma universidade.
    Passe para essa ferramenta como argumento o nome da universidade."""

    def _run(self, input:str) -> str:
        """
        Executa a extração dos dados de uma universidade a partir do input fornecido.

        Args:
            input (str): Texto contendo o nome da universidade.

        Returns:
            str: Dados da universidade em formato JSON.
        """
        # Inicializa o modelo de linguagem da Google com a chave de API
        LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
        
        # Define o parser para extrair o nome da universidade no formato JSON
        parser = JsonOutputParser(pydantic_object=ExtratorDeUniversidade)
        
        # Cria o prompt para o modelo de linguagem, instruindo como extrair o nome
        template = PromptTemplate(
            template="""Você deve analisar a entrada a seguir e extrair o nome de universidade informado em minúsculo.
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
        
        # Extrai o nome da universidade do resultado e converte para minúsculas
        universidade = result['universidade']
        universidade = universidade.lower().strip()
        
        # Busca os dados da universidade no CSV
        dados = busca_dados_da_universidade(universidade)
        
        # Retorna os dados encontrados em formato JSON
        return json.dumps(dados)
    
# Ferramenta para carregar dados de todas as universidades
class TodasUniversidades(BaseTool):
    name: str = "TodasUniversidades"
    description: str = """Carrega os dados de todas as universidades. Não é necessário nenhum parâmetro de entrada."""
    
    def _run(self, input: str):
        """
        Executa a busca e retorna os dados de todas as universidades.

        Args:
            input (str): Não utilizado.

        Returns:
            dict: Dados de todas as universidades.
        """
        # Busca os dados de todas as universidades
        dados = busca_dados_das_universidades()

        # Retorna os dados encontrados em formato JSON
        return dados