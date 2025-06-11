# Carrega variáveis de ambiente do arquivo .env
from dotenv import load_dotenv

# Importa classes e funções necessárias do LangChain
from langchain.agents import AgentExecutor
from langchain import hub
from agent import AgentGeminiFunctions

# Carrega as variáveis de ambiente
load_dotenv()

# Define a pergunta a ser respondida pelo agente
pergunta = "Quais os dados da Bianca? Busque pelo nome."
pergunta = "Quais os dados da Ana? Busque pelo nome."
pergunta = "Quais os dados da Ana e da Bianca? Busque pelo nome."
pergunta 
# Inicializa o agente personalizado
agent = AgentGeminiFunctions()

# Cria o executor do agente, passando o agente e suas ferramentas
executor = AgentExecutor(agent=agent.agent, tools=agent.tools, verbose=True)

# Executa o agente com a pergunta e obtém o resultado
result = executor.invoke({"input": pergunta})

# Exibe o resultado no console
print(result)
