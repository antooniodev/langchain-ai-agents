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
pergunta = "Busque os dados da Ana pelo seu nome e depois crie seu perfil acadêmico."
pergunta = "Compare o perfil acadêmico da Ana com o da Bianca!"
pergunta = "Tenho sentido Ana desanimada com cursos de matemática. Seria uma boa parear ela com o Marcos? "
pergunta = "Quais os dados da USP."
pergunta = "Quais os dados da UniCamp."
pergunta = "Quais os dados da Uni Camp."
pergunta = "Dentre USP e UFRJ, qual você recomenda para a acadêmica Ana?"
pergunta = "Dentre a uni camp e UFRJ, qual você recomenda para a Ana?"
pergunta = "Quais as faculdades com as melhores chances para a Ana entrar?"
pergunta = "Dentre todas as universidades existentes, quais Ana possui chances de entrar?"
pergunta = "Além das faculdades favoritas da Ana, existem outras faculdades. Considere elas também. Quais Ana tem mais chances de entrar?"

# Inicializa o agente personalizado
agent = AgentGeminiFunctions()

# Cria o executor do agente, passando o agente e suas ferramentas
executor = AgentExecutor(agent=agent.agent, tools=agent.tools, verbose=True)

# Executa o agente com a pergunta e obtém o resultado
result = executor.invoke({"input": pergunta})

# Exibe o resultado no console
print(result)
