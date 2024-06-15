from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.globals import set_debug

from dotenv import load_dotenv
import os

# Carregando o ambiente de desenvolvimento 
# o qual contém a chave na raiz do projeto. 
load_dotenv()

# Método de debug que me permite identificar
# como que o input está sendo passado por cada
# cadeia LLMChain. 
set_debug(True)

# Criando a LLM :

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.9,
    api_key=os.getenv("GOOGLE_API_KEY"))


# gêneros - dark fantasy
# estilos - gótica, sombria e com terror psicológico

# Criando o modelo (template) sobre o tema da história.

modelo_genero = ChatPromptTemplate.from_template(
    """Crie um esboço de uma letra de música do gênero {tema} que
    traga como seu estilo uma camada gótica, sombria e com terror psicológico. """
)

# temas - rejeição, do sofrimento, da angústia, do abuso psicológico e da busca de amor e acolhimento
# mensagem - a vida é extremamente fria, gelada, injusta e dolorida, tornando-se mais colorida e quente apenas quando é amado e acolhido.

# Criando um template para os temas que precisam ser explorados.

modelo_temas = ChatPromptTemplate.from_template(
    """A letra deve explorar os {temas} """
)

# local - São Paulo
# contexto - ambiente cyberpunk, com transhumanismo
# lore - Presença de ghouls, seres humanos que possuem habilidades tidas como paranormais e que são um produto de hibridismo de um povo antigo com os primeiros seres humanos.

# Criei um template para o local que deve servir como cenário da obra. 

# modelo_cenarios = ChatPromptTemplate.from_template(
 #   """O cenário deve ser {local}"""
#)

# Criando variáveis que armazenam a execução da classe LLMChain, 
# que executa o prompt informado para a LLM instanciada. 

cadeia_genero = LLMChain(prompt = modelo_genero, llm = llm) 
cadeia_temas = LLMChain(prompt = modelo_temas, llm = llm)

# cadeia_cenario = LLMChain(prompt = modelo_cenarios, llm = llm)

cadeia = SimpleSequentialChain(chains=[cadeia_genero, cadeia_temas],
                            verbose=True)

resposta = cadeia.invoke(input="post-hardcore")
print(resposta.content)