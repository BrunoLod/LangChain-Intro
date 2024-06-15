# Importando a biblioteca que permite requisitar, via API, o gemini. 

from langchain_google_genai import ChatGoogleGenerativeAI

# Importando a biblioteca que permite a criação de templates de conversa. 

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

# Importando as bibliotecas referentes ao sequenciamento em cadeias. 

from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain

# Importando o método que me permite analisar com maior profundidade, 
# debugando a saída gerada pelo SimpleSequentialChain ao ser invocada. 

from langchain.globals import set_debug

# Importando as bibliotecas que permitem a manipulação de saída
# do LLMChain, mediante a classe Destino criada. 

from langchain_core.pydantic_v1 import Field, BaseModel

from langchain_core.output_parsers import JsonOutputParser

# Importando as bibliotecas que permitem utilizar as variáveis de ambiente
# no meu ambiente de desenvolvimento para que eu guarde a API_KEY do google, 
# a qual me permite utilizar externamente o modelo gemini 1.5 pro. 

from dotenv import load_dotenv
import os

# Carregando o ambiente de desenvolvimento, o qual 
# contém a chave na raiz do projeto. 

load_dotenv()

# Método de debug que me permite identificar
# como que o input está sendo passado por cada cadeia LLMChain. 

set_debug(True)

# Criando a classe para tratamento das saídas da LLMChain :
# A classe criada está herdando elementos da classe BaseModel. 

class Destino(BaseModel):
    
    local = Field("local a visitar")
    motivo = Field("motivo pelo qual é interessante visitar")
    
    

# Instanciando a classe e num objeto :

# Nesse trecho estou dizendo o seguinte : 

# Crie um parseador que forneça uma saída Json - por isso a biblioteca informada - 
# Para que eu consiga a minha saída esperada, a qual elaborei na classe destino, informo
# ao JsonOutputParser no que ela está baseada, ou seja, num objeto pydantic, que 
# corresponde a classe Destino, a qual é modula a produzir como resposta o local
# o motivo pelo qual é interessante visitar. 

parseador = JsonOutputParser(pydantic_object=Destino)

# Criando a LLM :

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.9,
    api_key=os.getenv("GOOGLE_API_KEY"))

# Criando o modelo (template) sobre a dica de viagem.

modelo_viagem = PromptTemplate(
    template = """Forneça locais de viagens dado o meu interesse por {interesse}. 
    {formatacao_saida}""", 
    input_variables = ["interesse"],
    partial_variables = {"formatacao_saida": Destino}
)

# Criando um template para os possíveis locais atrelados ao interesse especificado.

modelo_roteiro = ChatPromptTemplate.from_template(
    """Informe quais lugares visitar nesses locais, para que eu possa experienciar {temas}."""
)

# Criando variáveis que armazenam a cadeia de processamento, 
# formada pela combinação do prompt com o modelo de llm utilizado. 

cadeia_genero = LLMChain(prompt = modelo_viagem, llm = llm) 
cadeia_temas = LLMChain(prompt = modelo_roteiro, llm = llm)

# Criando a cadeia de sequenciamento para cada elemento 
# a ela informada, assim como o verbose=True, que me permite
# ter uma saída mais detalhada para compreensão. 

cadeia = SimpleSequentialChain(chains=[cadeia_genero, cadeia_temas],
                            verbose=True)

# Invocando a cadeia, que armazena o SimpleSequentialChain, 
# informando como input o gênero musical com o qual espero 
# que ele crie um esboço de letra de música. 

resposta = cadeia.invoke(input="cyperbunk")
print(resposta.content) 