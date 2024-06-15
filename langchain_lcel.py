# Importando a biblioteca que permite requisitar, via API, o gemini. 

from langchain_google_genai import ChatGoogleGenerativeAI

# Importando a biblioteca que permite a criação de templates de conversa. 

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

# Importando o método que me permite analisar com maior profundidade, 
# debugando a saída gerada pelo SimpleSequentialChain ao ser invocada. 

from langchain.globals import set_debug

# Importando as bibliotecas que permitem a manipulação de saída
# do LLMChain, mediante a classe Destino criada. 

from langchain_core.pydantic_v1 import Field, BaseModel

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

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
    
    cidade = Field("cidade a visitar")
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
    temperature=0.75,
    api_key=os.getenv("GOOGLE_API_KEY"))

# Criando o modelo (template) sobre a dica de viagem.

modelo_cidade = PromptTemplate(
    template = """Sugira uma cidade para conhecer dado o meu interesse por {interesse}. 
    {formatacao_saida}""", 
    input_variables = ["interesse"],
    partial_variables = {"formatacao_saida": parseador.get_format_instructions()},
)

# Criando um template que seleciona os tipos de locais informados. 

modelo_espacos_culturais = ChatPromptTemplate.from_template(
    """Sugira os espaços culturais e artísticos encontrados no local {cidade} a visitar."""
)

# Criando um template para os locais selecionados.

modelo_lugares = ChatPromptTemplate.from_template(
    """Sugira livrarias, cafés e bares que eu posso encontrar em {cidade}."""
)



# Executando o mesmo processo em cadeia, mas com expression languange do LangChain. 

# Esse método consiste em tornar o processo de formar a cadeia mais fácil, 
# não seguindo a abordagem orientada a objetos. Para isso, informa o modelo 
# de prompt, a llm de uso e o parseador, que irá transformar a saída da llm 
# no formato esperado. 

# A sua principal vantagem é a de que permite que cada processo seja um 
# executável, permitindo desse o modo o parelelismo do processo, sendo 
# cada etapa síncrona. 

cadeia_1 = modelo_cidade | llm | parseador
cadeia_2 = modelo_espacos_culturais| llm | StrOutputParser()
cadeia_3 = modelo_lugares| llm | StrOutputParser()

# Criando um dicionário para as duas cadeias dessa forma, significa que estou 
# enviando a saída da cadeia_1 para a cadeia_2 e cadeia_3 ao mesmo tempo, 
# tornando o processo síncrono e paralelo. 

# Dessa forma, não mais a saída da cadeia_2 será a entrada da cadeia_3, 
# uma vez que ambas estão recebendo como tal a saída da cadeia_1. 

# É esse o processo que eu havia comentado sobre permitir que haja paralelismo. 

cadeia = (cadeia_1 | {"espaços_culturais_artísticos" : cadeia_2, "lugares": cadeia_3})


resposta = cadeia.invoke({"interesse" : "cultura gótica"})
print(resposta.content) 