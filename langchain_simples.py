from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

# Carregando o ambiente de desenvolvimento 
# o qual contém a chave na raiz do projeto. 
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.8,
    api_key=os.getenv("GOOGLE_API_KEY"))

# Variáveis que serão interpoladas no modelo 
# de prompt. 

estilo_poema = "existencialista"
estilo_autor = "Fernando Pessoa"
qt_estrofes = 4
tema = """que se percebe numa sobre posição temporal ao perceber que ecos do passado, manifestado
            na presença de um amor ausenta ainda ecoa sobre sua mente. """

# Modelo do prompt que serve de template à criação de prompts. 

modelo_do_prompt = PromptTemplate.from_template(
    """Faça um poema {estilo_poema} de {qt_estrofes} estrofes de um eu lírico {tema}. 
    Para a construção do poema, inspire-se no estilo de escrita de {estilo_autor}.
    """
)

# Prompt :
prompt = modelo_do_prompt.format(estilo_poema = estilo_poema, qt_estrofes = qt_estrofes, 
                        tema = tema, estilo_autor = estilo_autor)

resposta = llm.invoke(prompt)
print(resposta.content)