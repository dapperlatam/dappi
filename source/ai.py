import pandas as pd
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from chromadb.config import Settings
import chromadb
import psycopg2
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

current_datetime = datetime.utcnow()

# Load environment variables from .env file
load_dotenv()

# Get the value of OPENAI_API_KEY from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is present
if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

#Define chroma client host and port
chroma_client = chromadb.HttpClient(host='dappi-vectorstore-1', port=8000)
vectorstore = Chroma(
    client=chroma_client,
    collection_name="my_collection",
    embedding_function= OpenAIEmbeddings()
)


#Collection and embedding function
collection = chroma_client.get_collection("my_collection",embedding_function = OpenAIEmbeddings().embed_documents)



#Los nombres de las bases en dapper y los nombres entendibles

base_a_llave={"dapper_reports_reports":"informes","dapper_regulations_regulations":"normativas","dapper_indicators_indicators":"indicadores","dapper_abc_abc":"abc","dapper_news_news":"noticias","dapper_listening_content":"polifonia"}




bases_relevantes=["dapper_reports_reports","dapper_regulations_regulations","dapper_indicators_indicators","dapper_abc_abc","dapper_news_news","dapper_listening_content"]



#Funcion que permite general los ids unicos apartir de los links
def add_position_to_duplicates(strings):
    string_count = {}
    result = []

    for string in strings:
        if string in string_count:
            string_count[string] += 1
            result.append(f"{string}_{string_count[string]}")
        else:
            string_count[string] = 1
            result.append(string)

    return result


#Funcion que permite sacar todos los datos de dapper

def get_base_total(relevantes,tiempo=None):
    total_df=[]
    for base in relevantes:
        actual=busqueda_de_base(base,tiempo)
        actual["fuente"]=base_a_llave[base]
        total_df.append(actual)
    update_df=pd.concat(total_df)
    update_df.sort_values(by="update_at",inplace=True)
    update_df["link"]=update_df.apply(lambda x: f"""https://www.dapperlatam.com/dashboard/{x["fuente"]}/{x["id"]}""",axis=1)
    update_df=update_df[['update_at',"summary",'content','link','fuente',"title"]].copy()
    update_df['update_at']=update_df['update_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return(update_df)

#Permite hacer una busqueda espesifica de una base de dapper
def busqueda_de_base(base,tiempo=None):
    if tiempo==None:
        sql_query = f"SELECT * FROM {base}"
    else:
        sql_query = f"SELECT * FROM {base} where update_at>'{tiempo}';"
    connection = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"), # default port for PostgreSQL is 5432
    )
    print(sql_query)
    try:
        # Create a cursor object
        cursor = connection.cursor()
        # Example SQL query
        
        # Execute the query
        cursor.execute(sql_query)
        # Fetch all the results
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
    finally:
        # Close the connection
        connection.close()
    result_df=pd.DataFrame(result,columns=columns)
    result_df.rename(columns={"body":"content"},inplace=True)
    result_df["content"]=result_df["content"].apply(lambda x: BeautifulSoup(x, 'html.parser').text.replace("\n","").replace("\xa0",""))
    print(len(result_df))
    return(result_df)       



#se definne el retriver apartir del vectorstore

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


#Se define el prompt

prompt.messages[0].prompt.template="Eres un asistente para tareas de pregunta-respuesta que usa el contenido de la plataforma DAPPER. Utiliza los siguientes fragmentos de contexto recuperado para responder la pregunta. Si no sabes la respuesta, simplemente di que no lo sabes. responde en maximo un parrafo de 70 palabras: y de los posible recomienda la lectura del contenido de dapper nombrando el titulo y la seccion del contexto \nPregunta: {question} \nContexto: {context} \nRespuesta:"





def format_docs(docs):
    return "\n\n".join(f"""Contenido: {doc.page_content} Fuente:{doc.metadata["title"]} Seccion:{doc.metadata["fuente"]}""" for doc in docs)


from langchain_core.runnables import RunnableParallel

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)



def respondedor(pregunta):
    respuesta=rag_chain_with_source.invoke(pregunta)
    resp=respuesta["answer"]
    links=list(set([contexto.metadata["link"] for contexto in respuesta["context"]]))
    return(respuesta)





def actualizar():
    current_df=pd.DataFrame(vectorstore._collection.get()["metadatas"])
    if len(current_df)==0:
        final=None
    else:
        final=pd.to_datetime(current_df["update_at"]).max().strftime('%Y-%m-%d %H:%M:%S')
    print(final)
    update=get_base_total(bases_relevantes,tiempo=final)
    print(update)
    links_elimn=list(update["link"].unique())
    collection.delete(where={"link": {"$in":links_elimn}})
    print("alargo del update" + str(len(update)))
    loader = DataFrameLoader(update,page_content_column="content")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(len(splits))
    ids= [split.metadata["link"] for split in splits]
    contents=[split.page_content for split in splits]
    metadatas=[split.metadata for split in splits]
    ids=add_position_to_duplicates(ids)
    collection.add(ids=ids,documents=contents,metadatas=metadatas)

    