import pandas as pd
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
from langchain import hub
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
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
from datetime import datetime
from dotenv import load_dotenv
import time

current_datetime = datetime.utcnow()
today = datetime.now()

# Convert to string format
today_str = today.strftime("%Y-%m-%d")
import re

def string_to_dict_list(s):
    # Usar expresiones regulares para encontrar los segmentos de texto con números entre corchetes
    pattern = r'(.*?)\s*[\[\(](\d)\]|\)[\)]'
    matches = re.findall(pattern, s)

    # Convertir los matches en una lista de diccionarios
    dict_list = [{'frase': match[0].strip(), 'numero': int(match[1])} for match in matches]

    return dict_list
# Load environment variables from .env file
load_dotenv()

# Get the value of OPENAI_API_KEY from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is present
if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

#Define chroma client host and port
chroma_client = chromadb.HttpClient(host=os.getenv("CHROMADB_HOST"), port=os.getenv("CHROMADB_PORT"))
vectorstore = Chroma(
    client=chroma_client,
    collection_name=os.getenv("CHROMADB_COLLECTION"),
    embedding_function= OpenAIEmbeddings()
)


#Collection and embedding function
collection = chroma_client.get_or_create_collection(os.getenv("CHROMADB_COLLECTION"),embedding_function = OpenAIEmbeddings().embed_documents)



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
        if not actual.empty:
            actual["fuente"]=base_a_llave[base]
            total_df.append(actual)
    print('total_df', total_df)
    if total_df:

        update_df=pd.concat(total_df)
        if not("summary" in update_df.columns):
            update_df["summary"]=None
        update_df.sort_values(by="update_at",inplace=True)
        update_df["link"]=update_df.apply(lambda x: f"""/dashboard/{x["fuente"]}/{x["id"]}""",axis=1)
        update_df["summary"] = update_df.apply(lambda x: str(x["summary"]).encode('utf-8', 'ignore').decode('utf-8') if x["summary"] else "",axis=1)
        update_df["id"] = update_df.apply(lambda x: x["id"], axis=1)
        update_df=update_df[['update_at',"summary",'content','link','fuente',"title", "id","ciudad","componente","sector","image","mes","dia","año"]].copy()
        update_df.fillna(" ",inplace=True)
        return(update_df)

#Permite hacer una busqueda espesifica de una base de dapper
def busqueda_de_base(base,tiempo=None):
    print(base)
    extras=["ciudad","componente","sector"]
    root="_".join(base.split("_")[1:])
    last=base.split("_")[-1]
    query=f"""
    select {base}.id as id, {base}.created_at as creacion, {base}.update_at, {base}.title, {base}.summary, content,ARRAY_AGG(locations_city.name) as ciudad, ARRAY_AGG(component_components.name) as componente, ARRAY_AGG(sector_sector.name) as sector, {base}.image from 
                            {base} 
                        left join
                        dapper_{root}_component
                        on {base}.id = dapper_{root}_component.{last}_id
                        left join
                        component_components
                        on component_components.id=dapper_{root}_component.components_id
                        left join
                        dapper_{root}_cities
                        on {base}.id = dapper_{root}_cities.{last}_id
                        left join 
                        locations_city
                        on locations_city.id=dapper_{root}_cities.city_id
                        left join
                        dapper_{root}_sectors
                        on {base}.id = dapper_{root}_sectors.{last}_id
                        left join
                        sector_sector
                        on sector_sector.id=dapper_{root}_sectors.sector_id
                        --where_clause
                        group by {base}.id, {base}.created_at, {base}.update_at, {base}.title, {base}.summary, content;"""
    if root=="abc_abc":
        query=f""" select {base}.id as id, {base}.created_at as creacion, {base}.update_at, {base}.title, body as content, ARRAY_AGG(component_components.name) as componente, ARRAY_AGG(sector_sector.name) as sector from 
                        {base} 
                        left join
                        dapper_{root}_component
                        on {base}.id = dapper_{root}_component.{last}_id
                        left join
                        component_components
                        on component_components.id=dapper_{root}_component.components_id
                        left join
                        dapper_{root}_sectors
                        on {base}.id = dapper_{root}_sectors.{last}_id
                        left join
                        sector_sector
                        on sector_sector.id=dapper_{root}_sectors.sector_id
                        --where_clause
                        group by {base}.id, {base}.created_at, {base}.update_at, {base}.title,content;"""
    elif base=="dapper_listening_content":
        query=f"""select {base}.id as id, {base}.created_at as creacion, {base}.summary,{base}.update_at, {base}.title, {base}.body as content,ARRAY_AGG(locations_city.name) as ciudad, ARRAY_AGG(component_components.name) as componente, thumbnail as image from 
                                {base} 
                        left join
                        dapper_{root}_component
                        on {base}.id = dapper_{root}_component.{last}_id
                        left join
                        component_components
                        on component_components.id=dapper_{root}_component.components_id
                        left join
                        dapper_{root}_cities
                        on {base}.id = dapper_{root}_cities.{last}_id
                        left join 
                        locations_city
                        on locations_city.id=dapper_{root}_cities.city_id
                        --where_clause
                        group by {base}.id, {base}.summary, {base}.created_at, {base}.update_at, {base}.title,content;"""
    elif last=="news":
        query=f"""select {base}.id as id, {base}.created_at as creacion, {base}.update_at, {base}.title, {base}.content,ARRAY_AGG(locations_city.name) as ciudad, ARRAY_AGG(component_components.name) as componente, ARRAY_AGG(sector_sector.name) as sector, {base}.image from 
                        {base} 
                        left join
                        dapper_{root}_component
                        on {base}.id = dapper_{root}_component.{last}_id
                        left join
                        component_components
                        on component_components.id=dapper_{root}_component.components_id
                        left join
                        dapper_{root}_cities
                        on {base}.id = dapper_{root}_cities.{last}_id
                        left join 
                        locations_city
                        on locations_city.id=dapper_{root}_cities.city_id
                        left join
                        dapper_{root}_sectors
                        on {base}.id = dapper_{root}_sectors.{last}_id
                        left join
                        sector_sector
                        on sector_sector.id=dapper_{root}_sectors.sector_id
                    --where_clause
                    group by {base}.id, {base}.created_at, {base}.update_at, {base}.title, content;"""

    if tiempo==None:
        sql_query = query
    else:
        tiempo=datetime.fromtimestamp(tiempo)
        sql_query = query.replace("--where_clause",f"where {base}.update_at>'{tiempo}'")
    connection = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"), # default port for PostgreSQL is 5432
    )

    connection.set_client_encoding('UTF8')
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
    for columna in extras:
        try:
            result_df[columna]=result_df[columna].apply(lambda x: list(set([str(elemento) for elemento in x if elemento])))
        except:
            result_df["ciudad"]=[[] for _ in range(len(result_df))]
    result_df["ciudad"]=result_df["ciudad"].apply(lambda x:[]  if len(x)==6 else x)
    for columna in extras:
        try:
            result_df[columna]=result_df[columna].apply(lambda x: "-".join(x))
        except:
            pass
    result_df.rename(columns={"body":"content"},inplace=True)
    result_df["content"]=result_df["content"].apply(lambda x: BeautifulSoup(x.encode('utf-8', 'ignore').decode('utf-8'), 'html.parser').text.replace("\n"," ").replace("\xa0"," "))
    result_df["update_at"] = pd.to_datetime(result_df["update_at"])
# Crear las columnas de mes, día y año
    result_df["mes"] = result_df["update_at"].dt.month
    result_df["dia"] = result_df["update_at"].dt.day
    result_df["año"] = result_df["update_at"].dt.year
    result_df["update_at"]=result_df["update_at"].apply(pd.Timestamp).astype('int64') // 10**9
    result_df["creacion"]=pd.to_datetime(result_df["creacion"]).apply(pd.Timestamp).astype('int64') // 10**9
    return(result_df)


hoy=unix_timestamp = int(time.time())
#se definne el retriver apartir del vectorstore
metadata_field_info = [
    AttributeInfo(
        name="Componente",
        description="el tipo de contenido de la siguiente lista [Ciudades, PolifonÍA, Económico, Regulación, Ambiental-Sostenibilidad], unicamente estas opciones, cada una representa un tipo de contenido Ciudades contenido local de ciudades Economico contenido economico, PolifonÍA contenido de redes sociales y Ambiental-Sostenibilidad contenido ambiental se puede tener multiples componentes",
        type="string o lista de strings separados por -",
    ),
    AttributeInfo(
        name="Sector",
        description="el sector economico del que trata el texto",
        type="string",
    ),
    AttributeInfo(
        name="mes",
        description=f"mes de publicacion del contenido",
        type="Int",
    ),
    AttributeInfo(
        name="dia",
        description=f"dia publicacion del contenido",
        type="Int",
    ),
    AttributeInfo(
        name="año",
        description=f"mes de publicacion del contenido",
        type="Int",
    )
]

document_content_description = f"Contenido de la pagina web sobre regulacion economia y varios, la fecha de hoy es {today_str} "

#retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    fix_invalid=True,
    use_original_query=True
)

retriver2=vectorstore.as_retriever()

#Se define el prompt


prompt.messages[0].prompt.template="""Eres un asistente para tareas de pregunta-respuesta que usa el contenido de la plataforma DAPPER.
 Utiliza los siguientes fragmentos de contexto recuperado para responder la pregunta. Si no sabes la respuesta, simplemente di que no lo sabes.
   responde en maximo un parrafo de 70 palabras, al final de cada frase(punto) reponde entre corchetes el id de la fuente que usaste para responder esa frase ej dapper es lo mejor[1] recuerda responder el id entre corchetes cuadrados y no circulares,
     en caso de considerar la informacion insuficeinte o de no recibir informacion responder que no hay infomracion sin citar, porfavor solo responder con la informacpin proveida en contexto, si el contexto esta vacio reponder que no hay informacion
     \nPregunta: {question} \nContexto: {context} \nRespuesta:"""





def format_docs(docs):
    return "\n\n".join(f"""ID: {i} Contenido: {doc.page_content} Fuente:{doc.metadata["title"]} Seccion:{doc.metadata["fuente"]}""" for i, doc in enumerate(docs))


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
rag_chain_with_source2 = RunnableParallel(
    {"context": retriver2, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

def dict_in_list(dict_list, dict_item):
    for d in dict_list:
        if d == dict_item:
            return True
    return False

def respondedor(pregunta):
    try:
        respuesta=rag_chain_with_source.invoke(pregunta)
    except:
        print("no funciono el filtro se manda sin filtro")
        respuesta=rag_chain_with_source2.invoke(pregunta)

        
    contexto=respuesta["context"].copy()
    parseado=string_to_dict_list(respuesta["answer"])
    contexto_new=[]
    if len(contexto)<0:
        pass
    else: 
        for rep in parseado:
            try:
                a_agregar=contexto[rep["numero"]]
                if  dict_in_list(contexto_new,a_agregar):
                    pass
                else:
                    contexto_new.append(a_agregar)
            except:
                pass    
        contexto_new.extend([context for context in contexto if  not (dict_in_list(contexto_new,context))])
        respuesta["context"]=contexto_new

    respuesta["answer_parsed"]= [parsed["frase"] for parsed in parseado]
    
    return(respuesta)





def actualizar(all=False):
    current_df=pd.DataFrame(vectorstore._collection.get()["metadatas"])
    if all or len(current_df)==0:
        final=None
    else:
        final=current_df["update_at"].max()
    print(final)
    update=get_base_total(bases_relevantes,tiempo=final)
    if not update.empty:
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
        print('metadatas', metadatas[:2])
        print('contents', contents[:2])
        ids=add_position_to_duplicates(ids)
        collection.add(ids=ids,documents=contents,metadatas=metadatas)