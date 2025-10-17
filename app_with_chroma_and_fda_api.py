import os
import pandas as pd
import requests
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.docstore.document import Document

# ==============================
# Load environment variables
# ==============================
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

CHROMA_PERSIST_DIR = "./chroma_db"
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

CSV_FILE = "drug_ingredient.csv"
CHROMA_COLLECTION_NAME = "drug_ingredients_collection"

FDA_LABEL_API = "https://api.fda.gov/drug/label.json?search=openfda.brand_name:{}&limit=1"

# ==============================
# Load CSV and create documents
# ==============================
def load_csv_as_docs(csv_file: str):
    df = pd.read_csv(csv_file)
    docs = []
    for _, row in df.iterrows():
        content = (
            f"Active Ingredients: {row['Active Ingredients']}\n"
            f"Inactive Ingredients: {row['Inactive Ingredients']}"
        )
        docs.append(Document(page_content=content, metadata={"drug_name": row["Brand Name"]}))
    return docs, df

# ==============================
# Create or load Chroma vectorstore
# ==============================
def get_or_create_vectorstore(docs):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=OPENAI_API_VERSION
    )

    if os.path.exists(os.path.join(CHROMA_PERSIST_DIR, "index")):
        print("Loading existing Chroma DB...")
        vectordb = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
    else:
        print("Creating new Chroma DB and embedding CSV data...")
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME
        )
    return vectordb

# ==============================
# Fetch drug info from FDA API
# ==============================
def fetch_drug_from_fda(drug_name: str):
    try:
        response = requests.get(FDA_LABEL_API.format(drug_name))
        response.raise_for_status()
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            active = ", ".join(result.get("active_ingredient", [])) if "active_ingredient" in result else "N/A"
            inactive = ", ".join(result.get("inactive_ingredient", [])) if "inactive_ingredient" in result else "N/A"
            return f"Active Ingredients: {active}\nInactive Ingredients: {inactive}"
        else:
            return None
    except Exception as e:
        print(f"FDA API error: {e}")
        return None

# ==============================
# Build RAG chain
# ==============================
def build_rag_chain(vectordb):
    llm = AzureChatOpenAI(
        deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=OPENAI_API_VERSION,
        temperature=1.0
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 1})

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        k=5,
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return chain

# ==============================
# Main execution
# ==============================
if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        print(f"CSV file '{CSV_FILE}' not found!")
        exit()

    docs, df = load_csv_as_docs(CSV_FILE)
    vectordb = get_or_create_vectorstore(docs)
    print(f"Chroma DB '{CHROMA_COLLECTION_NAME}' is ready.")

    # Create a set of local drug names for exact matching
    local_drug_names = set([name.lower().strip() for name in df["Brand Name"]])

    chain = build_rag_chain(vectordb)

    while True:
        query = input("\nEnter drug name (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        query_lower = query.lower().strip()

        if query_lower in local_drug_names:
            # Drug exists locally → use RAG chain
            results = chain.invoke({"question": query, "chat_history": []})
            src_docs = results.get("source_documents", [])
            for doc in src_docs:
                print(doc.page_content)
        else:
            # Drug not in local → query FDA API
            fda_result = fetch_drug_from_fda(query)
            if fda_result:
                print(fda_result)
            else:
                print("Drug not found in local DB or FDA API.")
