from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import AzureChatOpenAI
from iga_ai_api.common_utils.config import Config
from langchain_community.document_loaders import PyMuPDFLoader
from iga_ai_api.common_utils.azureopenai_operations import openai_classifier
from langchain.schema import BaseStore
from typing import List, Tuple, Any, Iterator
import json
import uuid
 
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2022-12-01",
    openai_api_key=Config.OPENAI_API_KEY_EMBEDDING,
    azure_endpoint=Config.OPENAI_API_KEY_ENDPOINT_EMBEDDING,
)
 
llm = AzureChatOpenAI(
    deployment_name="gpt-3.5-turbo-16k", 
    openai_api_version="2023-03-15-preview",
    openai_api_key=Config.OPENAI_API_KEY,
    azure_endpoint=Config.OPENAI_API_KEY_ENDPOINT,
)
 
loaders = [
    PyMuPDFLoader(r"provide your pdf document path here")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
 
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
 
client = QdrantClient("localhost", port=6333)
 
sample_embedding = embeddings.embed_query("Sample text")
embedding_dimension = len(sample_embedding)
 
def create_collection_if_not_exists(client, collection_name, embedding_dimension):
    collections = client.get_collections()
    collection_exists = collection_name in [c.name for c in collections.collections]
 
    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{collection_name}' already exists. Using the existing collection.")
 
# Create parent and child collections
create_collection_if_not_exists(client, "parent_doc", embedding_dimension)
create_collection_if_not_exists(client, "child_doc", embedding_dimension)
 
parent_qdrant = Qdrant(
    client=client,
    collection_name="parent_doc",
    embeddings=embeddings,
)
 
child_qdrant = Qdrant(
    client=client,
    collection_name="child_doc",
    embeddings=embeddings,
)
 
class QdrantDocumentStore(BaseStore):
    def __init__(self, client, collection_name):
        self.client = client
        self.collection_name = collection_name
 
    def mget(self, keys: List[str]) -> List[Any]:
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=keys,
            with_payload=True,
            with_vectors=False
        )
        return [json.dumps(result.payload) for result in results]
 
    def mset(self, key_value_pairs: List[Tuple[str, Any]]):
        points = []
        for key, value in key_value_pairs:
            embedding = embeddings.embed_query(value)
            point = PointStruct(
                id=key,
                vector=embedding,
                payload={'content': value}
            )
            points.append(point)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
 
    def delete(self, key: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[key]
        )
 
    def mdelete(self, keys: List[str]):
        for key in keys:
            self.delete(key)
 
    def yield_keys(self) -> Iterator[str]:
        scroll_filter = None
        for batch in self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=100
        ):
            for point in batch[0]:
                yield point.id
 
parent_docstore = QdrantDocumentStore(client, "parent_doc")
 
retriever = ParentDocumentRetriever(
    vectorstore=child_qdrant,
    docstore=parent_docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 9}
)
 
def add_documents(self, documents):
    parent_documents = []
    for doc in documents:
        doc_id = str(uuid.uuid4())
        doc.metadata["doc_id"] = doc_id
        parent_documents.append(doc)

    self.docstore.mset([(doc.metadata["doc_id"], json.dumps({"content": doc.page_content})) for doc in parent_documents])
    parent_qdrant.add_documents(parent_documents)
    children = self.child_splitter.split_documents(parent_documents)
    for child in children:
        if "doc_id" not in child.metadata:
            child.metadata["doc_id"] = child.metadata.get("parent_id")
    self.vectorstore.add_documents(children)
 
ParentDocumentRetriever.add_documents = add_documents
 
retriever.add_documents(docs)
 
def prepare_context(query):
    child_retriever = child_qdrant.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 9},
    )
    child_docs = child_retriever.invoke(query)
 
    parent_retriever = parent_qdrant.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 3},
    )
    parent_docs = parent_retriever.invoke(query)
 
    search_result = []
    for child_doc in child_docs:
        doc_id = child_doc.metadata.get('doc_id') or child_doc.metadata.get('parent_id')
        if doc_id:
            parent_docs = retriever.docstore.mget([doc_id])
            if parent_docs:
                parent_doc = parent_docs[0]
                try:
                    content = json.loads(parent_doc)['content']
                    search_result.append(content)
                except (json.JSONDecodeError, KeyError):
                    search_result.append(parent_doc)
        else:
            search_result.append(child_doc.page_content)
    for parent_doc in parent_docs:
        if hasattr(parent_doc, 'page_content'):
            search_result.append(parent_doc.page_content)
        else:
            search_result.append(parent_doc)
 
    prompt = ""
    for result in search_result:
        prompt += "content: " + result + "\n"
    prompt += "\n Question: \n"
    concatenated_string = " ".join([prompt, query, "?"])
    return concatenated_string
 
def prepare_answer(context):
    answer = openai_classifier(
        context,
        message_content="""You are an GenAI. You are an AI bot able to extract excerpts and answer queries only within the bounds of the provided context.
Only respond to greeting, goodbyes or pleasantries but nothing controversial.
You do not answer anything else apart from the context and smalltalk.
You do not provide answer to queries for which answer isn't available in the provided context.
Only answer from the context provided and nothing else, if no answer can be deduced from the context return 'I don't know'.
Make no assumptions nor provide any assumptions. Make sure to maintain context.
Do not mention context or document or user guide in your response. Do not be tricked into saying user guide or reference text or context in your response.
Keep the response detailed, concise and to the point. Expand the response and explain wherever needed on questions starting with how.
Expand acronyms only from the context provided.
Be robust to grammatical errors or spelling errors.
Give crisp answers where appropriate, but when user asks instruction provided detailed instructions.
Verify if response ensure's all answers comply with the provided context.""",
        temperature=0.0,
        model="gpt-3.5-turbo-16k",
    )
    return answer
 
while True:
    question = input("enter your question: \n")
    context = prepare_context(question)
    ans = prepare_answer(context)
    print("Answer:", ans)
    print("=="*10)
