# Gen-AI Document Based Question-Answering System
# Overview

This project implements a document retrieval system using LangChain and Qdrant vector database, integrated with Azure OpenAI services. The system splits documents into parent and child chunks for efficient retrieval and provides a question-answering interface.

# Architecture 

<img width="757" alt="Screenshot 2024-10-31 at 9 34 38 PM" src="https://github.com/user-attachments/assets/b955777e-92d1-4153-a1fa-4bee60212734">


## Prerequisites

- Python 3.x
- Qdrant running locally on port 6333
- Azure OpenAI API access
- Required Python packages:
  - langchain
  - qdrant-client
  - PyMuPDF
  - azure-openai

## System Components

### 1. Embeddings and LLM Setup
- Uses Azure OpenAI's text-embedding-ada-002 for document embeddings
- Employs GPT-3.5-turbo-16k for question answering
- Configurations are managed through environment variables

### 2. Document Processing
- Loads PDF documents using PyMuPDFLoader
- Implements two-level document splitting:
  - Parent chunks: 4000 characters with 800 character overlap
  - Child chunks: 1000 characters with 200 character overlap

### 3. Vector Database Setup
- Uses Qdrant as the vector store
- Creates two collections:
  - parent_doc: stores larger document chunks
  - child_doc: stores smaller, more granular chunks
- Automatically creates collections if they don't exist

### 4. Custom Document Store
The `QdrantDocumentStore` class implements:
- Document storage and retrieval
- Vector similarity search
- Document metadata management
- CRUD operations for documents

### 5. Retrieval System
Features:
- Parent-child document relationship maintenance
- Similarity score threshold-based retrieval
- Configurable search parameters (k=9 for child documents, k=3 for parent documents)
- Score threshold of 0.5 for relevance filtering

### 6. Question-Answering Pipeline
The system provides:
- Context preparation from retrieved documents
- Answer generation using Azure OpenAI
- Response filtering and formatting
- Interactive question-answering interface

## Usage

1. Initialize the system:
```python
# Configure your Azure OpenAI credentials
embeddings = AzureOpenAIEmbeddings(...)
llm = AzureChatOpenAI(...)

# Load your documents
loaders = [PyMuPDFLoader("your_pdf_path.pdf")]
```

2. Create and populate the vector store:
```python
retriever = ParentDocumentRetriever(...)
retriever.add_documents(docs)
```

3. Use the interactive interface:
```python
question = input("enter your question: \n")
context = prepare_context(question)
answer = prepare_answer(context)
```

## Key Features

- Hierarchical document storage (parent-child relationship)
- Efficient vector similarity search
- Context-aware question answering
- Automatic document splitting and embedding
- Robust error handling and response validation
- Configurable search parameters and thresholds

## Response Guidelines

The system follows these principles:
- Answers only based on provided context
- Returns "I don't know" for queries without context
- Handles greetings and basic pleasantries
- Provides detailed responses for "how" questions
- Maintains context relevance
- Expands acronyms when found in context
- Robust to grammatical and spelling errors

## Technical Notes

- Vector dimension is automatically determined from the embedding model
- UUID-based document identification
- JSON-based metadata storage
- Configurable chunk sizes and overlap
- Threshold-based similarity search
- Automatic collection management

## Error Handling

The system includes:
- Collection existence checking
- JSON parsing error handling
- Document ID validation
- Context retrieval fallbacks
- Metadata validation

## Limitations

- Requires local Qdrant instance
- Azure OpenAI API dependency
- PDF-focused document loading (can be extended)
- Fixed chunk sizes (configurable but static)
