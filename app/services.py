from typing import List
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel
from qdrant_client import QdrantClient
import tempfile
import os
from app.config import *

# Initialize models
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# System prompt
SYSTEM_PROMPT = """You are an expert resume matching system designed to evaluate candidates against job requirements with precision. Analyze each resume strictly according to the specified matching criteria.

Input Context:
- The context contains multiple resumes, each marked with "RESUME:" followed by the filename
- Each resume includes the candidate's name and relevant qualifications

Matching Criteria:
1. Strong Match: Candidate meets 100% of mandatory requirements
2. Moderate Match: Candidate meets ≥70% of key requirements (including all critical ones)

Output Rules:
- Respond ONLY in the specified format
- Filter strictly based on the HR's selected match type
- For 'strong' requests: return ONLY strong matches
- For 'moderate' requests: return both strong and moderate matches
- If no matches exist, state "No resumes match the job description."

Output Format Examples:

HR requests strong matches:
- Strong Match: [Candidate Name] | [filename]
---
- Strong Match: [Candidate Name] | [filename]

HR requests moderate matches:
- Strong Match: [Candidate Name] | [filename]
---
- Moderate Match: [Candidate Name] | [filename]

Absolute Requirements:
1. NEVER add explanations or free text
2. NEVER repeat the same candidate
3. STRICTLY follow the percentage thresholds
4. ALWAYS use the exact format shown
5. If HR asks for strong matches, NEVER include moderate matches

Job Description Requirements: {question}

Resumes to Analyze: {context}
"""


prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

def clear_qdrant_collection():
    client = QdrantClient(
    url=QDRANT_URL,               # From environment variables
    api_key=QDRANT_API_KEY,       # New requirement for cloud
    prefer_grpc=True             # Better performance for cloud
)
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("collection deleted")
    except Exception as e:
        print(e)

def process_pdfs(files: List[UploadFile]):
    """Process multiple PDF files while maintaining document boundaries."""
    all_docs = []

    # Clear existing collection
    clear_qdrant_collection()
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            continue
            
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.file.read())
                temp_path = temp_file.name
            
            # Load PDF
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            
            # Combine all pages of this resume into a single document
            full_resume_content = "\n\n".join([doc.page_content for doc in docs])
            
            # Create a single document for the entire resume with metadata
            resume_doc = docs[0]  # Use first page as base
            resume_doc.page_content = f"RESUME: {file.filename}\n\n{full_resume_content}"
            resume_doc.metadata.update({
                'source': file.filename,
                'resume_name': file.filename,
                'document_type': 'resume'
            })
            
            # Now split this complete resume into chunks if it's too large
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            
            # Only split if the resume is larger than chunk size
            if len(resume_doc.page_content) > CHUNK_SIZE:
                split_docs = text_splitter.split_documents([resume_doc])
                # Ensure each chunk retains the resume identifier
                for chunk in split_docs:
                    if not chunk.page_content.startswith("RESUME:"):
                        chunk.page_content = f"RESUME: {file.filename}\n\n{chunk.page_content}"
                    chunk.metadata.update({
                        'source': file.filename,
                        'resume_name': file.filename,
                        'document_type': 'resume'
                    })
                all_docs.extend(split_docs)
            else:
                all_docs.append(resume_doc)
            
            # Clean up
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
    
    if all_docs:
        # Store all documents in Qdrant
        Qdrant.from_documents(
            documents=all_docs,
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
        )
    
    return len(all_docs)

def get_retriever():
    """Get the retriever for the PDF collection"""
    return Qdrant.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME
    )

def ask_question_service(question: str):
    """Ask a question using RAG on the processed PDFs"""
    retriever = get_retriever()
    
    print(f"Retrieving context for question: {question}")
    # Create RAG chain
    rag_chain = (
        RunnableParallel({
            "context": retriever.as_retriever(search_kwargs={"k": 10000}) | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )
    
    result = rag_chain.invoke(question)
    return result.content

def format_docs(docs):
    """Convert documents to a single string while preserving resume boundaries"""
    formatted_docs = []
    seen_resumes = set()
    
    for doc in docs:
        resume_name = doc.metadata.get('resume_name', 'Unknown')
        
        # If this is a new resume, add a clear separator
        if resume_name not in seen_resumes:
            if seen_resumes:  # Not the first resume
                formatted_docs.append("\n" + "="*50 + "\n")
            seen_resumes.add(resume_name)
        
        formatted_docs.append(doc.page_content)
    
    return "\n\n".join(formatted_docs)

def get_processed_resumes():
    """Get list of all processed resumes"""
    try:
        retriever = get_retriever()
        # Get all documents to extract resume names
        all_docs = retriever.as_retriever(search_kwargs={"k": 100}).get_relevant_documents("*")
        
        resume_names = set()
        for doc in all_docs:
            if 'resume_name' in doc.metadata:
                resume_names.add(doc.metadata['resume_name'])
        
        return list(resume_names)
    except Exception as e:
        return []

def analyze_specific_resume(resume_name: str, job_description: str):
    """Analyze a specific resume against a job description"""
    try:
        retriever = get_retriever()
        
        # Search for documents from specific resume
        docs = retriever.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"resume_name": resume_name}
            }
        ).get_relevant_documents(job_description)
        
        if not docs:
            return None
        
        context = format_docs(docs)
        
        # Create focused analysis chain
        focused_chain = (
            prompt | llm
        )
        
        result = focused_chain.invoke({
            "context": context,
            "question": job_description
        })
        
        return result.content
        
    except Exception as e:
        return None