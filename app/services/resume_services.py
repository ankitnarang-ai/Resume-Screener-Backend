import asyncio
import concurrent.futures
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
import io
import PyPDF2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import re
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

# Updated System prompt for JSON response
SYSTEM_PROMPT = """You are an expert resume matching system designed to evaluate candidates against job requirements with precision. Analyze each resume strictly according to the specified matching criteria.

Input Context:
- The context contains multiple resumes, each marked with "RESUME:" followed by the filename
- Each resume includes the candidate's name and relevant qualifications

Matching Criteria:
1. Strong Match: Candidate meets 100% of mandatory requirements
2. Moderate Match: Candidate meets ≥70% of key requirements (including all critical ones)

Output Rules:
- Respond ONLY in valid JSON format
- Filter strictly based on the HR's selected match type
- For 'strong' requests: return ONLY strong matches
- For 'moderate' requests: return both strong and moderate matches
- If no matches exist, return empty arrays

JSON Output Format:
{{
    "matches": [
        {{
            "name": "John Doe",
            "filename": "john_doe_resume.pdf",
            "email": "john.doe@example.com"
        }},
        {{
            "name": "Jane Smith",
            "filename": "jane_smith_resume.pdf",
            "email": "jane.smith@example.com"
        }}
    ],

    total : 2,
}}

Absolute Requirements:
1. ALWAYS return valid JSON
2. NEVER add explanations outside JSON
3. NEVER repeat the same candidate, ( Eg: if email is same, do not repeat)
4. STRICTLY follow the percentage thresholds
5. Extract email addresses from resumes when available
6. Return ONLY name, email & filename for each matched candidate
7. In total, return only the number of matches else 0

Job Description Requirements: {question}

Resumes to Analyze: {context}
"""

prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

def clear_qdrant_collection():
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True
    )
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("collection deleted")
    except Exception as e:
        print(e)

def extract_text_from_pdf_bytes(pdf_bytes: bytes, filename: str) -> str:
    """Extract text from PDF bytes without saving to temp file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text_content = []
        
        for page in pdf_reader.pages:
            text_content.append(page.extract_text())
        
        return "\n\n".join(text_content)
    except Exception as e:
        print(f"Error extracting text from {filename}: {str(e)}")
        return ""

def process_single_pdf(file_data: tuple) -> dict:
    """Process a single PDF file and return document data"""
    filename, pdf_bytes = file_data

    if not filename.lower().endswith('.pdf'):
        return None
    
    try:
        # Extract text directly from bytes
        full_resume_content = extract_text_from_pdf_bytes(pdf_bytes, filename)
        
        if not full_resume_content.strip():
            print(f"No text extracted from {filename}")
            return None
        
        # Create document data
        doc_data = {
            'page_content': f"RESUME: {filename}\n\n{full_resume_content}",
            'metadata': {
                'source': filename,
                'resume_name': filename,
                'document_type': 'resume'
            }
        }
        
        return doc_data
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None

def create_chunks_from_doc_data(doc_data: dict) -> List[dict]:
    """Create chunks from document data if needed"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = []
    
    # Only split if the resume is larger than chunk size
    if len(doc_data['page_content']) > CHUNK_SIZE:
        # Split the content
        split_texts = text_splitter.split_text(doc_data['page_content'])
        
        for i, chunk_text in enumerate(split_texts):
            # Ensure each chunk retains the resume identifier
            if not chunk_text.startswith("RESUME:"):
                chunk_text = f"RESUME: {doc_data['metadata']['resume_name']}\n\n{chunk_text}"
            
            chunk_data = {
                'page_content': chunk_text,
                'metadata': {
                    **doc_data['metadata'],
                    'chunk_id': i
                }
            }
            chunks.append(chunk_data)
    else:
        chunks.append(doc_data)
    
    return chunks

async def process_pdfs(files: List[UploadFile]):
    """Process multiple PDF files with high performance using async operations."""
    
    # Clear existing collection first
    clear_qdrant_collection()
    
    if not files:
        return 0
        
    # Read all files into memory first (async)
    file_data_list = []
    for file in files:
        content = await file.read()
        file_data_list.append((file.filename, content))
        # Reset file pointer for potential reuse
        await file.seek(0)
    
    print(f"Processing {len(file_data_list)} files...")
    
    # Process PDFs in parallel using ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=min(32, len(file_data_list))) as executor:
        # Submit all PDF processing tasks
        future_to_filename = {
            loop.run_in_executor(executor, process_single_pdf, file_data): file_data[0] 
            for file_data in file_data_list
        }
        
        # Collect results as they complete
        doc_data_list = []
        completed = 0
        
        for future in asyncio.as_completed(future_to_filename):
            try:
                doc_data = await future
                if doc_data:
                    doc_data_list.append(doc_data)
                completed += 1
                
                # Progress indicator
                if completed % 100 == 0:
                    print(f"Processed {completed}/{len(file_data_list)} files...")
                    
            except Exception as e:
                filename = future_to_filename[future]
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"Successfully processed {len(doc_data_list)} PDF files")
    
    if not doc_data_list:
        return 0
    
    # Create chunks in parallel
    print("Creating document chunks...")
    with ThreadPoolExecutor(max_workers=min(16, len(doc_data_list))) as executor:
        chunk_futures = [
            loop.run_in_executor(executor, create_chunks_from_doc_data, doc_data)
            for doc_data in doc_data_list
        ]
        
        all_chunks = []
        for future in asyncio.as_completed(chunk_futures):
            chunks = await future
            all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} document chunks")
    
    # Convert chunk data back to Document objects for Qdrant
    from langchain_core.documents import Document
    documents = []
    for chunk_data in all_chunks:
        doc = Document(
            page_content=chunk_data['page_content'],
            metadata=chunk_data['metadata']
        )
        documents.append(doc)
    
    # Store in Qdrant in batches for better performance
    print("Storing documents in Qdrant...")
    batch_size = 100  # Process in smaller batches to avoid memory issues
    
    try:
        # Create collection with first batch
        if documents:
            first_batch = documents[:batch_size]
            vectorstore = Qdrant.from_documents(
                documents=first_batch,
                embedding=embeddings,
                url=QDRANT_URL,
                collection_name=COLLECTION_NAME,
            )
            
            # Add remaining documents in batches
            for i in range(batch_size, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                vectorstore.add_documents(batch)
                print(f"Stored batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        print(f"Successfully stored {len(documents)} documents in Qdrant")
        
    except Exception as e:
        print(f"Error storing documents in Qdrant: {str(e)}")
        return 0
    
    return len(documents)

def get_retriever():
    """Get the retriever for the PDF collection"""
    return Qdrant.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME
    )

def parse_json_response(response_text: str) -> dict:
    """Parse JSON response from LLM, handling potential formatting issues"""
    try:
        # Try to parse as-is first
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from response if it's wrapped in text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return error structure
        return {
            "error": "Failed to parse JSON response",
            "raw_response": response_text,
            "matches": []
        }

def ask_question_service(question: str) -> dict:
    """Ask a question using RAG on the processed PDFs and return JSON response"""
    try:
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
        
        # Parse the JSON response
        json_response = parse_json_response(result.content)
        return json_response
        
    except Exception as e:
        return {
            "error": f"Service error: {str(e)}",
            "matches": []
        }

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

def get_processed_resumes() -> dict:
    """Get list of all processed resumes in JSON format"""
    try:
        retriever = get_retriever()
        # Get all documents to extract resume names
        all_docs = retriever.as_retriever(search_kwargs={"k": 100}).get_relevant_documents("*")
        
        resume_names = set()
        for doc in all_docs:
            if 'resume_name' in doc.metadata:
                resume_names.add(doc.metadata['resume_name'])
        
        return {
            "status": "success",
            "total_resumes": len(resume_names),
            "resumes": list(resume_names)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "total_resumes": 0,
            "resumes": []
        }

def analyze_specific_resume(resume_name: str, job_description: str) -> dict:
    """Analyze a specific resume against a job description and return JSON response"""
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
            return {
                "status": "error",
                "error": f"Resume '{resume_name}' not found",
                "analysis": None
            }
        
        context = format_docs(docs)
        
        # Create focused analysis chain
        focused_chain = (
            prompt | llm
        )
        
        result = focused_chain.invoke({
            "context": context,
            "question": job_description
        })
        
        # Parse the JSON response
        json_response = parse_json_response(result.content)
        
        return {
            "status": "success",
            "resume_name": resume_name,
            "analysis": json_response
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "resume_name": resume_name,
            "analysis": None
        }