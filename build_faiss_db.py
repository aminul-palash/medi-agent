# build_faiss_db.py
"""
Single script to build FAISS vector database from PDF files
Usage: python build_faiss_db.py
"""

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

# Configuration
PDF_FOLDER = "data"  # Folder containing your PDF files
FAISS_INDEX_PATH = "faiss_index"  # Where to save the FAISS index
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 20  # Overlap between chunks

def load_pdfs(folder_path):
    """Load all PDF files from folder"""
    print(f"📂 Loading PDFs from: {folder_path}")
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    loader = DirectoryLoader(
        folder_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from PDFs")
    return documents

def split_documents(documents):
    """Split documents into chunks"""
    print(f"✂️  Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE[:300],
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks")
    return chunks

def create_embeddings():
    """Initialize Azure OpenAI embeddings"""
    print("🔧 Initializing embeddings...")
    
    load_dotenv()
    
    embedding = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    
    print("✅ Embeddings initialized")
    return embedding

def build_faiss_index(chunks, embedding, batch_size=100):
    """Build FAISS index from chunks"""
    print(f"🔨 Building FAISS index...")
    
    total_chunks = len(chunks)
    
    # Process in batches to avoid memory issues
    if total_chunks > batch_size:
        print(f"Processing in batches of {batch_size}...")
        
        # Create initial index with first batch
        docsearch = FAISS.from_documents(
            documents=chunks[:batch_size],
            embedding=embedding
        )
        print(f"✅ Processed batch 1/{(total_chunks + batch_size - 1) // batch_size}")
        
        # Add remaining batches
        for i in range(batch_size, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            docsearch.add_documents(batch)
            print(f"✅ Processed batch {batch_num}/{total_batches}")
    else:
        # Process all at once if small dataset
        docsearch = FAISS.from_documents(
            documents=chunks,
            embedding=embedding
        )
    
    print(f"✅ FAISS index created with {total_chunks} documents")
    return docsearch

def save_index(docsearch, path):
    """Save FAISS index to disk"""
    print(f"💾 Saving index to: {path}")
    docsearch.save_local(path)
    print("✅ Index saved successfully!")

def main():
    """Main execution"""
    print("=" * 60)
    print("🏗️  Building FAISS Vector Database")
    print("=" * 60)
    
    try:
        # Step 1: Load PDFs
        documents = load_pdfs(PDF_FOLDER)
        
        # Step 2: Split into chunks
        chunks = split_documents(documents)
        
        # Step 3: Initialize embeddings
        embedding = create_embeddings()
        
        # Step 4: Build FAISS index
        docsearch = build_faiss_index(chunks, embedding)
        
        # Step 5: Save to disk
        save_index(docsearch, FAISS_INDEX_PATH)
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! FAISS database built successfully")
        print("=" * 60)
        print(f"📊 Statistics:")
        print(f"   - Total pages: {len(documents)}")
        print(f"   - Total chunks: {len(chunks)}")
        print(f"   - Index location: {FAISS_INDEX_PATH}")
        print(f"\n💡 You can now use this index in your agent!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise

if __name__ == "__main__":
    main()