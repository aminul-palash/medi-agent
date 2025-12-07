from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

# Import our custom classes
from agent_tools import RetrieverTool
from agent_critic import SelfReflectionCritic
from medical_agent import MedicalAgent

def setup_agent():
    """Initialize all components"""
    load_dotenv()
    
    # Load embeddings
    embedding = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    
    # Load FAISS index
    docsearch = FAISS.load_local(
        "faiss_index",
        embedding,
        allow_dangerous_deserialization=True
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Setup LLM
    llm = AzureChatOpenAI(
        model="gpt-4o",
        azure_deployment="gpt-4o",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2025-01-01-preview",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.7,
    )
    
    # Create components
    retriever_tool = RetrieverTool(retriever)
    critic = SelfReflectionCritic(llm)
    agent = MedicalAgent(retriever_tool, llm, critic)
    
    return agent

def chat_loop():
    """Simple command-line chat interface"""
    print("=" * 60)
    print("🏥 Medical Assistant Agent")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    
    agent = setup_agent()
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not question:
            continue
        
        try:
            result = agent.run(question)
            print(f"\n🤖 Agent: {result['answer']}")
            print(f"📊 Sources used: {result['sources']}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    chat_loop()