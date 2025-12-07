import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Read env
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")

CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
CHAT_API_VERSION = os.getenv("AZURE_OPENAI_CHAT_API_VERSION")

EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
EMBED_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")

# Initialize clients
chat_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=CHAT_API_VERSION,
)

embed_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=EMBED_API_VERSION,
)

# Chat Test
print("🔍 Testing Chat Model...")
try:
    chat_response = chat_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": "Say 'Azure chat working'"}]
    )
    # Correct way to access
    print("✔ Chat Test OK:", chat_response.choices[0].message.content)
except Exception as e:
    print("❌ Chat Test Failed:", e)

# Embedding Test
print("\n🔍 Testing Embedding Model...")
try:
    emb_response = embed_client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input="Hello world"
    )
    emb_vector = emb_response.data[0].embedding
    print("✔ Embedding Test OK. Vector length:", len(emb_vector))
except Exception as e:
    print("❌ Embedding Test Failed:", e)
