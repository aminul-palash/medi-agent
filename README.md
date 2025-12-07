# Medi Agent Project

A **RAG-based AI assistant** leveraging **Azure OpenAI** and **FAISS** for intelligent responses.

---

## Setup Instructions

### Step 1: Create a Conda Environment

Open your terminal in the project repository and run:

```bash
# Create a new Conda environment with Python 3.10
conda create -n medibot python=3.10 -y

# Activate the environment
conda activate medibot
```

---

### Step 2: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

### Step 3: Configure Environment Variables

Create a `.env` file in the root directory and add your Azure OpenAI and Pinecone credentials:

```dotenv
# ==== Azure OpenAI Core ====
AZURE_OPENAI_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key

# Chat Model (GPT-4o deployment)
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_CHAT_API_VERSION=2025-01-01-preview

# Embedding Model (text-embedding-ada-002 deployment)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment_name
AZURE_OPENAI_EMBEDDING_API_VERSION=2023-05-15
```

**⚠️ Note:** Do **not** commit your `.env` file to version control. Ensure deployment names exactly match your Azure OpenAI setup.

---

### Step 4: Verify Setup (Optional)

Test that your environment, Azure OpenAI, and Pinecone keys are working:

```bash
python test_azure_env.py
```

> You can replace the above script with your own verification scripts as needed.

---

### Step 5: Run the Project

Build the document embedding in faiss vector DB. places you documents in 'data' folder. currently only pdf medical documents are supported.

```bash
python build_faiss_db.py
```

Run the main application:

```bash
python main.py
```

---

### Step 6: Run Flask API

Start the Flask API server:

```bash
python api.py
```

By default, it runs at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### Step 7: Test the API

Example Python code to test the API:

```python
import requests

url = "http://127.0.0.1:5000/ask"
data = {"question": "What is the treatment for acne?"}

response = requests.post(url, json=data)
print(response.json())
```

You should receive a JSON response from the AI assistant.

---

**Notes:**

* Ensure Azure OpenAI deployment names match exactly with the `.env` configuration.
* Keep your `.env` file private and out of version control.

---

## 🏗️ Architecture Components

### 1. **Retriever Tool** (`agent_tools.py`)
- **Purpose**: Fetch relevant medical documents from vector database
- **Input**: User query
- **Output**: Top-3 similar documents
- **Tech**: FAISS similarity search with Azure OpenAI embeddings

### 2. **Self-Reflection Critic** (`agent_critic.py`)
- **Purpose**: Quality assurance - evaluates if answer needs improvement
- **Decision Logic**: 
  - ✅ "GOOD" → Accept answer
  - 🔄 "IMPROVE" → Regenerate with feedback
- **Max Iterations**: 2 attempts to improve

### 3. **Medical Agent** (`medical_agent.py`)
- **Core Workflow**:
  ```
  Query → Retrieve Context → Generate Answer → Critique → Improve (if needed) → Return
  ```
- **Memory**: Stores last 5 Q&A pairs for context-aware follow-ups
- **State Management**: In-memory conversation history

### 4. **API Layer** (`api.py`)
- **Endpoints**:
  - `POST /ask` - Process query
  - `POST /clear` - Reset conversation
  - `GET /health` - Health check
- **Session**: Single agent instance (stateful per server)

### 5. **UI** (`templates/chat.html`)
- **Two-Panel Design**:
  - Left: Chat interface
  - Right: Real-time agent thinking logs
- **Features**: Message history, clear button

## 🔄 Agent Execution Flow

```
1. User Question
   ↓
2. Retrieve Context (FAISS search)
   ↓
3. Generate Initial Answer (LLM + Context + History)
   ↓
4. Self-Reflection Loop:
   ├─ Critique Answer
   ├─ If GOOD → Done ✓
   └─ If IMPROVE → Regenerate → Repeat
   ↓
5. Store in Memory
   ↓
6. Return Response
```

## 💡 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **FAISS Vector DB** | Fast similarity search, no external dependencies |
| **In-Memory History** | Simple, fast, sufficient for single-user demo |
| **Max 2 Reflection Loops** | Balance quality vs latency (prevents infinite loops) |
| **Last 5 Conversations** | Prevents context overflow while maintaining continuity |
| **Stateful Agent** | Single instance per server, simpler than per-session state |

## 🚀 System Flow Example

**Query**: "What is Acne?"

```python
# Step 1: Retrieve
docs = retriever.search("What is Acne?")  # Returns 3 docs

# Step 2: Generate
answer = llm.generate(context=docs, question="What is Acne?")

# Step 3: Critique
feedback = critic.evaluate(answer)  
# Result: "IMPROVE: Add treatment options"

# Step 4: Improve
final_answer = llm.generate(
    context=docs, 
    question="What is Acne?",
    previous=answer,
    feedback="Add treatment options"
)

# Step 5: Store
history.append({"q": "What is Acne?", "a": final_answer})
```

## 🔧 Technical Stack
```
Backend:     Python 3.x, Flask
LLM:         Azure OpenAI GPT-4o
Embeddings:  text-embedding-ada-002
Vector DB:   FAISS (in-memory)
Frontend:    Vanilla HTML/CSS/JS
```

## 📝 Code Organization

```
project/
|__ data                # folder to store pdf
|__ research            # notebook for intial r&d
|__ build_faiss_db      # to build the faiss db from the pdf
├── agent_tools.py      # Retriever wrapper
├── agent_critic.py     # Self-reflection logic
├── medical_agent.py    # Main orchestration + memory
├── main.py             # CLI interface
├── api.py              # Web API + routes
└── templates/
    └── chat.html       # UI
```