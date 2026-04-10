# AutoStream AI Agent 🎬

A conversational AI sales agent for **AutoStream** — a SaaS platform for automated video editing tools.
Built for the **ServiceHive / Inflx ML Intern Assignment** using LangGraph + Gemini 1.5 Flash.

---

## Demo Video
> 📹 [Watch Demo on Loom](https://www.loom.com/share/0f38e566738444d6a38fe72d99c3c094)

---

## Features

| Capability | Description |
|---|---|
| Intent Detection | Classifies messages into `greeting`, `product_inquiry`, or `high_intent` |
| RAG | Answers questions from a local JSON knowledge base |
| Lead Capture | Collects name → email → platform when high intent is detected |
| Tool Execution | Calls `mock_lead_capture()` only after all 3 fields collected |
| State Management | Full conversation + lead state persisted via LangGraph across all turns |

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key
- Create a .env file
- Add:

GROQ_API_KEY=your_key_here
```

### 5. Run the agent
```bash
python agent.py
```

---

## Project Structure

```
autostream-agent/
├── agent.py              # Main agent (LangGraph graph)
├── knowledge_base.json   # Local RAG knowledge base
├── requirements.txt      # Python dependencies
├── .env.example          # API key template (rename to .env)
├── .gitignore            # Keeps .env out of GitHub
└── README.md
```

---

## Architecture Explanation

### Why LangGraph?
LangGraph was chosen because this agent needs to manage **multi-step, stateful conversations** across multiple turns — not just single-turn Q&A. LangGraph models the agent as an explicit directed graph of nodes and edges, making the state transitions transparent, debuggable, and easy to extend compared to a simple LangChain chain.

### Graph Structure
The graph has two nodes that run in sequence on every user message:

**Node 1 — `detect_intent`**: Sends the latest user message to Gemini 1.5 Flash with a strict classification prompt. Returns one of three intent labels: `greeting`, `product_inquiry`, or `high_intent`.

**Node 2 — `generate_response`**: Uses the detected intent plus the full `AgentState` to decide what to do:
- Greetings/product inquiries → calls Gemini with the full system prompt (knowledge base is embedded inline as a RAG context window).
- High-intent users → enters a sequential data-collection flow: name → email → platform, storing each answer in `AgentState`.
- Once all three fields are confirmed → calls `mock_lead_capture()` exactly once and sets the `lead_captured` flag.

### State Management
`AgentState` (a `TypedDict`) flows through every node. It stores the complete message history, detected intent, partial lead data (`lead_name`, `lead_email`, `lead_platform`), and control flags (`lead_captured`, `awaiting_field`). This ensures all conversation context is retained across 5–6 turns with no external database needed for local use.

---

## WhatsApp Deployment via Webhooks

### Overview
To deploy on WhatsApp, we use the **Meta WhatsApp Business API** with a **webhook** that receives incoming messages and sends replies.

### Step 1 – Get WhatsApp API Access
Register at [Meta for Developers](https://developers.facebook.com/docs/whatsapp) and obtain:
- `PHONE_NUMBER_ID`
- `WHATSAPP_TOKEN`

### Step 2 – Create a FastAPI Webhook Server

```python
from fastapi import FastAPI, Request
import httpx
from agent import agent, initial_state
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()
sessions = {}  # Use Redis in production

PHONE_NUMBER_ID = "your_phone_number_id"
WHATSAPP_TOKEN  = "your_whatsapp_token"

@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()
    message  = data["entry"][0]["changes"][0]["value"]["messages"][0]
    sender   = message["from"]
    text     = message["text"]["body"]

    # Load or create session
    state = sessions.get(sender, initial_state())
    state["messages"] = state["messages"] + [HumanMessage(content=text)]
    state = agent.invoke(state)
    sessions[sender] = state

    # Get latest AI reply
    reply = next(m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage))

    # Send reply back via WhatsApp
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            json={"messaging_product": "whatsapp", "to": sender, "text": {"body": reply}}
        )
    return {"status": "ok"}

@app.get("/webhook")
async def verify(request: Request):
    params = dict(request.query_params)
    if params.get("hub.verify_token") == "MY_VERIFY_TOKEN":
        return int(params["hub.challenge"])
    return {"error": "forbidden"}
```

### Step 3 – Deploy & Register
1. Deploy to a public HTTPS URL (Railway, Render, or AWS Lambda are free options)
2. Register the `/webhook` URL in the Meta Developer Console
3. Set the verify token to `MY_VERIFY_TOKEN`

### Production Considerations
- Replace in-memory `sessions` dict with **Redis** keyed by `sender_id`
- Add a **message queue** (Celery + Redis) to handle high traffic
- Use **async FastAPI** workers for concurrent conversations

---

## Example Conversation

```
You: Hi there!
Assistant: Hey! Welcome to AutoStream. I'm here to help with pricing,
           features, or getting you started. What can I help you with?

You: What's the difference between Basic and Pro?
Assistant: Great question! Basic ($29/mo): 10 videos/month, 720p, email support.
           Pro ($79/mo): Unlimited videos, 4K, AI captions, 24/7 support.

You: I want to try the Pro plan for my YouTube channel!
Assistant: That's awesome! To get started, could I have your name?

You: Alex Johnson
Assistant: Nice to meet you, Alex Johnson! What's your email address?

You: alex@example.com
Assistant: Got it! Which creator platform do you primarily use?

You: YouTube
==================================================
  Lead captured successfully!
   Name     : Alex Johnson
   Email    : alex@example.com
   Platform : YouTube
==================================================
Assistant: Perfect! You're all set, Alex Johnson. We'll be in touch
           at alex@example.com for YouTube creators. Welcome to AutoStream!
```
