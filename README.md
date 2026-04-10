# AutoStream AI Agent 🎬

A conversational AI sales agent for **AutoStream** — a fictional SaaS platform for automated video editing. Built as part of the ServiceHive / Inflx ML Intern assignment.

---

## Features

| Capability | Description |
|---|---|
| 🧠 Intent Detection | Classifies user messages into `greeting`, `product_inquiry`, or `high_intent` |
| 📚 RAG | Answers questions from a local JSON knowledge base (pricing, policies, FAQs) |
| 🎯 Lead Capture | Collects name → email → platform sequentially when high intent is detected |
| 🔧 Tool Execution | Calls `mock_lead_capture()` only after all three fields are collected |
| 💾 State Management | Persists full conversation + lead state across all turns via LangGraph |

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
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

Create a `.env` file (or export directly):

```bash
# For Claude (Anthropic)
export ANTHROPIC_API_KEY=your_key_here

# OR for GPT-4o-mini (OpenAI)
export OPENAI_API_KEY=your_key_here

# OR for Gemini (Google)
export GOOGLE_API_KEY=your_key_here
```

> Swap the LLM import in `agent.py` to match your chosen provider (see comments in file).

### 5. Run the agent

```bash
python agent.py
```

---

## Project Structure

```
autostream-agent/
├── agent.py              # Main agent logic (LangGraph graph)
├── knowledge_base.json   # Local RAG knowledge base
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen because the agent needs to manage **multi-step, stateful conversations** — not just single-turn Q&A. Unlike a simple LangChain chain, LangGraph models the agent as an explicit directed graph of nodes and edges, making state transitions transparent and easy to extend.

### Graph Structure

The graph has two nodes:

1. **`detect_intent`** — Sends the latest user message to the LLM with a strict classification prompt. Returns one of three intents: `greeting`, `product_inquiry`, or `high_intent`.

2. **`generate_response`** — Uses the detected intent and current state to decide what to do:
   - For greetings/inquiries → invokes the LLM with the full system prompt (which includes the knowledge base inline as a RAG context window).
   - For high-intent users → enters a sequential data-collection flow (name → email → platform), storing each piece in `AgentState`.
   - Once all three fields are collected → calls `mock_lead_capture()` exactly once.

### State Management

`AgentState` (a `TypedDict`) is passed through every node. It holds the full message history, current intent, partial lead data (`lead_name`, `lead_email`, `lead_platform`), and flags like `lead_captured` and `awaiting_field`. This ensures memory is retained across all 5–6 conversation turns with zero external storage needed for local use.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp:

### 1. WhatsApp Business API Setup
Register for the [Meta WhatsApp Business API](https://developers.facebook.com/docs/whatsapp) and obtain your `PHONE_NUMBER_ID` and `ACCESS_TOKEN`.

### 2. Webhook Endpoint (FastAPI example)

```python
from fastapi import FastAPI, Request
import httpx

app = FastAPI()
sessions = {}  # In-memory session store (use Redis in production)

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    
    # Extract sender ID and message text
    message = data["entry"][0]["changes"][0]["value"]["messages"][0]
    sender_id = message["from"]
    user_text = message["text"]["body"]
    
    # Retrieve or create session state
    state = sessions.get(sender_id, initial_state())
    state["messages"].append(HumanMessage(content=user_text))
    
    # Run the LangGraph agent
    state = agent.invoke(state)
    sessions[sender_id] = state
    
    # Get the latest AI reply
    reply = next(m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    
    # Send reply back via WhatsApp API
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages",
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "to": sender_id,
                "text": {"body": reply}
            }
        )
    return {"status": "ok"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    # Meta webhook verification handshake
    params = dict(request.query_params)
    if params.get("hub.verify_token") == "MY_VERIFY_TOKEN":
        return int(params["hub.challenge"])
    return {"error": "invalid token"}
```

### 3. Deploy & Register
- Deploy to a public HTTPS endpoint (e.g. Railway, Render, AWS Lambda).
- Register the `/webhook` URL in the Meta Developer Console.
- Set the verify token to match `MY_VERIFY_TOKEN`.

### Key Considerations
- **Session persistence**: Use Redis or a database keyed by `sender_id` instead of in-memory `sessions` dict for production.
- **Concurrency**: Use async FastAPI workers to handle multiple simultaneous conversations.
- **Rate limits**: Meta's API has per-number rate limits — add a queue (e.g. Celery) for high traffic.

---

## Example Conversation

```
You: Hi there!
Assistant: Hey! Welcome to AutoStream 👋 I'm here to help you with pricing, features, or getting started. What can I help you with?

You: What's the difference between Basic and Pro?
Assistant: Great question! Here's a quick comparison:
  • Basic ($29/mo): 10 videos/month, 720p, email support
  • Pro ($79/mo): Unlimited videos, 4K, AI captions, 24/7 support
Would you like more details on either plan?

You: I want to try the Pro plan for my YouTube channel!
Assistant: That's awesome! I'd love to get you set up. Could I have your name?

You: Alex Johnson
Assistant: Nice to meet you, Alex Johnson! What's your email address?

You: alex@example.com
Assistant: Got it! Which creator platform do you primarily use? (e.g. YouTube, Instagram, TikTok)

You: YouTube
✅ Lead captured: Alex Johnson | alex@example.com | YouTube
Assistant: 🎉 Perfect! You're all set, Alex Johnson. We'll be in touch at alex@example.com with your Pro plan details for YouTube creators. Welcome to AutoStream!
```
