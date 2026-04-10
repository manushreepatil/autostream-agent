"""
AutoStream Conversational AI Agent
Built with LangGraph + Gemini 1.5 Flash (FREE API)
Features: Intent Detection, RAG, Lead Capture Tool
"""

import os
import json
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

# Load .env file automatically
load_dotenv()


# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE (RAG)
# ─────────────────────────────────────────────

def load_knowledge_base(path: str = "knowledge_base.json") -> str:
    """Load the local knowledge base and return it as a formatted string."""
    kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(kb_path, "r") as f:
        kb = json.load(f)

    text = f"""
COMPANY: {kb['company']}
DESCRIPTION: {kb['description']}

== PRICING ==
Basic Plan ({kb['pricing']['basic_plan']['price']}):
  Features: {', '.join(kb['pricing']['basic_plan']['features'])}

Pro Plan ({kb['pricing']['pro_plan']['price']}):
  Features: {', '.join(kb['pricing']['pro_plan']['features'])}

== POLICIES ==
- Refund: {kb['policies']['refund_policy']}
- Support: {kb['policies']['support']}
- Free Trial: {kb['policies']['free_trial']}
- Cancellation: {kb['policies']['cancellation']}

== FAQs ==
"""
    for faq in kb["faqs"]:
        text += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"

    return text.strip()


KNOWLEDGE_BASE = load_knowledge_base("knowledge_base.json")


# ─────────────────────────────────────────────
# 2. MOCK LEAD CAPTURE TOOL
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API that simulates saving a lead to a CRM."""
    print(f"\n{'='*50}")
    print(f"  Lead captured successfully!")
    print(f"   Name     : {name}")
    print(f"   Email    : {email}")
    print(f"   Platform : {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured: {name} | {email} | {platform}"


# ─────────────────────────────────────────────
# 3. AGENT STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None
    lead_captured: bool
    awaiting_field: str | None


# ─────────────────────────────────────────────
# 4. LLM SETUP — GEMINI 1.5 FLASH (FREE)
# ─────────────────────────────────────────────

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file!")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.3,
    )

llm = get_llm()


# ─────────────────────────────────────────────
# 5. SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are an AI sales assistant for AutoStream, a SaaS platform that provides 
automated video editing tools for content creators.

Your goals:
1. Greet users warmly.
2. Answer product, pricing, and policy questions using ONLY the knowledge base below.
3. Identify high-intent users (those who want to sign up, try the product, or buy a plan).
4. Collect name, email, and creator platform (YouTube, Instagram, etc.) from high-intent users.

KNOWLEDGE BASE:
{KNOWLEDGE_BASE}

RULES:
- Never make up information not in the knowledge base.
- Do NOT ask for lead details until the user has shown clear buying intent.
- Collect only ONE piece of information per message (name first, then email, then platform).
- Be concise, friendly, and helpful.
- If unsure of intent, ask a clarifying question rather than assuming.
"""


# ─────────────────────────────────────────────
# 6. INTENT DETECTION NODE
# ─────────────────────────────────────────────

def detect_intent(state: AgentState) -> AgentState:
    """Classify the latest user message into one of three intents."""

    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    classification_prompt = f"""Classify the following user message into exactly one of these intents:
1. greeting          - casual hello, how are you, etc.
2. product_inquiry   - asking about features, pricing, policies, comparisons
3. high_intent       - ready to sign up, wants to try, wants to buy, asking how to get started

User message: "{last_user_msg}"

Reply with ONLY the intent label (greeting / product_inquiry / high_intent). No explanation."""

    result = llm.invoke([HumanMessage(content=classification_prompt)])
    intent = result.content.strip().lower()

    if "high" in intent:
        intent = "high_intent"
    elif "product" in intent or "inquiry" in intent:
        intent = "product_inquiry"
    else:
        intent = "greeting"

    return {**state, "intent": intent}


# ─────────────────────────────────────────────
# 7. RESPONSE GENERATION NODE
# ─────────────────────────────────────────────

def generate_response(state: AgentState) -> AgentState:
    """Generate the agent's reply based on current state."""

    intent        = state.get("intent", "greeting")
    lead_captured = state.get("lead_captured", False)
    awaiting      = state.get("awaiting_field")
    lead_name     = state.get("lead_name")
    lead_email    = state.get("lead_email")

    # A. Lead already captured
    if lead_captured:
        response_text = (
            f"You're all set, {lead_name}! Our team will reach out to {lead_email} "
            f"shortly. Is there anything else I can help you with?"
        )
        return {**state, "messages": state["messages"] + [AIMessage(content=response_text)]}

    # B. Collecting lead info step-by-step
    if intent == "high_intent" or awaiting:

        last_user_msg = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        ).strip()

        updates: dict = {}

        if awaiting == "name":
            updates["lead_name"]      = last_user_msg
            updates["awaiting_field"] = "email"
            response_text = f"Nice to meet you, {last_user_msg}! What's your email address?"

        elif awaiting == "email":
            updates["lead_email"]     = last_user_msg
            updates["awaiting_field"] = "platform"
            response_text = (
                "Got it! Which creator platform do you primarily use? "
                "(e.g. YouTube, Instagram, TikTok, Twitter/X)"
            )

        elif awaiting == "platform":
            updates["lead_platform"]  = last_user_msg
            updates["awaiting_field"] = None
            mock_lead_capture(lead_name, lead_email, last_user_msg)
            updates["lead_captured"]  = True
            response_text = (
                f"Perfect! You're all set, {lead_name}. "
                f"We'll be in touch at {lead_email} with your Pro plan details "
                f"for {last_user_msg} creators. Welcome to AutoStream!"
            )

        else:
            # First high-intent detection — start collecting
            updates["awaiting_field"] = "name"
            response_text = (
                "That's awesome! I'd love to get you set up. "
                "To get started, could I have your name?"
            )

        return {**state, **updates, "messages": state["messages"] + [AIMessage(content=response_text)]}

    # C. Normal RAG-powered conversation
    messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    result = llm.invoke(messages_for_llm)
    return {**state, "messages": state["messages"] + [AIMessage(content=result.content)]}


# ─────────────────────────────────────────────
# 8. BUILD THE LANGGRAPH
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("generate_response", generate_response)
    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()


agent = build_graph()


# ─────────────────────────────────────────────
# 9. INITIAL STATE
# ─────────────────────────────────────────────

def initial_state() -> AgentState:
    return AgentState(
        messages=[],
        intent="greeting",
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
        awaiting_field=None,
    )


# ─────────────────────────────────────────────
# 10. MAIN CHAT LOOP
# ─────────────────────────────────────────────

def chat():
    print("\n" + "="*50)
    print("   Welcome to AutoStream AI Assistant!")
    print("   Type 'exit' or 'quit' to end the chat.")
    print("="*50 + "\n")

    state = initial_state()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("\nAssistant: Thanks for chatting! Have a great day.\n")
            break

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        state = agent.invoke(state)

        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
        )
        if last_ai:
            print(f"\nAssistant: {last_ai.content}\n")


if __name__ == "__main__":
    chat()
