import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


class RetentionState(TypedDict, total=False):
    profile: Dict[str, Any]
    churn_probability: float
    risk_level: str
    factors: List[str]
    suggestions: List[str]
    positive_reasons: List[str]
    rewards: List[str]
    email_draft: str
    next_actions: List[str]
    chat_history: List[Dict[str, str]]
    user_message: str
    chat_reply: str
    llm_result: Dict[str, Any]
    error: str


class RetentionOutput(BaseModel):
    risk_level: str = Field(description="Use HIGH, MEDIUM, or LOW.")
    factors: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    positive_reasons: List[str] = Field(default_factory=list)
    rewards: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    email_draft: str = Field(default="")
    chat_reply: str = Field(default="")


def _build_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Set it in your environment first.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=api_key,
    )


def _build_messages(state: RetentionState) -> List[Any]:
    history_lines = [
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in state.get("chat_history", [])
    ]
    history_text = "\n".join(history_lines)
    system_guardrails = SystemMessage(
        content=(
            "You are a strictly scoped churn-retention copilot. "
            "You must only work on churn interpretation, retention strategy, reward strategy, "
            "and customer outreach content based on the provided model prediction and profile context. "
            "If the user asks for anything outside churn-retention scope, politely refuse in chat_reply "
            "and redirect to churn analysis tasks. "
            "Use only supplied context; never invent CRM events, never claim certainty, "
            "never produce policy-violating or discriminatory recommendations, and keep output practical. "
            "Treat churn_probability as the primary model signal and explain likely drivers from profile. "
            "Return structured output only using the schema fields."
        )
    )
    user_payload = HumanMessage(
        content=(
            f"Customer profile: {state['profile']}\n"
            f"Model churn probability: {state['churn_probability']}\n"
            f"Conversation history:\n{history_text}\n"
            f"Latest user request: {state.get('user_message', 'Generate retention plan')}"
        )
    )
    return [system_guardrails, user_payload]


def run_copilot(state: RetentionState) -> RetentionState:
    llm = _build_llm()
    structured = llm.with_structured_output(RetentionOutput)
    parsed: RetentionOutput = structured.invoke(_build_messages(state))
    state["risk_level"] = parsed.risk_level
    state["factors"] = parsed.factors
    state["suggestions"] = parsed.suggestions
    state["positive_reasons"] = parsed.positive_reasons
    state["rewards"] = parsed.rewards
    state["next_actions"] = parsed.next_actions
    state["email_draft"] = parsed.email_draft
    state["chat_reply"] = parsed.chat_reply
    state["llm_result"] = parsed.model_dump()
    return state


def compile_output(state: RetentionState) -> RetentionState:
    return state


def _build_graph():
    graph = StateGraph(RetentionState)
    graph.add_node("run_copilot", run_copilot)
    graph.add_node("compile_output", compile_output)
    graph.set_entry_point("run_copilot")
    graph.add_edge("run_copilot", "compile_output")
    graph.add_edge("compile_output", END)
    return graph.compile()


RETENTION_GRAPH = _build_graph()


def run_retention_workflow(
    profile: Dict[str, Any],
    churn_probability: float,
    user_message: str = "Generate retention plan",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> RetentionState:
    initial_state: RetentionState = {
        "profile": profile,
        "churn_probability": float(churn_probability),
        "user_message": user_message,
        "chat_history": chat_history or [],
    }
    try:
        return RETENTION_GRAPH.invoke(initial_state)
    except Exception as exc:  # pragma: no cover
        return {
            "profile": profile,
            "churn_probability": float(churn_probability),
            "risk_level": "UNKNOWN",
            "factors": [],
            "suggestions": [],
            "positive_reasons": [],
            "rewards": [],
            "email_draft": "",
            "next_actions": [],
            "chat_reply": "",
            "error": str(exc),
        }
