"""Streamlit dashboard: chat with the correlation agent, watch its Execution Log, explore the results.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from corr_agent import charts, config
from corr_agent.agent import AgentEvent, run_agent
from corr_agent.analysis import AnalysisResult
from corr_agent.data_store import DatasetStore
from corr_agent.llm import PROVIDERS, get_chat_model, list_gemini_models, list_groq_models, list_ollama_models

EXAMPLES = [
    "Do hot days in New York drive interest in ice cream? Forecast next week.",
    "Does grass pollen in Berlin correlate with views of the German article 'Heuschnupfen'?",
    "Does snowfall in Innsbruck correlate with interest in skiing?",
    "Is PM2.5 pollution in Delhi related to interest in asthma?",
    "Does rain in London affect views of the Bitcoin article?",
    "Does market fear (VIX) drive interest in 'Recession'?",
    "Is US unemployment correlated with views of 'Unemployment benefits'?",
]

ICONS = {"start": "🚀", "decision": "🧠", "step": "⚙️", "tool_result": "✅", "error": "⚠️", "usage": "📊",
         "final": "📝"}
SOURCE_ICONS = {"open-meteo": "🌦️", "wikipedia": "📚", "fred": "🏦", "analysis": "🧮"}


@dataclass
class Turn:
    question: str
    provider: str
    model: str
    events: list[AgentEvent] = field(default_factory=list)
    answer: str = ""
    analyses: list[AnalysisResult] = field(default_factory=list)


# ---------------------------------------------------------------------------- state & sidebar

st.set_page_config(page_title="Correlation Agent", page_icon="📈", layout="wide")
st.session_state.setdefault("turns", [])
st.session_state.setdefault("store", DatasetStore())


@st.cache_data(ttl=600, show_spinner=False)
def _gemini_models() -> list[str]:
    return list_gemini_models()


@st.cache_data(ttl=600, show_spinner=False)
def _groq_models() -> list[str]:
    return list_groq_models()


with st.sidebar:
    st.header("⚙️ Settings")
    providers = list(PROVIDERS)
    provider = st.radio("LLM provider", providers, horizontal=True,
                        index=providers.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in providers else 0,
                        format_func=lambda p: {"gemini": "Gemini", "groq": "Groq", "ollama": "Ollama (local)"}[p])
    if provider == "gemini":
        models = _gemini_models() or [config.GEMINI_MODEL]
        default = config.GEMINI_MODEL
        st.caption("✅ GEMINI_API_KEY found" if config.GEMINI_API_KEY else "❌ GEMINI_API_KEY missing (see .env.example)")
    elif provider == "groq":
        models = _groq_models() or [config.GROQ_MODEL]
        default = config.GROQ_MODEL
        st.caption("✅ GROQ_API_KEY found" if config.GROQ_API_KEY else "❌ GROQ_API_KEY missing (see .env.example)")
    else:
        models = list_ollama_models()
        default = config.OLLAMA_MODEL
        st.caption(f"✅ Ollama reachable at {config.OLLAMA_HOST}" if models
                   else f"❌ Ollama not reachable at {config.OLLAMA_HOST}")
        models = models or [config.OLLAMA_MODEL]
    model = st.selectbox("Model", models, index=models.index(default) if default in models else 0,
                         accept_new_options=True, help="Pick a listed model or type another name.")

    st.caption("**Data sources**")
    st.caption("🌦️ Open-Meteo: weather, air quality, pollen ✅")
    st.caption("📚 Wikipedia pageviews ✅")
    st.caption("🏦 FRED: economic data " + ("✅" if config.FRED_API_KEY else "❌ (set FRED_API_KEY)"))

    st.divider()
    st.subheader("💡 Try a question")
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"ex{i}", width="stretch"):
            st.session_state.queued = q
    st.divider()
    if st.button("🗑️ New conversation", width="stretch"):
        st.session_state.turns = []
        st.session_state.store = DatasetStore()
        st.rerun()


# ---------------------------------------------------------------------------- rendering helpers

def log_line(event: AgentEvent) -> str:
    icon = SOURCE_ICONS.get(event.data.get("source"), ICONS.get(event.kind, "•"))
    if event.kind == "error":
        icon = ICONS["error"]
    return f"`{event.ts:%H:%M:%S}` {icon} {event.message}"


def short(text: str, n: int = 55) -> str:
    return text if len(text) <= n else text[:n] + "…"


def render_log(turn: Turn, idx: int, expanded: bool) -> None:
    with st.expander(f"#{idx + 1} · {short(turn.question)}", expanded=expanded):
        st.caption(f"{turn.provider} · {turn.model}")
        for ev in turn.events:
            if ev.kind != "final":
                st.markdown(log_line(ev))


def _p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def render_analysis(res: AnalysisResult, key: str) -> None:
    st.markdown(f"**{res.x_label}** vs **{res.y_label}** · {res.strength} · "
                f"{'significant' if res.significant else 'not significant'} at 5%")
    cols = st.columns(6)
    cols[0].metric("Pearson r", f"{res.pearson_r:.2f}")
    cols[1].metric("Spearman ρ", f"{res.spearman_rho:.2f}")
    cols[2].metric("p-value", _p(min(res.pearson_p, res.spearman_p)))
    cols[3].metric(f"n ({res.period}s)", res.n)
    cols[4].metric("Best lag", f"{res.best_lag:+d} {res.period[0]}", help="Positive: X leads Y")
    best_r2 = max((v for v in (res.linear_r2_cv, res.rf_r2_cv) if v is not None), default=None)
    cols[5].metric("Model R² (CV)", "—" if best_r2 is None else f"{best_r2:.2f}",
                   help=f"Out-of-sample R², linear = {res.linear_r2_cv}, random forest = {res.rf_r2_cv}")

    names = ["📈 Time series", "🔵 Scatter", "↔️ Lag scan"]
    if res.forecast is not None:
        names.append("🔮 Forecast")
    names.append("🗂️ Data")
    tabs = dict(zip(names, st.tabs(names)))
    tabs["📈 Time series"].plotly_chart(charts.timeseries(res), key=f"{key}-ts")
    tabs["🔵 Scatter"].plotly_chart(charts.scatter(res), key=f"{key}-sc")
    tabs["↔️ Lag scan"].plotly_chart(charts.lags(res), key=f"{key}-lag")
    if res.forecast is not None:
        tabs["🔮 Forecast"].plotly_chart(charts.forecast(res), key=f"{key}-fc")
        tabs["🔮 Forecast"].dataframe(res.forecast.rename(columns={
            "x": res.x_label, "y_pred_linear": "Predicted (linear)", "y_pred_rf": "Predicted (random forest)"}))
    table = res.data.rename(columns={"x": res.x_label, "y": res.y_label})
    tabs["🗂️ Data"].dataframe(table, height=300)
    for note in res.notes:
        st.caption("ℹ️ " + note)


def render_turn(turn: Turn, idx: int) -> None:
    with st.chat_message("user"):
        st.markdown(f"**#{idx + 1}** · {turn.question}")
    with st.chat_message("assistant"):
        st.markdown(turn.answer)
        for j, res in enumerate(turn.analyses):
            render_analysis(res, key=f"t{idx}-a{j}")


def llm_history(turns: list[Turn], max_turns: int = 4) -> list[BaseMessage]:
    """Previous Q&A as plain messages so follow-up questions have context."""
    msgs: list[BaseMessage] = []
    for t in turns[-max_turns:]:
        msgs += [HumanMessage(t.question), AIMessage(t.answer)]
    return msgs


# ---------------------------------------------------------------------------- page

# Keep the Execution Log column in view while the chat scrolls (the marker div identifies the column).
LOG_PANEL_HEIGHT = 620
st.markdown("""
<style>
div[data-testid="stColumn"]:has(.log-panel-marker) {
    position: sticky;
    top: 3.75rem;
    align-self: flex-start;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Correlation Agent")
st.caption("Ask about relationships between weather, air quality, public interest (Wikipedia) and the economy "
           "(FRED). The agent picks the data sources, fetches the data, runs the statistics and explains them.")
st.caption("🤖 Answers are generated by an AI model and can be wrong; check the Execution Log and the data tabs. "
           "Data: Open-Meteo (CC BY 4.0), Wikimedia, and FRED® (this product uses the FRED® API but is not "
           "endorsed or certified by the Federal Reserve Bank of St. Louis).")

chat_col, log_col = st.columns([3, 2], gap="large")
turns: list[Turn] = st.session_state.turns

with log_col:
    st.markdown('<div class="log-panel-marker"></div>', unsafe_allow_html=True)
    st.subheader("🛠️ Execution Log")
    # Its own scrolling window, oldest run first; autoscroll keeps the newest step in view.
    log_panel = st.container(height=LOG_PANEL_HEIGHT, border=True, autoscroll=True)
    with log_panel:
        if not turns:
            st.caption("The agent's steps (LLM decisions, API calls, model training) will appear here.")
        for i, t in enumerate(turns):
            render_log(t, i, expanded=(i == len(turns) - 1))

with chat_col:
    for i, t in enumerate(turns):
        render_turn(t, i)

question = st.chat_input("e.g. Does rain in London affect views of the Bitcoin article?")
question = question or st.session_state.pop("queued", None)

if question:
    turn = Turn(question=question, provider=provider, model=model)
    store: DatasetStore = st.session_state.store
    n_before = len(store.analyses)

    with chat_col:
        with st.chat_message("user"):
            st.markdown(f"**#{len(turns) + 1}** · {question}")
        thinking = st.chat_message("assistant").empty()
        thinking.markdown("_Working on it… follow along in the Execution Log._")

    with log_panel:
        status = st.status(f"#{len(turns) + 1} · {short(question)}", expanded=True)
        try:
            llm = get_chat_model(provider, model)
        except Exception as e:
            status.update(label="Could not start the LLM", state="error")
            status.error(str(e))
            st.stop()
        failed = False
        for event in run_agent(question, llm, store, history=llm_history(turns)):
            turn.events.append(event)
            if event.kind == "final":
                turn.answer = event.message
            else:
                status.markdown(log_line(event))
                failed = failed or event.message.startswith("Agent run failed")
        status.update(label="Agent failed" if failed else "Done", state="error" if failed else "complete")

    turn.analyses = store.analyses[n_before:]
    turns.append(turn)
    st.rerun()
