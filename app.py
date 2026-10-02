"""Hello Agent - CSV FAQ Agent.

Upload one or more CSV files, ask a question in plain English, and get an
answer that comes only from the uploaded data.

Run with:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0  # low temperature = predictable, non-creative answers
NOT_FOUND_MESSAGE = "I could not find this information in the uploaded files."
MAX_ROWS_IN_PROMPT = 30  # small FAQ files are shown to the model in full

SYSTEM_PROMPT = f"""You are Hello Agent, a support assistant that answers questions
using ONLY the data in the pandas data frames provided to you.

The data frames are named df1, df2, ... in the order listed below:
{{file_list}}

Rules you must follow:
1. Use only the values in the data frames. Never use general world knowledge,
   assumptions, or guesses - even if you think you know the answer.
2. Find the most relevant row or rows. For text questions, return the key text
   from the relevant column (for example Answer, Policy Text, Detail Text, or
   Description). Keep the original facts, numbers, and wording accurate.
3. For numeric questions (counts, totals, averages, min/max), compute the result
   with Python on the data frames rather than estimating.
4. If the data does not contain the answer, reply with exactly:
   "{NOT_FOUND_MESSAGE}"
   Do not add anything else in that case.
5. Write the final answer in clear, friendly English that a support agent can
   copy and paste straight into an email or chat. Do not mention data frames,
   Python, code, or row numbers in the final answer.
6. At the end of the answer, on a new line, add "Source:" followed by the file
   name and the ID of the row(s) you used (for example "Source: ecommerce_faqs.csv, ID 102").
   Skip the source line when you return the not-found message.
"""

EXAMPLE_QUESTIONS = [
    "What is the return policy for electronics?",
    "What does the extended warranty cover?",
    "What are the visiting hours in the hospital?",
    "What is the API rate limit for the free plan?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Read uploaded CSV bytes into a data frame (cached per file)."""
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


def build_agent(dataframes: dict[str, pd.DataFrame], api_key: str):
    """Create a LangChain pandas agent over all uploaded data frames."""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, api_key=api_key)

    file_list = "\n".join(
        f"- df{i}: {name} ({len(df)} rows; columns: {', '.join(map(str, df.columns))})"
        for i, (name, df) in enumerate(dataframes.items(), start=1)
    )
    largest = max(len(df) for df in dataframes.values())

    return create_pandas_dataframe_agent(
        llm,
        list(dataframes.values()),
        agent_type="tool-calling",
        # The agent calls .format() on the prefix again, so escape any braces.
        prefix=SYSTEM_PROMPT.format(file_list=file_list).replace("{", "{{").replace("}", "}}"),
        number_of_head_rows=min(largest, MAX_ROWS_IN_PROMPT),
        return_intermediate_steps=True,
        max_iterations=8,
        allow_dangerous_code=True,  # the agent runs pandas code locally on your own data
        agent_executor_kwargs={"handle_parsing_errors": True},
    )


def ask(agent, question: str) -> tuple[str, list]:
    result = agent.invoke({"input": question})
    answer = (result.get("output") or "").strip() or NOT_FOUND_MESSAGE
    return answer, result.get("intermediate_steps", [])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Hello Agent - CSV FAQ", page_icon="💬", layout="wide")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Read from the OPENAI_API_KEY environment variable or a .env file if set.",
    )
    st.caption(f"Model: `{MODEL_NAME}` · temperature {TEMPERATURE}")
    show_steps = st.toggle("Show agent reasoning steps", value=False)
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Upload CSV files\n"
        "2. Ask a question\n"
        "3. The agent searches the tables and answers using only that data"
    )

st.title("💬 Hello Agent")
st.caption("Ask questions about your FAQ and policy CSV files. Answers come only from the uploaded data.")

# 1. File upload ------------------------------------------------------------
st.subheader("1. Upload CSV files")
uploads = st.file_uploader(
    "Drop one or more CSV files here",
    type=["csv"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

dataframes: dict[str, pd.DataFrame] = {}
for f in uploads or []:
    try:
        dataframes[f.name] = load_csv(f.getvalue(), f.name)
    except Exception as e:  # malformed CSV, wrong encoding, etc.
        st.error(f"Could not read **{f.name}**: {e}")

if dataframes:
    tabs = st.tabs([f"{name} ({len(df)} rows)" for name, df in dataframes.items()])
    for tab, df in zip(tabs, dataframes.values()):
        with tab:
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
else:
    st.info("Upload at least one CSV file to get started. Sample files are in the `sample_data` folder.")

# 2. Question ----------------------------------------------------------------
st.subheader("2. Ask a question")

if "question" not in st.session_state:
    st.session_state.question = ""

st.caption("Try an example:")
cols = st.columns(2)
for i, example in enumerate(EXAMPLE_QUESTIONS):
    col = cols[i % 2]
    if col.button(example, use_container_width=True):
        st.session_state.question = example

with st.form("ask_form"):
    question = st.text_input(
        "Your question",
        key="question",
        placeholder="e.g. What is the late payment fee on the Student Saver card?",
    )
    submitted = st.form_submit_button("Get answer", type="primary")

# 3. Answer ------------------------------------------------------------------
if submitted:
    if not dataframes:
        st.warning("Please upload at least one CSV file first.")
    elif not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not question.strip():
        st.warning("Please type a question.")
    else:
        st.subheader("3. Answer")
        with st.spinner("Searching your files..."):
            try:
                agent = build_agent(dataframes, api_key)
                answer, steps = ask(agent, question.strip())
            except Exception as e:
                st.error(f"Something went wrong while answering: {e}")
                st.stop()

        if answer.startswith(NOT_FOUND_MESSAGE):
            st.warning(answer)
        else:
            st.success(answer)
        st.code(answer, language=None, wrap_lines=True)  # built-in copy button
        st.caption("Use the copy icon above to paste this answer into an email or chat.")

        if show_steps and steps:
            with st.expander("Agent reasoning steps"):
                for action, observation in steps:
                    st.markdown("**Code run by the agent**")
                    st.code(action.tool_input.get("query", str(action.tool_input))
                            if isinstance(action.tool_input, dict) else str(action.tool_input),
                            language="python")
                    st.markdown("**Result**")
                    st.text(str(observation)[:2000])
