import sys
import os
import subprocess

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config import DOCUMENTS_DIR, AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT
from src.vectorstore.store import load_faiss_index
from src.agent.pipeline import run_agent

# --- Page config ---
st.set_page_config(
    page_title="AI Leadership Insight Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Leadership Insight Agent")
st.caption("Ask questions about company performance, strategy, and operations — powered by your documents.")


# --- Sidebar: API key & document management ---
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Azure OpenAI Key",
        value=AZURE_OPENAI_KEY or "",
        type="password",
        help="Enter your Azure OpenAI API key",
    )
    azure_endpoint = st.text_input(
        "Azure OpenAI Endpoint",
        value=AZURE_OPENAI_ENDPOINT or "",
        help="Enter your Azure OpenAI endpoint URL",
    )
    if api_key and azure_endpoint:
        import config
        config.AZURE_OPENAI_KEY = api_key
        config.AZURE_OPENAI_ENDPOINT = azure_endpoint

    st.divider()
    st.header("📁 Documents")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload company documents",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} file(s) to documents folder.")

    # Show existing documents
    if os.path.exists(DOCUMENTS_DIR):
        existing_files = os.listdir(DOCUMENTS_DIR)
        if existing_files:
            st.write(f"**Documents in store ({len(existing_files)}):**")
            for f in existing_files:
                st.write(f"  📄 {f}")
        else:
            st.info("No documents yet. Upload some above.")

    st.divider()

    # Index building
    if st.button("🔄 Build / Rebuild Index", use_container_width=True):
        if not api_key or not azure_endpoint:
            st.error("Please enter your Azure OpenAI key and endpoint first.")
        elif not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
            st.error("No documents found. Upload documents first.")
        else:
            with st.spinner("Building FAISS index (this may take a moment)..."):
                # Run index building in a subprocess to avoid connection pool contamination
                project_root = os.path.dirname(os.path.abspath(__file__))
                proc = subprocess.run(
                    [sys.executable, "build_index.py"],
                    cwd=project_root,
                    capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    st.error(f"Index build failed:\n{proc.stderr}")
                else:
                    # Load from disk in fresh state
                    vectorstore = load_faiss_index()
                    st.session_state["vectorstore"] = vectorstore
                    st.success(f"Index built! {proc.stdout.strip()}")


# --- Load existing index on startup ---
if "vectorstore" not in st.session_state:
    vs = load_faiss_index()
    if vs is not None:
        st.session_state["vectorstore"] = vs


# --- Chat interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

# Chat input
if prompt := st.chat_input("Ask a leadership question about your company..."):
    # Validate
    if not api_key or not azure_endpoint:
        st.error("Please enter your Azure OpenAI credentials in the sidebar.")
    elif "vectorstore" not in st.session_state:
        st.error("Please build the document index first (sidebar → Build Index).")
    else:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Researching documents & building decision brief..."):
                result = run_agent(st.session_state["vectorstore"], prompt)

            # Show the factual answer
            answer = result.get("answer", "I couldn't generate an answer.")
            st.markdown(answer)

            # Show decision brief if available
            decision = result.get("decision", "")
            if decision:
                st.divider()
                st.markdown(decision)

            # Extract sources
            sources = list(set(
                doc.metadata.get("source", "Unknown")
                for doc in result.get("filtered_documents", [])
            ))
            if sources:
                with st.expander("📎 Sources"):
                    for src in sources:
                        st.write(f"- {src}")

            iterations = result.get("iteration", 0)
            sub_qs = result.get("sub_questions", [])
            with st.expander("🔍 Agent Trace"):
                st.write(f"**Sub-queries explored:** {len(sub_qs)}")
                for sq in sub_qs:
                    st.write(f"  → {sq}")
                st.write(f"**Docs retrieved:** {len(result.get('all_documents', []))}")
                st.write(f"**Docs after grading:** {len(result.get('filtered_documents', []))}")
                st.write(f"**Grounded:** {result.get('is_grounded', '?')}")
                st.write(f"**Rewrite iterations:** {iterations}")

            # Save to history (show decision in history too)
            display_content = answer
            if decision:
                display_content += "\n\n---\n\n" + decision

            st.session_state["messages"].append({
                "role": "assistant",
                "content": display_content,
                "sources": sources,
            })
