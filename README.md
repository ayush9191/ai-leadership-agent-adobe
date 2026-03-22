# AI Leadership Insight & Decision Agent

An AI-powered assistant for company leadership that answers questions about organizational performance using internal documents and handles open-ended strategic questions through autonomous research and decision-making.

## Problem Statement

Leadership teams often need fast, defensible answers to questions about business performance, strategic risk, and operational priorities. In practice, those answers are buried across quarterly earnings reports, annual filings, dashboards, and strategy documents. Manually reviewing those materials is slow, fragmented, and difficult to scale, while generic chatbots can produce answers that are not grounded in the source material.

This repository addresses that problem by building a document-grounded leadership assistant. It ingests internal business documents, indexes them for retrieval, decomposes broad strategic questions into targeted sub-questions, retrieves the most relevant evidence, checks whether the response is supported by the documents, and returns a structured answer with source-backed decision guidance.

In short, the system is designed to help leadership teams move from scattered documents to faster, evidence-based decisions.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Agentic Pipeline (LangGraph)                 │
│                                                                  │
│  ┌───────────┐   ┌──────────┐   ┌───────┐   ┌──────────┐       │
│  │ Decompose │──▶│ Retrieve │──▶│ Grade │──▶│ Generate │       │
│  └───────────┘   └──────────┘   └───────┘   └──────────┘       │
│                       ▲                           │              │
│                       │                           ▼              │
│                  ┌─────────┐              ┌──────────────┐       │
│                  │ Rewrite │◀── (no) ◀───│ Hallucination│       │
│                  └─────────┘              │    Check     │       │
│                                           └──────┬───────┘       │
│                                             (yes)│               │
│                                                  ▼               │
│                                          ┌────────────┐          │
│                                          │ Synthesize │──▶ END   │
│                                          └────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

**Nodes:**
| Node | Purpose |
|------|---------|
| **Decompose** | Breaks open-ended questions into 2–4 focused sub-queries |
| **Retrieve** | Multi-query FAISS retrieval with deduplication |
| **Grade** | LLM-based relevance scoring — filters irrelevant chunks |
| **Generate** | Synthesizes factual answer grounded in documents |
| **Hallucination Check** | Verifies answer is supported by source documents |
| **Rewrite** | Rewrites query for better retrieval if not grounded (max 2 retries) |
| **Synthesize** | Produces structured decision brief: Situation → Findings → Risks → Recommendation |

## Tech Stack

- **Python 3.13**
- **LangChain** + **LangGraph** — document processing, embeddings, agentic graph
- **FAISS** — local vector store for similarity search
- **Azure OpenAI** — GPT-4.1 (LLM) + text-embedding-3-small (embeddings)
- **Streamlit** — chat UI with document management

## Project Structure

```
ai-leadership-agent/
├── app.py                          # Streamlit UI
├── build_index.py                  # Standalone index builder
├── config.py                       # Configuration (models, paths, params)
├── requirements.txt
├── .env.example                    # Environment template
├── data/
│   └── documents/                  # Place company docs here (PDF, DOCX, TXT, CSV, XLSX)
├── faiss_index/                    # Generated vector index (auto-created)
└── src/
    ├── ingestion/
    │   ├── loader.py               # Multi-format document loader
    │   └── chunker.py              # Hybrid chunking (section-aware + semantic)
    ├── vectorstore/
    │   └── store.py                # FAISS index creation / loading
    ├── retrieval/
    │   └── retriever.py            # Document retrieval helpers
    ├── generation/
    │   └── generator.py            # Azure OpenAI REST calls, LLM + embeddings
    └── agent/
        └── pipeline.py             # LangGraph agentic pipeline (7 nodes)
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your Azure OpenAI credentials:
   ```
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
   AZURE_OPENAI_KEY=your-api-key
   ```

3. **Add documents:**
   Place company documents (PDF, DOCX, TXT, CSV, XLSX) in `data/documents/`.

4. **Build the vector index:**
   ```bash
   python build_index.py
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Chunking Strategy

The system uses a **hybrid chunking approach** optimized for retrieval quality:

- **Structured documents** (with headers/sections): Section-aware splitting preserves topic boundaries. Headers are detected via regex patterns (markdown, ALL CAPS, separator lines). Large sections are sub-split.
- **Unstructured documents** (no headers): Semantic chunking via `SemanticChunker` embeds each sentence and cuts at topic shifts (75th percentile embedding dissimilarity). Oversized chunks are sub-split.

## Usage

### Streamlit UI
- Upload documents via sidebar
- Build/rebuild index with one click
- Ask questions in the chat interface
- View decision brief, source citations, and agent trace

### Example Questions
- *"What is our current revenue trend?"*
- *"Which departments are underperforming?"*
- *"What were the key risks highlighted in the last quarter?"*
- *"Should we increase investment in the India market?"*
- *"What is our biggest competitive vulnerability?"*

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_DEPLOYMENT` | `gpt-4.1` | Azure OpenAI chat model |
| `EMBEDDING_DEPLOYMENT` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per sub-query |
| `TOP_N` | `3` | Chunks kept after re-ranking |
