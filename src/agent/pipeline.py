"""
LangGraph-based Autonomous Research & Decision Agent.

Agentic graph flow:
  START → decompose → retrieve → grade → generate → check_hallucination
    → (grounded)     → synthesize → END
    → (not grounded) → rewrite → retrieve  (loop, max MAX_ITERATIONS)

Handles open-ended strategic questions by decomposing them into
sub-queries, grading retrieval relevance, self-correcting via
rewrite loops, and synthesizing structured decision recommendations.
"""

from typing import List, TypedDict
from langgraph.graph import StateGraph, END

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.generation.generator import call_llm_rest, embed_query_rest
from config import TOP_K, TOP_N

MAX_ITERATIONS = 2  # max rewrite loops before forcing synthesis


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str              # original question
    sub_questions: list        # decomposed sub-queries
    all_documents: list        # retrieved docs (before grading)
    filtered_documents: list   # after relevance grading
    answer: str                # generated answer
    is_grounded: bool          # hallucination check result
    decision: str              # final decision / recommendation
    iteration: int             # current loop count


# ---------------------------------------------------------------------------
# Node: decompose — break open-ended question into sub-queries
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """You are a strategic research planner. Analyze a leadership question and decide:

1. If the question is already SPECIFIC (asks about one clear topic, metric, or event) — return ONLY that question unchanged, on a single line.

2. If the question is OPEN-ENDED or STRATEGIC (requires exploring multiple dimensions like financial impact, competitive risk, operational readiness) — decompose it into 2-4 focused sub-questions, one per line.

Examples of SPECIFIC (return as-is):
- "What were the key risks in the last quarter?"
- "What is Adobe's FY2024 revenue?"
- "What are the risks highlighted in Q3?"

Examples of OPEN-ENDED (decompose):
- "Should we expand into the India market?"
- "How is our competitive position evolving?"
- "What strategic decisions should we make for FY2025?"

Return ONLY the question(s), one per line, no numbering, bullets, or explanation."""


def decompose(state: AgentState) -> dict:
    """Break an open-ended strategic question into focused sub-queries."""
    import time
    t0 = time.perf_counter()
    question = state["question"]
    print(f"  [decompose] Analyzing: {question[:80]}", flush=True)

    messages = [
        {"role": "system", "content": DECOMPOSE_PROMPT},
        {"role": "user", "content": question},
    ]
    raw = call_llm_rest(messages)
    sub_qs = [q.strip() for q in raw.strip().split("\n") if q.strip()]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in sub_qs:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique.append(q)
    sub_qs = unique[:4]  # cap at 4

    elapsed = time.perf_counter() - t0
    print(f"  [decompose] {len(sub_qs)} sub-questions in {elapsed:.2f}s:", flush=True)
    for sq in sub_qs:
        print(f"    → {sq}", flush=True)

    return {"sub_questions": sub_qs}


# ---------------------------------------------------------------------------
# Node: retrieve — multi-query retrieval with deduplication
# ---------------------------------------------------------------------------

def retrieve(state: AgentState, vectorstore: FAISS) -> dict:
    """Retrieve docs for each sub-question, deduplicate by content hash."""
    import time
    t0 = time.perf_counter()
    sub_questions = state.get("sub_questions") or [state["question"]]
    print(f"  [retrieve] Searching across {len(sub_questions)} sub-queries (top_k={TOP_K})", flush=True)

    seen_content = set()
    all_docs = []

    for sq in sub_questions:
        query_embedding = embed_query_rest(sq)
        results = vectorstore.similarity_search_with_score_by_vector(query_embedding, k=TOP_K)

        for doc, score in results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                doc.metadata["relevance_score"] = float(score)
                all_docs.append(doc)
                src = doc.metadata.get("source", "?")
                sec = doc.metadata.get("section", "")
                print(f"    Score: {score:.4f} | {src} | {sec[:40]}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  [retrieve] {len(all_docs)} unique chunks in {elapsed:.2f}s", flush=True)
    return {"all_documents": all_docs}


# ---------------------------------------------------------------------------
# Node: grade — batch LLM relevance grading (single API call)
# ---------------------------------------------------------------------------

GRADE_PROMPT = """You are a relevance grader for a corporate leadership Q&A system.
Given a question and a numbered list of document chunks, decide which chunks are relevant.

A chunk is relevant if it contains information that helps answer the question.

Respond with ONLY a comma-separated list of the relevant chunk numbers (e.g. 1,3,5).
If none are relevant, respond with: none"""


def grade(state: AgentState) -> dict:
    """Grade all retrieved docs in a single LLM call and keep relevant ones."""
    import time
    t0 = time.perf_counter()
    question = state["question"]
    docs = state["all_documents"]
    print(f"  [grade] Grading {len(docs)} docs (batch)...", flush=True)

    # Build numbered list of chunks
    chunks_text = "\n\n".join(
        f"[{i+1}] Source: {doc.metadata.get('source','?')}\n{doc.page_content[:600]}"
        for i, doc in enumerate(docs)
    )

    messages = [
        {"role": "system", "content": GRADE_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nChunks:\n{chunks_text}"},
    ]
    verdict = call_llm_rest(messages, temperature=0).strip().lower()
    print(f"  [grade] Verdict: '{verdict}'", flush=True)

    # Parse which indices are relevant
    filtered = []
    if verdict != "none":
        for part in verdict.replace(" ", "").split(","):
            try:
                idx = int(part) - 1
                if 0 <= idx < len(docs):
                    filtered.append(docs[idx])
                    src = docs[idx].metadata.get("source", "?")
                    print(f"    ✓ relevant: [{idx+1}] {src}", flush=True)
            except ValueError:
                pass

    elapsed = time.perf_counter() - t0
    print(f"  [grade] {len(filtered)}/{len(docs)} docs kept in {elapsed:.2f}s", flush=True)

    # If grading removed everything, fall back to top docs by score
    if not filtered and docs:
        print(f"  [grade] All filtered out — keeping top {TOP_N} by score", flush=True)
        filtered = sorted(docs, key=lambda d: d.metadata.get("relevance_score", 999))[:TOP_N]

    return {"filtered_documents": filtered}


# ---------------------------------------------------------------------------
# Node: generate — synthesize answer from graded docs
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """You are an AI Leadership Insight Agent. Your role is to answer questions 
from company leadership based ONLY on the provided company documents.

Rules:
1. Answer ONLY based on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information, clearly say so.
3. Cite the source document names when referencing specific data or claims.
4. Be concise, factual, and structured. Use bullet points for clarity.
5. For financial data, include specific numbers and time periods when available.
6. Highlight key risks or concerns when relevant to the question."""


def generate(state: AgentState) -> dict:
    """Generate answer from relevant documents."""
    import time
    t0 = time.perf_counter()
    question = state["question"]
    documents = state["filtered_documents"]
    print(f"  [generate] Generating answer from {len(documents)} docs...", flush=True)

    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in documents
    )

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context from company documents:\n{context}\n\nQuestion: {question}\n\nProvide a clear, well-structured answer grounded in the documents above."},
    ]

    answer = call_llm_rest(messages)
    elapsed = time.perf_counter() - t0
    print(f"  [generate] Answer generated in {elapsed:.2f}s", flush=True)
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Node: check_hallucination — verify answer is grounded in docs
# ---------------------------------------------------------------------------

HALLUCINATION_PROMPT = """You are a hallucination grader for a corporate Q&A system.
Assess whether the generated answer is broadly supported by the source documents.

Give a binary score:
- 'yes' if the key facts and claims in the answer are supported by or consistent with the source documents.
- 'no' if the answer contains specific facts, numbers, or claims that directly contradict or are completely absent from the source documents.

Note: The answer may synthesize or rephrase information — that is acceptable.
Respond with ONLY 'yes' or 'no'."""


def check_hallucination(state: AgentState) -> dict:
    """Check if the generated answer is grounded in source documents."""
    import time
    t0 = time.perf_counter()
    answer = state["answer"]
    documents = state["filtered_documents"]
    print(f"  [hallucination] Checking grounding...", flush=True)

    # Use full chunk content (not truncated) so grader has complete context
    doc_text = "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', '?')}]\n{doc.page_content}"
        for doc in documents
    )

    messages = [
        {"role": "system", "content": HALLUCINATION_PROMPT},
        {"role": "user", "content": f"Source documents:\n{doc_text}\n\nGenerated answer:\n{answer}"},
    ]

    verdict = call_llm_rest(messages, temperature=0).strip().lower()
    is_grounded = verdict.startswith("yes")

    elapsed = time.perf_counter() - t0
    print(f"  [hallucination] Grounded={is_grounded} (verdict='{verdict[:20]}') in {elapsed:.2f}s", flush=True)
    return {"is_grounded": is_grounded}


# ---------------------------------------------------------------------------
# Node: rewrite — improve question for better retrieval
# ---------------------------------------------------------------------------

REWRITE_PROMPT = """You are a question rewriter for a corporate document search system.
The previous search did not produce a well-grounded answer. Rewrite the question to be 
more specific and better suited for finding relevant information in financial reports, 
earnings summaries, and strategy documents.

Keep the core intent but make it more precise and search-friendly.
Return ONLY the rewritten question."""


def rewrite(state: AgentState) -> dict:
    """Rewrite the question for better retrieval in the next iteration."""
    import time
    t0 = time.perf_counter()
    question = state["question"]
    iteration = state.get("iteration", 0)
    print(f"  [rewrite] Rewriting question (iteration {iteration + 1})...", flush=True)

    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"Original question: {question}"},
    ]

    rewritten = call_llm_rest(messages)
    print(f"  [rewrite] New question: {rewritten[:100]}", flush=True)
    return {
        "question": rewritten.strip(),
        "sub_questions": [rewritten.strip()],
        "iteration": iteration + 1,
    }


# ---------------------------------------------------------------------------
# Node: synthesize — produce structured decision recommendation
# ---------------------------------------------------------------------------

DECISION_PROMPT = """You are a senior strategic advisor to company leadership. Based on the 
factual answer derived from company documents, produce a structured decision brief.

Format your response as:

**SITUATION SUMMARY**
(2-3 sentence summary of the current state based on the data)

**KEY FINDINGS**
(Bullet points of the most important facts and metrics)

**RISKS & CONCERNS**
(Bullet points of risks, uncertainties, or gaps in the data)

**RECOMMENDATION**
(Clear, actionable recommendation with rationale. If the data is insufficient for a confident recommendation, say so and outline what additional information would be needed.)

Be direct and decisive. Ground every statement in the document evidence."""


def synthesize(state: AgentState) -> dict:
    """Synthesize a structured decision brief from the grounded answer."""
    import time
    t0 = time.perf_counter()
    question = state["question"]
    answer = state["answer"]
    print(f"  [synthesize] Building decision brief...", flush=True)

    messages = [
        {"role": "system", "content": DECISION_PROMPT},
        {"role": "user", "content": f"Leadership Question: {question}\n\nFactual Answer (from documents):\n{answer}\n\nProduce a decision brief."},
    ]

    decision = call_llm_rest(messages)
    elapsed = time.perf_counter() - t0
    print(f"  [synthesize] Decision brief in {elapsed:.2f}s", flush=True)
    return {"decision": decision}


# ---------------------------------------------------------------------------
# Routing function for conditional edges
# ---------------------------------------------------------------------------

def route_after_hallucination(state: AgentState) -> str:
    """Decide whether to synthesize or rewrite based on hallucination check."""
    if state.get("is_grounded", False):
        return "synthesize"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        print(f"  [route] Max iterations reached — proceeding to synthesis", flush=True)
        return "synthesize"
    return "rewrite"


# ---------------------------------------------------------------------------
# Graph builder (documents the full agentic flow)
# ---------------------------------------------------------------------------

def build_agent_graph(vectorstore: FAISS):
    """Build the LangGraph autonomous research & decision agent.

    Flow:
      decompose → retrieve → grade → generate → check_hallucination
        → (grounded)     → synthesize → END
        → (not grounded) → rewrite → retrieve (loop)
    """
    graph = StateGraph(AgentState)

    graph.add_node("decompose", decompose)
    graph.add_node("retrieve", lambda state: retrieve(state, vectorstore))
    graph.add_node("grade", grade)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("rewrite", rewrite)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "generate")
    graph.add_edge("generate", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {"synthesize": "synthesize", "rewrite": "rewrite"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("synthesize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_agent(vectorstore: FAISS, question: str) -> dict:
    """Run the autonomous research & decision agent.

    Executes nodes directly following the graph flow for reliable execution.
    The LangGraph graph definition above documents the full agentic structure.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  AGENT START: {question[:80]}", flush=True)
    print(f"{'='*60}", flush=True)

    state: AgentState = {
        "question": question,
        "sub_questions": [],
        "all_documents": [],
        "filtered_documents": [],
        "answer": "",
        "is_grounded": False,
        "decision": "",
        "iteration": 0,
    }

    # Step 1: Decompose
    state.update(decompose(state))

    for iteration in range(MAX_ITERATIONS + 1):
        # Step 2: Retrieve
        state.update(retrieve(state, vectorstore))

        # Step 3: Grade relevance
        state.update(grade(state))

        # Step 4: Generate answer
        state.update(generate(state))

        # Step 5: Check hallucination
        state.update(check_hallucination(state))

        # Step 6: Route
        route = route_after_hallucination(state)
        if route == "synthesize":
            break
        else:
            # Rewrite and loop
            state.update(rewrite(state))

    # Step 7: Synthesize decision
    state.update(synthesize(state))

    print(f"\n{'='*60}", flush=True)
    print(f"  AGENT COMPLETE (iterations: {state['iteration']})", flush=True)
    print(f"{'='*60}\n", flush=True)

    return state
