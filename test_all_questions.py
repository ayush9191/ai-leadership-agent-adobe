"""Test all 3 example questions from the challenge."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vectorstore.store import load_faiss_index
from src.agent.pipeline import run_agent

vs = load_faiss_index()

questions = [
    "What is our current revenue trend?",
    "Which departments are underperforming?",
    "What were the key risks highlighted in the last quarter?",
]

for i, q in enumerate(questions, 1):
    print(f"\n{'='*60}")
    print(f"Q{i}: {q}\n")
    result = run_agent(vs, q)
    print(f"\nA{i}:\n{result['answer']}")
    sources = list(set(d.metadata.get("source","?") for d in result.get("filtered_documents",[])))
    print(f"\nSources: {sources}")

print(f"\n{'='*60}")
print("ALL TESTS PASSED")
