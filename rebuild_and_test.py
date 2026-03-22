"""Rebuild FAISS index with Adobe documents and test."""
import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

# Step 1-3: Build index in a subprocess to avoid connection contamination
print("=== Building FAISS Index (subprocess) ===")
result = subprocess.run(
    [sys.executable, "build_index.py"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print(f"ERROR building index:\n{result.stderr}")
    sys.exit(1)

import time
print("Cooldown (5s)...")
time.sleep(5)

# Step 4: Load index and test
print("=== Loading Index from Disk ===")
from src.vectorstore.store import load_faiss_index
from src.agent.pipeline import run_agent

vs = load_faiss_index()

print("\n=== Testing Agent ===")
question = "What is Adobe's current revenue trend?"
print(f"Q: {question}\n")
result = run_agent(vs, question)
print(f"\nAnswer:\n{result['answer']}")
sources = list(set(d.metadata.get("source","?") for d in result.get("filtered_documents",[])))
print(f"\nSources: {sources}")
print("\n=== DONE ===")
