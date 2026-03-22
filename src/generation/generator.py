import json

from langchain_openai import AzureChatOpenAI

from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_API_VERSION, LLM_DEPLOYMENT, EMBEDDING_DEPLOYMENT


def _rest_call(deployment: str, payload: dict, timeout: int = 60) -> dict:
    """Make a REST call to Azure OpenAI via requests with a fresh session."""
    import requests
    import time
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{deployment}?api-version={AZURE_API_VERSION}"
    headers = {"api-key": AZURE_OPENAI_KEY, "Content-Type": "application/json"}
    # Brief pause to avoid connection throttling (corporate firewall/proxy)
    time.sleep(1)
    with requests.Session() as session:
        resp = session.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
    return result


def get_llm() -> AzureChatOpenAI:
    """Get the Azure OpenAI LLM instance."""
    return AzureChatOpenAI(
        azure_deployment=LLM_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_API_VERSION,
        temperature=0,
        max_retries=2,
        timeout=60,
    )


def call_llm_rest(messages: list[dict], temperature: float = 0, max_retries: int = 3) -> str:
    """Call Azure OpenAI chat completions via httpx (fresh connection each call)."""
    import time

    t0 = time.perf_counter()
    payload = {"messages": messages, "temperature": temperature}

    for attempt in range(max_retries):
        try:
            body = _rest_call(f"{LLM_DEPLOYMENT}/chat/completions", payload)
            elapsed = time.perf_counter() - t0
            usage = body.get("usage", {})
            print(f"    [LLM] Response in {elapsed:.2f}s | Tokens: prompt={usage.get('prompt_tokens','?')} completion={usage.get('completion_tokens','?')} total={usage.get('total_tokens','?')}", flush=True)
            return body["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    [LLM] Retry {attempt+1}/{max_retries} after {wait}s: {e}", flush=True)
                import time as t; t.sleep(wait)
            else:
                raise


def embed_query_rest(text: str) -> list[float]:
    """Embed a query string via Azure OpenAI REST API (fresh connection)."""
    import time
    t0 = time.perf_counter()
    body = _rest_call(f"{EMBEDDING_DEPLOYMENT}/embeddings", {"input": text})
    elapsed = time.perf_counter() - t0
    print(f"    [Embed] Query embedded in {elapsed:.2f}s", flush=True)
    return body["data"][0]["embedding"]



