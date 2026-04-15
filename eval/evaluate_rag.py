"""
RAG pipeline evaluation script.

Evaluates the chatbot against a golden dataset measuring:
- Intent classification accuracy
- Retrieval precision (correct source document in top-k)
- Answer keyword coverage (expected keywords present in response)
- Answer faithfulness (LLM-as-judge)

Usage:
    python eval/evaluate_rag.py
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_provider import LLMProvider
from chatbot import ChatbotModel
from rag.ingest import DocumentIngestor
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import HybridRetriever
from rag.generator import GroundedGenerator


def load_golden_dataset(filepath: str = None) -> list[dict]:
    """Load the golden evaluation dataset."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(filepath, "r") as f:
        return json.load(f)


def setup_rag_pipeline(kb_dir: str = None) -> tuple:
    """Initialize the full RAG pipeline."""
    if kb_dir is None:
        kb_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base"
        )

    # Ingest documents
    ingestor = DocumentIngestor(chunk_size=512, chunk_overlap=50)
    documents = ingestor.load_documents(kb_dir)
    chunks = ingestor.chunk_documents(documents)

    # Build embeddings
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.embed([c["content"] for c in chunks])

    # Store in vector DB
    vector_store = VectorStore(
        collection_name="blys_eval", persist_dir="./chroma_db_eval"
    )
    vector_store.reset()
    vector_store.add_documents(chunks, embeddings)

    # Build retriever
    retriever = HybridRetriever(vector_store, embedding_model, chunks)

    # Build generator
    llm = LLMProvider()
    generator = GroundedGenerator(llm)

    # Build chatbot
    chatbot = ChatbotModel(
        llm_provider=llm, retriever=retriever, generator=generator
    )

    return chatbot, retriever


def evaluate_intent_accuracy(chatbot: ChatbotModel, dataset: list[dict]) -> dict:
    """Evaluate intent classification accuracy."""
    correct = 0
    total = 0
    mismatches = []

    for item in dataset:
        result = chatbot.predict(item["query"], session_id=f"eval_{item['id']}")
        predicted = result["intent"]
        expected = item["expected_intent"]

        if predicted == expected:
            correct += 1
        else:
            mismatches.append(
                {
                    "id": item["id"],
                    "query": item["query"],
                    "expected": expected,
                    "predicted": predicted,
                }
            )
        total += 1

    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "correct": correct,
        "total": total,
        "mismatches": mismatches,
    }


def evaluate_retrieval_precision(
    retriever: HybridRetriever, dataset: list[dict], k: int = 5
) -> dict:
    """Evaluate whether the correct source document appears in top-k results."""
    correct = 0
    total = 0
    details = []

    for item in dataset:
        if "expected_source" not in item:
            continue  # Skip action intents

        results = retriever.retrieve(item["query"], top_k=k)
        retrieved_sources = [r["source"] for r in results]
        expected_source = item["expected_source"]

        hit = expected_source in retrieved_sources
        if hit:
            correct += 1
        total += 1

        details.append(
            {
                "id": item["id"],
                "query": item["query"][:50],
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "hit": hit,
            }
        )

    return {
        f"retrieval_precision@{k}": round(correct / total, 4) if total > 0 else 0,
        "correct": correct,
        "total": total,
        "details": details,
    }


def evaluate_answer_coverage(chatbot: ChatbotModel, dataset: list[dict]) -> dict:
    """Evaluate whether expected keywords appear in the chatbot response."""
    scores = []
    details = []

    for item in dataset:
        expected_keywords = item.get("expected_answer_contains", [])
        if not expected_keywords:
            continue

        result = chatbot.predict(item["query"], session_id=f"coverage_{item['id']}")
        response = result["response"].lower()

        hits = sum(1 for kw in expected_keywords if kw.lower() in response)
        coverage = hits / len(expected_keywords) if expected_keywords else 0
        scores.append(coverage)

        details.append(
            {
                "id": item["id"],
                "query": item["query"][:50],
                "coverage": round(coverage, 2),
                "hits": hits,
                "total_keywords": len(expected_keywords),
                "missing": [
                    kw for kw in expected_keywords if kw.lower() not in response
                ],
            }
        )

    avg_coverage = sum(scores) / len(scores) if scores else 0

    return {
        "avg_keyword_coverage": round(avg_coverage, 4),
        "total_evaluated": len(scores),
        "details": details,
    }


def evaluate_faithfulness(
    chatbot: ChatbotModel, dataset: list[dict], llm: LLMProvider
) -> dict:
    """
    LLM-as-judge: evaluate whether responses are grounded in retrieved context.
    Only evaluates information queries (not action intents).
    """
    scores = []
    details = []

    for item in dataset:
        if item.get("expected_intent") in ("reschedule_booking", "cancel_booking", "booking_new"):
            continue

        result = chatbot.predict(
            item["query"], session_id=f"faith_{item['id']}"
        )
        response = result["response"]
        sources = result.get("sources", [])

        judge_prompt = (
            f"You are evaluating whether an AI assistant's response is faithful "
            f"to the information it was given.\n\n"
            f"QUESTION: {item['query']}\n\n"
            f"RESPONSE: {response}\n\n"
            f"SOURCES USED: {', '.join(sources)}\n\n"
            f"Rate the faithfulness on a scale of 1-5:\n"
            f"1 = Completely fabricated / contradicts sources\n"
            f"2 = Mostly fabricated with some grounded elements\n"
            f"3 = Partially grounded but includes unverifiable claims\n"
            f"4 = Mostly grounded with minor extrapolations\n"
            f"5 = Fully grounded in the provided sources\n\n"
            f"Respond with ONLY a single number (1-5)."
        )

        judge_result = llm.complete(judge_prompt)
        try:
            score = int(judge_result.strip()[0])
            score = max(1, min(5, score))
        except (ValueError, IndexError):
            score = 3  # default if parsing fails

        scores.append(score)
        details.append(
            {
                "id": item["id"],
                "query": item["query"][:50],
                "faithfulness_score": score,
                "sources": sources,
            }
        )

    avg_score = sum(scores) / len(scores) if scores else 0

    return {
        "avg_faithfulness": round(avg_score, 2),
        "total_evaluated": len(scores),
        "score_distribution": {
            str(i): scores.count(i) for i in range(1, 6)
        },
        "details": details,
    }


def run_full_evaluation(chatbot=None, retriever=None) -> dict:
    """Run the complete evaluation suite."""
    dataset = load_golden_dataset()

    if chatbot is None or retriever is None:
        chatbot, retriever = setup_rag_pipeline()

    print("=" * 60)
    print("BLYS RAG CHATBOT — EVALUATION REPORT")
    print("=" * 60)

    # 1. Intent accuracy
    print("\n📊 Evaluating intent classification...")
    intent_results = evaluate_intent_accuracy(chatbot, dataset)
    print(f"   Intent Accuracy: {intent_results['accuracy']:.1%}")
    print(f"   ({intent_results['correct']}/{intent_results['total']} correct)")
    if intent_results["mismatches"]:
        print(f"   Mismatches:")
        for m in intent_results["mismatches"][:5]:
            print(f"     - [{m['id']}] '{m['query'][:40]}...' → expected={m['expected']}, got={m['predicted']}")

    # 2. Retrieval precision
    print("\n🔍 Evaluating retrieval precision...")
    retrieval_results = evaluate_retrieval_precision(retriever, dataset, k=5)
    print(f"   Retrieval Precision@5: {retrieval_results['retrieval_precision@5']:.1%}")
    print(f"   ({retrieval_results['correct']}/{retrieval_results['total']} correct)")

    # 3. Answer keyword coverage
    print("\n📝 Evaluating answer keyword coverage...")
    coverage_results = evaluate_answer_coverage(chatbot, dataset)
    print(f"   Avg Keyword Coverage: {coverage_results['avg_keyword_coverage']:.1%}")

    # 4. Faithfulness (LLM-as-judge)
    print("\n🧠 Evaluating faithfulness (LLM-as-judge)...")
    llm = chatbot.llm
    faith_results = evaluate_faithfulness(chatbot, dataset, llm)
    print(f"   Avg Faithfulness: {faith_results['avg_faithfulness']:.1f}/5.0")
    print(f"   Score distribution: {faith_results['score_distribution']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = {
        "intent_accuracy": intent_results["accuracy"],
        "retrieval_precision@5": retrieval_results["retrieval_precision@5"],
        "avg_keyword_coverage": coverage_results["avg_keyword_coverage"],
        "avg_faithfulness": faith_results["avg_faithfulness"],
    }
    for metric, value in summary.items():
        print(f"  {metric}: {value}")

    # Save full results
    full_results = {
        "summary": summary,
        "intent": intent_results,
        "retrieval": retrieval_results,
        "coverage": coverage_results,
        "faithfulness": faith_results,
    }

    results_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n💾 Full results saved to {results_path}")

    return full_results


if __name__ == "__main__":
    run_full_evaluation()
