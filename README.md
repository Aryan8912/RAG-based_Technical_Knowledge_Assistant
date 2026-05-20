# RAG-based Technical Knowledge Assistant 

**End-to-end applied ML engineering across three production-grade problem domains:**
LLM failure diagnosis, retrieval-augmented generation, and text classification.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quick Start](#quick-start)
3. [Section 1 — Diagnose a Failing LLM Pipeline](#section-1--diagnose-a-failing-llm-pipeline)
4. [Section 2 — Production-Grade RAG Pipeline](#section-2--production-grade-rag-pipeline)
5. [Section 3 — Ticket Classifier](#section-3--ticket-classifier)
6. [Evaluation Results](#evaluation-results)
7. [Design Decisions](#design-decisions)
8. [References](#references)

---

## Project Structure

```
.
├── Artikate_ML_Engineer_Task.ipynb   # Main notebook — all three sections
├── README.md                          # This file
├── legal_pdfs/                        # Real PDF corpus for Section 2
│   ├── master_subscription_agreement.pdf
│   ├── data_processing_addendum.pdf
│   ├── service_level_agreement.pdf
│   ├── acceptable_use_policy.pdf
│   └── refund_cancellation_policy.pdf
├── qa_eval/
│   ├── qa_pairs.json                  # 10 golden QA pairs with relevant_doc_ids
│   └── eval_results.json              # Per-query Precision@3 and faithfulness scores
├── investigation_log.json             # Timestamped Section 1 diagnosis log
└── confusion_matrix.png               # Section 3 confusion matrix output
```

---

## Quick Start

### Prerequisites

Python 3.9 or later. GPU is optional — DistilBERT runs comfortably on CPU for the dataset size used here.

```bash
pip install sentence-transformers faiss-cpu transformers datasets \
            rank-bm25 accelerate pypdf reportlab scikit-learn \
            pandas numpy seaborn matplotlib evaluate
```

### Running the notebook

```bash
jupyter notebook Artikate_ML_Engineer_Task.ipynb
```

Or open directly in Google Colab using the badge at the top of the notebook. Run cells in order from top to bottom. Section 2 expects the `legal_pdfs/` directory to be present alongside the notebook.

---

## Section 1 — Diagnose a Failing LLM Pipeline

### Overview

Three production failure modes were diagnosed end-to-end using simulated production traces, structured root-cause analysis, and prioritised mitigations. A non-technical post-mortem was written for stakeholder communication.

### Failure Modes Investigated

| # | Failure | Root Cause | Severity |
|---|---------|-----------|----------|
| 1 | **Hallucination** | Conflicting retrieved chunks; parametric memory override; numeric fabrication | 🔴 Critical |
| 2 | **Language Switching** | Language detection output never propagated to retrieval or prompt layers | 🟠 High |
| 3 | **Latency Degradation** | `top_k` config changed 5 → 20 silently; no prompt token budget guard | 🟡 Medium |

### Methodology

**Hallucination diagnosis** runs a keyword faithfulness score (fraction of response tokens present in the retrieved context) against four simulated production traces. Traces scoring below 0.70 are flagged and routed to the root-cause tree. Three categories of hallucination are identified — retrieval noise, parametric override, and numeric fabrication — each with a distinct mitigation.

**Language switching diagnosis** models a five-trace multilingual pipeline and traces the signal loss between the language detection layer and the retrieval and prompt assembly stages. The causal chain is printed as an explicit flowchart.

**Latency diagnosis** simulates 72 hours of production traffic across three phases (baseline, creep, spike) and computes Pearson correlation between latency and both prompt token count (r ≈ +0.97) and chunk count (r ≈ +0.95). The root cause — an unreviewed `top_k` config change — is pinpointed from the correlation evidence and a timestamped deployment timeline.

### Outputs

- **`investigation_log.json`** — 30 structured, timestamped log entries covering all three failure modes, suitable for audit trail use.
- **Non-technical post-mortem** — printed in-notebook, covering what happened, why, customer impact, fix timeline, and a before/after systemic changes table.

---

## Section 2 — Production-Grade RAG Pipeline

### Overview

A full retrieval-augmented generation pipeline built over five real legal and contractual PDF documents, with two-stage retrieval, source citation, and a Precision@3 evaluation harness.

### Architecture

```
PDF Documents
      │
      ▼
  PdfReader (pypdf)
      │
      ▼
  Sentence-Aware Chunker          ← 400-char max, 1-sentence overlap
      │
      ▼
  SentenceTransformer Embedder    ← all-MiniLM-L6-v2
      │
      ▼
  FAISS IndexFlatL2               ← Stage 1: dense semantic recall
      │
      ▼
  BM25Okapi Re-Ranker             ← Stage 2: exact-keyword precision
      │
      ▼
  Prompt Builder                  ← "Answer ONLY from context"
      │
      ▼
  Flan-T5-Base Generator
      │
      ▼
  Source Citation (doc_id + title)
      │
      ▼
  Precision@3 Evaluation Harness
```

### Corpus

Five real, multi-section PDF documents generated with `reportlab` and read with `pypdf`. Each document is two pages and 380–550 extractable words:

| File | Domain | Words |
|------|--------|-------|
| `master_subscription_agreement.pdf` | Contract terms, fees, termination, liability | 554 |
| `data_processing_addendum.pdf` | GDPR, encryption, breach notification, sub-processors | 533 |
| `service_level_agreement.pdf` | Uptime SLA, incident response, service credits | 447 |
| `acceptable_use_policy.pdf` | Prohibited activities, enforcement, content rules | 379 |
| `refund_cancellation_policy.pdf` | Cancellation, monthly/annual refunds, processing times | 411 |

**Total corpus: 2,324 extractable words across 5 documents.**

### Retrieval Design

**Stage 1 — FAISS dense search** encodes all chunks with `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings, `IndexFlatL2`) and retrieves the top-K candidates by semantic similarity. Dense retrieval is fast and handles paraphrase queries well but can miss exact legal terms.

**Stage 2 — BM25Okapi re-ranking** re-scores the FAISS candidates using BM25 term-frequency weighting. This boosts chunks containing exact-match legal phrases — "AES-256", "seventy-two (72) hours", "99.99 percent" — that dense embeddings may dilute.

**Chunking** splits on sentence boundaries (not raw word count) and carries one sentence of overlap forward, preventing mid-clause cuts that lose contractual context.

### Evaluation — Precision@3

For each of 10 golden QA pairs, Precision@3 measures the fraction of the top-3 retrieved chunks whose `doc_id` matches the `relevant_doc_ids` for that query. A score of 1.0 means all three returned chunks came from the correct source document.

| Metric | Threshold | Description |
|--------|-----------|-------------|
| **Precision@3** | ≥ 0.67 | Relevant doc in top-3 retrieved chunks |
| **Faithfulness** | ≥ 0.70 | Response token overlap with retrieved context |
| **Fact Coverage** | — | Ground-truth string present in generated answer |
| **Retrieval Latency P95** | — | End-to-end retrieval time |

QA pairs and full results are saved to `qa_eval/qa_pairs.json` and `qa_eval/eval_results.json`.

---

## Section 3 — Ticket Classifier

### Overview

A DistilBERT sequence classifier fine-tuned to route customer support tickets into five mutually exclusive categories, with macro F1 evaluation on genuinely unseen test data and a sub-500ms latency assertion.

### Label Set

Per task specification:

| ID | Label | Description |
|----|-------|-------------|
| 0 | `billing` | Invoice errors, payment failures, refund requests, pricing disputes |
| 1 | `technical_issue` | Bugs, crashes, API errors, data sync failures, broken UI |
| 2 | `feature_request` | New functionality suggestions, integration requests, UX improvements |
| 3 | `complaint` | Unresolved frustration, service quality, trust/escalation issues |
| 4 | `other` | General questions, how-to queries, account setup, compliance enquiries |

### Dataset

| Split | Sentences | Per Class | Overlap |
|-------|-----------|-----------|---------|
| Training | 250 | 50 | — |
| Test (held-out) | 50 | 10 | 0 sentences shared with training |

Every sentence in both splits was written independently. A runtime assertion (`assert len(overlap) == 0`) verifies there is no contamination between splits before any training occurs.

> **Why this matters:** The original dataset ran `for _ in range(200): expanded_data.extend(data)`, producing 2,000 rows from 10 unique sentences. After `train_test_split`, the test set contained identical sentences to training. The model memorised rather than generalised, and any reported accuracy was statistically invalid.

### Model

**DistilBERT-base-uncased** fine-tuned for 5-class sequence classification using the HuggingFace `Trainer` API.

- 66M parameters (40% fewer than BERT-base)
- Retains ~97% of BERT-base performance on GLUE benchmarks
- Runs on CPU within the 500ms per-ticket latency constraint
- `max_length=64` — sufficient for single-sentence support tickets

**Training configuration:**

```python
TrainingArguments(
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    num_train_epochs            = 2,
    save_strategy               = "no",
)
```

### Evaluation

Evaluation uses **macro-averaged F1** on the 50 held-out test sentences. Macro averaging weights each class equally regardless of support size, which is appropriate given the balanced 10-sample-per-class test distribution.

The confusion matrix is saved to `confusion_matrix.png`. A latency assertion verifies that per-ticket inference stays below 500ms on CPU:

```python
assert (end - start) / len(sample_tickets) < 0.5
```

---

## Evaluation Results

### Section 1 — Diagnosis Log

| Failure | Log Entries | Root Causes Found | Mitigations |
|---------|-------------|------------------|-------------|
| Hallucination | 11 | 3 (retrieval noise, parametric override, fabrication) | 4 |
| Language Switching | 10 | 1 (detection-propagation gap) | 4 |
| Latency Degradation | 9 | 1 (top_k config bloat) | 6 |

### Section 2 — RAG Evaluation

| Metric | Result | Threshold |
|--------|--------|-----------|
| Mean Precision@3 | Reported in `qa_eval/eval_results.json` | ≥ 0.67 |
| Mean Faithfulness | Reported in `qa_eval/eval_results.json` | ≥ 0.70 |
| Fact Coverage | Reported at runtime | — |
| Retrieval P95 | Reported at runtime (ms) | — |

### Section 3 — Classifier

| Metric | Result |
|--------|--------|
| Macro F1 (50 held-out samples) | Reported at runtime |
| Per-ticket latency | < 500ms ✅ |
| Label validity assertion | Passes ✅ |
| Train/test overlap | 0 sentences ✅ |

Exact scores vary by hardware and model initialisation. The notebook prints the full `classification_report` (per-class precision, recall, F1) and saves the confusion matrix on every run.

---

## Design Decisions

### Why two-stage retrieval (FAISS + BM25)?

Dense vector search optimises for recall — it finds semantically similar chunks but can miss exact legal phrases. BM25 re-ranking adds a precision layer that rewards exact term matches ("AES-256", "seventy-two (72) hours") without requiring a GPU or API call. The two stages are complementary: FAISS handles paraphrase, BM25 handles specificity.

### Why sentence-boundary chunking with overlap?

Raw word-count splitting (the original `chunk_size=50` approach) cuts mid-sentence, breaking contractual clauses across chunk boundaries. Sentence-boundary splitting preserves the complete meaning of each clause. One-sentence overlap means that the final sentence of one chunk is also the first sentence of the next, preventing context loss at boundaries.

### Why real PDFs instead of hardcoded strings?

Precision@3 requires that different queries retrieve chunks from *different* source documents. With only three one-sentence strings, every query retrieves the same three chunks regardless of content — the metric is trivially 0.33 for all queries and tells you nothing about retrieval quality. Real, multi-section documents with distinct vocabulary per topic make the retrieval problem non-trivial and the metric meaningful.

### Why DistilBERT over a larger model?

The task specification includes a 500ms per-ticket latency constraint. DistilBERT achieves ~97% of BERT-base GLUE performance at 60% of the parameter count and runs within budget on CPU. For a support ticket router — where inputs are single short sentences — the capacity of a larger model is unnecessary and the latency penalty is not justified.

### Why macro-averaged F1?

Macro averaging weights each class equally, which is the correct choice when class imbalance would otherwise cause a model to appear accurate by simply predicting the majority class. With a balanced 10-samples-per-class test set, macro and weighted F1 are numerically similar, but macro F1 is the more conservative and defensible choice for reporting.

---

## References

**Es, S., James, J., Anke, L. E., & Schockaert, S. (2023).** RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217.* — Defines faithfulness as the fraction of answer claims supported by retrieved context; used as the basis for the token-overlap faithfulness proxy in Section 1 and Section 2.

**Reimers, N., & Gurevych, I. (2019).** Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP 2019.* — Describes the `all-MiniLM-L6-v2` model family used for dense chunk encoding in Section 2.

**Robertson, S., & Zaragoza, H. (2009).** The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval, 3(4), 333–389.* — The algorithm underlying `rank-bm25` (BM25Okapi) used for re-ranking in Section 2.

**Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019).** DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS 2019 EMC² Workshop.* — Describes the DistilBERT architecture used for ticket classification in Section 3.

**Lewis, P., Perez, E., Piktus, A., et al. (2020).** Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems (NeurIPS 2020).* — The foundational RAG architecture that motivates the retrieve-then-generate design in Section 2.
