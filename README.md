# Cancer Bioinformatics RAG | Local LLM & Agentic AI

A Retrieval-Augmented Generation (RAG) system for answering scientific questions in cancer bioinformatics using peer-reviewed biomedical literature. It combines semantic retrieval, cross-encoder reranking, local LLM inference, and developing external literature augmentation to produce evidence-grounded responses.

## Project Overview

Large language models can generate plausible but unsupported answers to specialized biomedical questions. This project explores a literature-grounded question-answering pipeline using open-source tools and local inference, without requiring paid LLM APIs.

Key objectives:
- Retrieve relevant biomedical articles using semantic similarity.
- Improve relevance with cross-encoder reranking.
- Generate context-aware answers with a locally hosted LLM.
- Incorporate PubMed retrieval when local evidence is insufficient.
- Evaluate retrieval using established information retrieval metrics.
- Extend the workflow toward conditional, agentic retrieval.

## System Architecture

### 1. Literature collection

Approximately 860 biomedical articles were collected through NCBI Entrez and PubMed Central resources.

- PubMed ESearch for literature discovery
- PMC Open Access BioC for full-text retrieval
- Article metadata: PMID, PMCID, title, authors, publication date
- JSON storage for downstream processing

### 2. Document processing and embedding

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Chunk size: 1,000 characters
- Chunk overlap: 150 characters
- Vector database: ChromaDB (persistent local storage)
- Metadata mapping chunks to source publications

### 3. Semantic retrieval

User queries are embedded with the same sentence-transformer model and matched against the local corpus using cosine similarity to retrieve relevant text chunks.

### 4. Cross-encoder reranking

- Model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Framework: Sentence Transformers
- Hardware acceleration: Apple Metal Performance Shaders (MPS)

A second-stage relevance model reranks query–passage pairs before prompting the LLM.

### 5. Local LLM inference

- Model: Meta Llama 3.1 8B Instruct
- Quantization: GGUF Q4_K_M
- Inference engine: `llama-cpp-python`
- Context window: 4,096 tokens
- GPU acceleration: Apple Metal
- Temperature: 0.2

The highest-ranked passages are supplied as context to encourage literature-grounded responses.

### 6. External literature augmentation (in development)

The project is being extended with evidence sufficiency checks and conditional retrieval through NCBI Entrez/PubMed when the local corpus is inadequate.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Literature Retrieval | NCBI Entrez, PubMed, PMC |
| Document Processing | JSON, Python |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Initial Retrieval | Cosine similarity |
| Reranking | ms-marco-MiniLM-L6-v2 |
| LLM | Meta Llama 3.1 8B Instruct |
| LLM Inference | llama-cpp-python |
| GPU Acceleration | Apple Metal / MPS |
| Evaluation | Precision@K, Recall@K, MRR@K |

## Retrieval Evaluation

Initial evaluation compared cosine-similarity retrieval with cross-encoder reranking using a question set and known-positive relevance judgments.

| Metric (K = 5) | Baseline | Reranked |
|---|---:|---:|
| Precision@5 | 0.071 | 0.100 |
| Recall@5 | 0.355 | 0.500 |
| MRR@5 | 0.238 | 0.429 |

**Observations:** Precision@5 increased by approximately 41%, Recall@5 increased from 0.355 to 0.500, and MRR@5 increased by approximately 80% after reranking.

> These are initial benchmark results. Known-positive judgments may not cover all relevant passages; different evaluation runs should not be directly compared without aligning the test set and qrels.

## Agentic RAG Development

The intended conditional workflow is:

1. Receive a biomedical question.
2. Retrieve candidate passages from ChromaDB.
3. Rerank the passages with a cross-encoder.
4. Assess whether the retrieved evidence is sufficient.
5. If sufficient, generate a locally grounded answer.
6. Otherwise, retrieve additional literature through NCBI Entrez.
7. Incorporate the additional evidence into the answer.

Evidence sufficiency assessment and the end-to-end conditional workflow are still being developed and tested.

## Example Research Questions

- How can machine learning distinguish malignant and non-malignant cells from single-cell RNA-seq data?
- How can scRNA-seq and machine learning characterize tumor microenvironment heterogeneity?
- Which computational methods investigate cancer-associated expression signatures?
- How can machine learning support cancer biomarker discovery?

## Project Status

| Component | Status |
|---|---|
| Literature collection | Completed |
| Chunking and embeddings | Completed |
| Persistent ChromaDB storage | Completed |
| Semantic retrieval | Completed |
| Cross-encoder reranking | Completed |
| Local Llama 3.1 inference | Completed |
| Initial retrieval evaluation | Completed |
| PubMed/Entrez integration | Implemented; refinement ongoing |
| Evidence sufficiency assessment | In development |
| Conditional agentic workflow | In development |
| End-to-end agentic evaluation | Planned |

## Future Work

- Refine evidence sufficiency checks and conditional literature retrieval.
- Improve citation consistency and source attribution.
- Strengthen relevance judgments and expand the evaluation set.
- Compare full agentic performance against the baseline RAG pipeline.
- Reduce redundant retrieval and improve document diversity.

## Hardware

Developed locally on a 14-inch MacBook Pro with Apple M2 Pro and 16 GB unified memory, using Metal acceleration where supported.

## Disclaimer

This is a biomedical research and machine-learning demonstration. Generated answers should be verified against primary literature and are not intended for clinical decision-making.

## Author

**Nilavrah Sensarma**  
M.S. Computational Science and Engineering  
Professional Master's in Bioinformatics  
Georgia Institute of Technology

[GitHub](https://github.com/neilsensarma) | [LinkedIn](https://www.linkedin.com/in/nilavra-sensarma/)
