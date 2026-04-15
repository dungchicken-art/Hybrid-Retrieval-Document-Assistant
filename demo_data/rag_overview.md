# Retrieval-Augmented Generation Overview

Retrieval-augmented generation, or RAG, combines document retrieval with language model generation.

The retrieval step finds relevant chunks from a knowledge base before the model answers the question. This reduces hallucination and makes the answer more traceable to real sources.

Modern RAG systems often combine dense semantic search and sparse keyword search. This is usually called hybrid retrieval.

For small projects, a local index on disk can be enough. For larger systems, teams often move to vector databases and add rerankers.
