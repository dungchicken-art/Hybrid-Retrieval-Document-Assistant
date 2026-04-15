# CPU-First Deployment Notes

Running a RAG system on a computer without a GPU is possible if the design is lightweight.

Small embedding models such as multilingual E5 small or MiniLM are practical on CPU. Retrieval quality matters more than model size for many document Q and A tasks.

If a local language model is used on CPU, response time will be slower, so hosted APIs are often used for the generation step while retrieval remains local.

For portfolio projects, a CPU-first system is a strong demonstration of engineering tradeoffs because it shows practical design under hardware constraints.
