# Model Card: StudyMind AI

## Reflection and Ethics

### 1. What are the limitations or biases in your system?
StudyMind AI is heavily dependent on the quality and comprehensiveness of the user-provided study notes. If the ingested documents contain biased, outdated, or incorrect information, the RAG engine will retrieve and amplify those flaws. Furthermore, the underlying LLM (GPT-4.1-mini) may exhibit systemic biases present in its pre-training data, particularly when answering questions where the retrieved context is sparse or ambiguous. The system's task classifier is currently rule-based (keyword matching), which limits its ability to understand nuanced or complex intents that do not contain explicit keywords like "quiz" or "summarize."

### 2. Could your AI be misused, and how would you prevent that?
Yes, the system could be misused by students attempting to have the AI complete assignments for them rather than using it as a study aid. To mitigate this, I implemented a Guardrails module that uses regex pattern matching to detect and warn against off-topic queries (e.g., "write my essay for me"). Additionally, the system prompt explicitly instructs the AI to act as an academic assistant, focusing on explanations and quizzes rather than generating long-form essays. The system also blocks prompt injection attempts (e.g., "ignore previous instructions") to prevent users from overriding these safety constraints.

### 3. What surprised you while testing your AI's reliability?
I was surprised by how sensitive the system's confidence scoring was to the length of the retrieved context. Initially, the system would often flag perfectly accurate answers with a "low confidence" warning simply because the relevant chunk was short. I had to adjust the confidence calculation to weight the generation confidence and self-verification steps more heavily than the raw cosine similarity score from the vector database. This highlighted the difficulty of creating a single, reliable metric for "confidence" in a multi-step agentic workflow.

### 4. Describe your collaboration with AI during this project.
During the development of this project, I collaborated extensively with AI tools for code generation and debugging.

**Helpful Suggestion:**
When designing the RAG engine, the AI suggested using a local `sentence-transformers` model (`all-MiniLM-L6-v2`) instead of relying on the OpenAI embeddings API. This was an excellent architectural decision because it eliminated the need for an API key for the embedding step, made the system faster for local testing, and improved privacy by keeping document processing entirely on the local machine.

**Flawed Suggestion:**
While building the automated test harness, the AI initially suggested using a purely string-matching approach to verify the LLM's output (e.g., checking if the exact phrase "overfitting occurs when" was in the answer). This proved to be brittle and incorrect, as the LLM would often rephrase the concept accurately but fail the test because the exact string was missing. I had to manually rewrite the test harness to check for a list of expected keywords and implement a more robust self-verification step within the agent itself.

## Portfolio Reflection
**What this project says about me as an AI engineer:**
This project demonstrates my ability to move beyond simple API wrappers and build robust, production-ready AI systems. By implementing a multi-step agentic workflow, a local RAG engine, and comprehensive guardrails, I have shown that I prioritize reliability, transparency, and user safety. The inclusion of an automated test harness reflects my commitment to rigorous evaluation and my understanding that an AI system is only as good as its ability to handle edge cases and fail gracefully.
