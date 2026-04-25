"""
Agentic Workflow Module — StudyMind AI.

Implements a multi-step reasoning agent that can:
  1. Decompose complex study tasks into sub-goals.
  2. Execute each sub-goal using RAG-grounded generation.
  3. Synthesize intermediate results into a final answer.
  4. Log all observable intermediate steps for transparency.

The agent supports the following tools / actions:
  - retrieve_context   : Fetch relevant chunks from the knowledge base.
  - generate_answer    : Generate a grounded answer from context.
  - generate_quiz      : Create quiz questions from retrieved material.
  - summarize          : Produce a concise summary of retrieved content.
  - explain_concept    : Provide a detailed explanation of a concept.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from openai import OpenAI

from src.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Few-shot examples for academic tone specialisation
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """
Example 1:
User: What is overfitting?
Assistant: Overfitting occurs when a machine learning model learns the training data too precisely, capturing noise and random fluctuations rather than the underlying pattern. As a result, the model performs well on training data but generalises poorly to unseen examples. Regularisation techniques such as L2 penalty, dropout, and early stopping are commonly employed to mitigate this phenomenon.

Example 2:
User: Explain the attention mechanism.
Assistant: The attention mechanism allows a neural network to dynamically focus on different parts of the input sequence when producing each element of the output. Rather than compressing the entire input into a single fixed-length vector, attention computes a weighted sum of all encoder hidden states, where the weights reflect the relevance of each input position to the current decoding step. This design substantially improves performance on long sequences and forms the foundation of the Transformer architecture.

Example 3:
User: What is RAG?
Assistant: Retrieval-Augmented Generation (RAG) is a framework that enhances language model outputs by first retrieving relevant documents or passages from an external knowledge base and then conditioning the generation on that retrieved context. This approach reduces hallucination, enables the model to access up-to-date or domain-specific information, and makes the system's reasoning more transparent by allowing citations to source material.
"""

SYSTEM_PROMPT = f"""You are StudyMind AI, an expert academic study assistant. Your role is to help students understand complex topics clearly and accurately.

Guidelines:
- Always ground your answers in the provided context when available.
- Use precise academic language while remaining accessible.
- When uncertain, explicitly state your level of confidence.
- Cite the source documents when referencing specific facts.
- Structure longer answers with clear headings or numbered steps.
- Never fabricate facts; if the context does not contain the answer, say so honestly.

{FEW_SHOT_EXAMPLES}"""


# ---------------------------------------------------------------------------
# Data classes for agentic step tracking
# ---------------------------------------------------------------------------
@dataclass
class AgentStep:
    """Represents one observable step in the agent's reasoning chain."""
    step_number: int
    action: str
    input_summary: str
    output_summary: str
    confidence: float = 0.0
    details: Dict = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Full response from the agent including reasoning trace."""
    query: str
    final_answer: str
    steps: List[AgentStep]
    overall_confidence: float
    sources_used: List[str]
    task_type: str


# ---------------------------------------------------------------------------
# StudyMindAgent
# ---------------------------------------------------------------------------
class StudyMindAgent:
    """
    Multi-step reasoning agent for StudyMind AI.

    The agent follows a Plan → Retrieve → Generate → Verify loop:
      1. Classify the task type (QA, quiz, summary, explanation).
      2. Retrieve relevant context from the RAG engine.
      3. Execute the appropriate generation tool.
      4. Perform a self-consistency check on the output.
      5. Return the final answer with a full reasoning trace.
    """

    TASK_TYPES = {
        "quiz": ["quiz", "question", "test me", "practice", "flashcard"],
        "summary": ["summarize", "summarise", "summary", "overview", "tldr", "brief"],
        "explain": ["explain", "what is", "define", "describe", "how does", "why does"],
        "qa": [],  # Default fallback
    }

    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        self.llm = OpenAI()
        logger.info("StudyMindAgent initialised.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, query: str) -> AgentResponse:
        """
        Execute the full agentic reasoning pipeline for a user query.

        Parameters
        ----------
        query : str
            The user's study question or task.

        Returns
        -------
        AgentResponse
            Structured response with answer, steps, and confidence.
        """
        steps: List[AgentStep] = []
        step_num = 1

        # Step 1: Classify task
        task_type = self._classify_task(query)
        steps.append(AgentStep(
            step_number=step_num,
            action="classify_task",
            input_summary=f"Query: '{query[:80]}...' " if len(query) > 80 else f"Query: '{query}'",
            output_summary=f"Identified task type: {task_type}",
            confidence=1.0,
            details={"task_type": task_type},
        ))
        step_num += 1
        logger.info("[Agent] Step 1 — Task classified as: %s", task_type)

        # Step 2: Retrieve context
        context_str, avg_confidence = self.rag.get_context_string(query)
        retrieved_chunks = self.rag.retrieve(query)
        sources = list({c["source"] for c in retrieved_chunks})

        steps.append(AgentStep(
            step_number=step_num,
            action="retrieve_context",
            input_summary=f"Query: '{query[:60]}'",
            output_summary=(
                f"Retrieved {len(retrieved_chunks)} chunks from {len(sources)} source(s). "
                f"Avg confidence: {avg_confidence:.2f}"
            ),
            confidence=avg_confidence,
            details={
                "chunks_retrieved": len(retrieved_chunks),
                "sources": sources,
                "avg_confidence": avg_confidence,
            },
        ))
        step_num += 1
        logger.info("[Agent] Step 2 — Retrieved %d chunks (avg conf: %.2f).", len(retrieved_chunks), avg_confidence)

        # Step 3: Generate response based on task type
        if task_type == "quiz":
            answer, gen_confidence = self._generate_quiz(query, context_str)
        elif task_type == "summary":
            answer, gen_confidence = self._generate_summary(query, context_str)
        elif task_type == "explain":
            answer, gen_confidence = self._generate_explanation(query, context_str)
        else:
            answer, gen_confidence = self._generate_answer(query, context_str)

        steps.append(AgentStep(
            step_number=step_num,
            action=f"generate_{task_type}",
            input_summary=f"Context length: {len(context_str)} chars. Task: {task_type}.",
            output_summary=f"Generated response ({len(answer)} chars). Generation confidence: {gen_confidence:.2f}",
            confidence=gen_confidence,
            details={"response_length": len(answer)},
        ))
        step_num += 1
        logger.info("[Agent] Step 3 — Generated %s response (%d chars).", task_type, len(answer))

        # Step 4: Self-consistency check
        verified, verify_confidence = self._verify_response(query, answer, context_str)
        steps.append(AgentStep(
            step_number=step_num,
            action="verify_response",
            input_summary="Check answer consistency with retrieved context.",
            output_summary=f"Verification passed: {verified}. Confidence: {verify_confidence:.2f}",
            confidence=verify_confidence,
            details={"verified": verified},
        ))
        logger.info("[Agent] Step 4 — Verification: passed=%s, confidence=%.2f.", verified, verify_confidence)

        # Compute overall confidence
        overall_confidence = round(
            (avg_confidence * 0.5 + gen_confidence * 0.3 + verify_confidence * 0.2), 4
        )

        return AgentResponse(
            query=query,
            final_answer=answer,
            steps=steps,
            overall_confidence=overall_confidence,
            sources_used=sources,
            task_type=task_type,
        )

    # ------------------------------------------------------------------
    # Task classification
    # ------------------------------------------------------------------

    def _classify_task(self, query: str) -> str:
        """Classify the query into a task type using keyword matching."""
        query_lower = query.lower()
        for task_type, keywords in self.TASK_TYPES.items():
            if any(kw in query_lower for kw in keywords):
                return task_type
        return "qa"

    # ------------------------------------------------------------------
    # Generation tools
    # ------------------------------------------------------------------

    def _generate_answer(self, query: str, context: str) -> tuple:
        """Generate a grounded Q&A response."""
        if context:
            user_msg = (
                f"Using the following study notes as context, answer the question accurately.\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {query}\n\n"
                f"Provide a thorough, well-structured answer. Cite the source documents where relevant."
            )
            confidence = 0.85
        else:
            user_msg = (
                f"Answer the following question based on your knowledge. "
                f"Note: no relevant context was found in the knowledge base.\n\n"
                f"QUESTION: {query}"
            )
            confidence = 0.50

        response = self._call_llm(user_msg)
        return response, confidence

    def _generate_quiz(self, query: str, context: str) -> tuple:
        """Generate quiz questions from retrieved material."""
        topic = query.replace("quiz", "").replace("question", "").replace("test me on", "").strip()
        if context:
            user_msg = (
                f"Based on the following study notes, generate 5 multiple-choice quiz questions "
                f"about: {topic if topic else 'the provided material'}.\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"Format each question as:\n"
                f"Q[N]: [Question text]\n"
                f"A) [Option]\nB) [Option]\nC) [Option]\nD) [Option]\n"
                f"Correct Answer: [Letter]\nExplanation: [Brief explanation]\n"
            )
            confidence = 0.80
        else:
            user_msg = (
                f"Generate 5 multiple-choice quiz questions about: {topic if topic else query}.\n"
                f"Note: no relevant context was found in the knowledge base; use general knowledge.\n"
                f"Format each question as:\n"
                f"Q[N]: [Question text]\n"
                f"A) [Option]\nB) [Option]\nC) [Option]\nD) [Option]\n"
                f"Correct Answer: [Letter]\nExplanation: [Brief explanation]\n"
            )
            confidence = 0.45

        response = self._call_llm(user_msg)
        return response, confidence

    def _generate_summary(self, query: str, context: str) -> tuple:
        """Generate a concise summary of retrieved content."""
        topic = query.replace("summarize", "").replace("summarise", "").replace("summary of", "").strip()
        if context:
            user_msg = (
                f"Based on the following study notes, provide a concise, well-structured summary "
                f"of: {topic if topic else 'the provided material'}.\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"Structure your summary with: (1) Key Definition, (2) Core Concepts, "
                f"(3) Important Details, (4) Key Takeaways."
            )
            confidence = 0.85
        else:
            user_msg = (
                f"Provide a concise academic summary of: {topic if topic else query}.\n"
                f"Note: no relevant context was found in the knowledge base.\n"
                f"Structure your summary with: (1) Key Definition, (2) Core Concepts, "
                f"(3) Important Details, (4) Key Takeaways."
            )
            confidence = 0.50

        response = self._call_llm(user_msg)
        return response, confidence

    def _generate_explanation(self, query: str, context: str) -> tuple:
        """Generate a detailed concept explanation."""
        if context:
            user_msg = (
                f"Using the following study notes as context, provide a clear and detailed "
                f"explanation for: {query}\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"Your explanation should: (1) Define the concept clearly, "
                f"(2) Explain how it works step by step, "
                f"(3) Provide a concrete example, "
                f"(4) Mention any important caveats or related concepts."
            )
            confidence = 0.88
        else:
            user_msg = (
                f"Provide a clear and detailed academic explanation for: {query}\n"
                f"Note: no relevant context was found in the knowledge base.\n"
                f"Your explanation should: (1) Define the concept clearly, "
                f"(2) Explain how it works step by step, "
                f"(3) Provide a concrete example, "
                f"(4) Mention any important caveats or related concepts."
            )
            confidence = 0.55

        response = self._call_llm(user_msg)
        return response, confidence

    def _verify_response(self, query: str, answer: str, context: str) -> tuple:
        """
        Self-consistency check: ask the LLM to verify the answer is grounded.
        Returns (verified: bool, confidence: float).
        """
        if not context:
            return True, 0.60  # No context to verify against

        verify_prompt = (
            f"You are a fact-checker. Given the context and the generated answer below, "
            f"determine whether the answer is consistent with and supported by the context.\n\n"
            f"CONTEXT (excerpt):\n{context[:1200]}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER: {answer[:800]}\n\n"
            f"Respond with ONLY: 'VERIFIED: [YES/NO] | CONFIDENCE: [0.0-1.0] | REASON: [one sentence]'"
        )

        try:
            result = self._call_llm(verify_prompt, max_tokens=120)
            verified = "YES" in result.upper()
            # Extract confidence from response
            conf = 0.75
            if "CONFIDENCE:" in result.upper():
                try:
                    conf_part = result.upper().split("CONFIDENCE:")[1].split("|")[0].strip()
                    conf = float(conf_part)
                except (ValueError, IndexError):
                    pass
            return verified, conf
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Verification step failed: %s", exc)
            return True, 0.60

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _call_llm(self, user_message: str, max_tokens: int = 1024) -> str:
        """Call the LLM with the system prompt and return the response text."""
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            message = str(exc)
            if "invalid_api_key" in message or "Incorrect API key" in message or "Error code: 401" in message:
                safe_message = "OpenAI authentication failed. Check that OPENAI_API_KEY is a valid key."
            elif "Connection error" in message:
                safe_message = "Could not reach the OpenAI API. Check your network connection and try again."
            else:
                safe_message = "The language model request failed. Check logs/studymind.log for details."
            logger.error("LLM call failed: %s", safe_message)
            raise RuntimeError(f"LLM generation failed: {safe_message}") from exc
