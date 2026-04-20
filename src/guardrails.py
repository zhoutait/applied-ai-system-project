"""
Guardrails Module — StudyMind AI.

Provides input validation, output safety checks, and rate-limiting
to ensure the system behaves responsibly and reliably.
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_QUERY_LENGTH = 1000
MIN_QUERY_LENGTH = 3

# Patterns that indicate potentially harmful or off-topic requests
BLOCKED_PATTERNS = [
    r"\b(hack|exploit|bypass|jailbreak|ignore previous|ignore all instructions)\b",
    r"\b(generate malware|write virus|create exploit)\b",
    r"\b(personal information|social security|credit card|password)\b",
]

# Patterns that suggest the query is off-topic for a study assistant
OFF_TOPIC_PATTERNS = [
    r"\b(buy|sell|purchase|order|shop|price|discount)\b",
    r"\b(weather|sports score|stock price|lottery)\b",
    r"\b(write my essay for me|do my homework for me|complete my assignment)\b",
]

# Low-confidence warning threshold
CONFIDENCE_WARNING_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Guardrails class
# ---------------------------------------------------------------------------
class Guardrails:
    """
    Validates inputs and outputs to ensure safe, on-topic interactions.

    Methods
    -------
    validate_input(query) -> (is_valid, message)
        Check whether the user's query is acceptable.
    check_output_confidence(confidence) -> (is_acceptable, message)
        Warn if the system's confidence is below threshold.
    sanitize_query(query) -> str
        Clean and normalise the query string.
    """

    def validate_input(self, query: str) -> Tuple[bool, str]:
        """
        Validate a user query before processing.

        Parameters
        ----------
        query : str
            Raw user input.

        Returns
        -------
        tuple
            (is_valid: bool, message: str)
            If invalid, message explains why.
        """
        if not isinstance(query, str):
            return False, "Input must be a text string."

        stripped = query.strip()

        if len(stripped) < MIN_QUERY_LENGTH:
            return False, f"Query is too short. Please enter at least {MIN_QUERY_LENGTH} characters."

        if len(stripped) > MAX_QUERY_LENGTH:
            return False, (
                f"Query is too long ({len(stripped)} characters). "
                f"Please limit your query to {MAX_QUERY_LENGTH} characters."
            )

        # Check for blocked patterns
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
                logger.warning("Blocked query matched pattern '%s': %s", pattern, stripped[:80])
                return False, (
                    "This query has been flagged as potentially harmful or outside the scope "
                    "of a study assistant. Please rephrase your question."
                )

        # Warn about off-topic queries (soft block — log but allow)
        for pattern in OFF_TOPIC_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
                logger.info("Off-topic query detected: %s", stripped[:80])
                return True, (
                    "Note: This query may be outside the scope of a study assistant. "
                    "I'll do my best to help, but my knowledge base focuses on academic topics."
                )

        return True, ""

    def check_output_confidence(self, confidence: float) -> Tuple[bool, str]:
        """
        Check whether the system's confidence meets the minimum threshold.

        Parameters
        ----------
        confidence : float
            Overall confidence score (0.0 – 1.0).

        Returns
        -------
        tuple
            (is_acceptable: bool, warning_message: str)
        """
        if confidence < CONFIDENCE_WARNING_THRESHOLD:
            msg = (
                f"Low confidence warning: The system's confidence in this response is "
                f"{confidence:.0%}. The answer may not be fully grounded in your study notes. "
                f"Please verify with additional sources."
            )
            logger.warning("Low confidence response: %.2f", confidence)
            return False, msg
        return True, ""

    @staticmethod
    def sanitize_query(query: str) -> str:
        """
        Sanitise and normalise a query string.

        - Strips leading/trailing whitespace.
        - Collapses multiple spaces.
        - Removes non-printable characters.
        """
        # Remove non-printable characters
        cleaned = re.sub(r"[^\x20-\x7E\n\t]", "", query)
        # Collapse multiple whitespace
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        # Normalise newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
