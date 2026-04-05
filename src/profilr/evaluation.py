import mlflow
from mlflow.genai.scorers import Guidelines


professional_tone_guideline = Guidelines(
    name="professional_tone",
    guidelines=[
        "The response must use a polite and professional tone throughout",
        "The response must avoid any dismissive, speculative or rude language",
        "The response must refer to the subject by name, not as 'the individual' or 'the person'",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

factual_guideline = Guidelines(
    name="factual_content",
    guidelines=[
        "The response must include a short biographical summary of the person",
        "The response must mention at least one specific project, tool, paper title, or technical contribution by name",
        "The response must focus on the person's professional or technical work",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)


@mlflow.genai.scorer
def has_facts_section(outputs: list) -> bool:
    """Check that the output contains an interesting facts section."""
    text = _extract_text(outputs)
    return "fact" in text.lower() or "- " in text


@mlflow.genai.scorer
def summary_not_empty(outputs: list) -> bool:
    """Check that the output is not empty and has meaningful content."""
    text = _extract_text(outputs)
    return len(text.split()) > 10


@mlflow.genai.scorer
def under_300_words(outputs: list) -> bool:
    """Check that the output is concise (under 300 words)."""
    text = _extract_text(outputs)
    return len(text.split()) < 300


def _extract_text(outputs: list) -> str:
    """Extract plain text from scorer outputs regardless of format."""
    if isinstance(outputs, list) and outputs:
        item = outputs[0]
        if isinstance(item, dict):
            return item.get("text", str(item))
        return str(item)
    return str(outputs)
