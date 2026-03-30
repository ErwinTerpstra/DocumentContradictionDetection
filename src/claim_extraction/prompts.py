DIRECT_CLAIM_PROMPT_TEMPLATE = """You are an expert logical analyst. 
Extract every individual, checkable claim from the text below as a standalone atomic statement.

Rules:
1) ATOMICITY: One statement per line. Break complex sentences into multiple simple ones.
2) DECONTEXTUALIZATION: Every claim must be understandable without the rest of the text. 
   - Replace all pronouns (he, she, they, it) with the specific names/entities they refer to.
   - Example: 'Mrs. Frisby saves Jeremy' instead of 'She saves him'.
3) QUANTIFIERS: Explicitly include quantities, frequencies, and limits (e.g., 'exactly two', 'all', 'never', 'only').
4) MODALITY: Capture if something is a fact, a goal, a belief, or a requirement.
   - Example: 'The rats intend to move to a farm' (Goal) vs 'The rats move to a farm' (Fact).
5) NO NOISE: No intro, no bullets (- or *), no numbering, no explanations. 
6) VERACITY: Do not interpret; stay 100% faithful to the text's literal meaning.

Text:
{text}
"""
