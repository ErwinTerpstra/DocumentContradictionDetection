DIRECT_CLAIM_PROMPT_TEMPLATE = """You are an expert logical analyst.
Extract every individual, checkable claim from the text below as a standalone atomic statement.

Content rules:
1) ATOMICITY: Each claim must express exactly one fact. Break complex sentences into multiple simple ones.
2) DECONTEXTUALIZATION: Every claim must be understandable without the rest of the text.
   - Replace all pronouns (he, she, they, it) with the specific names/entities they refer to.
   - Example: 'Mrs. Frisby saves Jeremy' instead of 'She saves him'.
3) QUANTIFIERS: Explicitly include quantities, frequencies, and limits (e.g., 'exactly two', 'all', 'never', 'only').
4) MODALITY: Capture if something is a fact, a goal, a belief, or a requirement.
   - Example: 'The rats intend to move to a farm' (Goal) vs 'The rats move to a farm' (Fact).
5) VERACITY: Do not interpret; stay 100% faithful to the text's literal meaning.

Output format rules (strictly enforced):
6) ONE CLAIM PER LINE: Every single claim must be on its own separate line. Never place two claims on the same line, not even separated by a period.
7) NO INTRO OR OUTRO: Do not write any introductory or closing sentence (e.g. do not start with "Here are the claims:").
8) NO BULLETS OR NUMBERS: Do not use bullet points (-, *) or numbering (1., 2.) — plain text only.
9) NO BLANK LINES: Do not insert empty lines between claims.

Text:
{text}
"""
