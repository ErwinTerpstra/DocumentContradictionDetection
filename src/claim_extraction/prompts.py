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


REPAIR_ONE_LINER_PROMPT_TEMPLATE = """Split the text below into separate claims. Each sentence/claim is currently on the same line, but they MUST be separated into individual lines. 

Follow these rules when splitting:

Rules:
1) Output plain text only.
2) Write exactly one claim per line, and a line end after each claim. Do not place two claims on the same line.
3) Do not add, remove, or rewrite facts.
4) Do not use bullets, numbering, or intro text.
5) Do not output blank lines.

Text:
{text}
"""


TEMPORAL_NORMALIZATION_ONLY_PROMPT_TEMPLATE = """You are an expert logical analyst.
You are given claims that are already extracted. Do NOT extract new claims.
Do NOT split or merge claims.
Apply ONLY temporal normalization rules 10 through 13 below.

Input assumptions:
- Each line is an existing standalone claim.
- Keep the same number of claims in the output as in the input.

Rules (apply strictly):
10) TEMPORAL NORMALIZATION (MANDATORY):
Unless the claim explicitly states a universal or timeless fact, prefix the claim with:
"At some point in time, ..."

This signals that events may occur at different moments and should not be assumed to happen simultaneously.

11) EXCEPTIONS FOR ABSOLUTE STATEMENTS:
Do NOT use "At some point in time" for claims that express:
- absolute negation (e.g., "never", "cannot", "no longer")
- universal truths (e.g., "always", "all", "every")

Keep these claims as-is, since they imply strong logical constraints.

12) PRESERVE EVENT DISTINCTNESS:
Do NOT attempt to merge, compare, or resolve relationships between claims.
Even if multiple claims involve the same entity, treat them as independent events that may occur at different times.

13) NO TEMPORAL INTERPRETATION:
Do NOT infer or introduce specific ordering (e.g., "before", "after") unless it is explicitly stated in the claim.
Only apply the generic "At some point in time" normalization.

Output format rules:
1) Output plain text only.
2) Write exactly one claim per line.
3) Keep the original claim order.
4) Do not add introductions, explanations, bullets, or numbering.
5) Do not output blank lines.

Claims:
{text}
"""
