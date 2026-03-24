"""
Paper from Microsoft: https://arxiv.org/abs/2502.10855
Blog: https://www.microsoft.com/en-us/research/blog/claimify-extracting-high-quality-claims-from-language-model-outputs/

NOTE: I have added some extra steps like fallback on regex, decomposition and some leniency in disambiguation stage parsing with some very minute things which will not be as critical to the working
NOTE: I tested it using GPT-5 API. Looked good but the only thing is that the temperature they mention 0 and 0.2 can't be achieved as it gave error

"""

import nltk
from typing import List, Dict, Optional
import re
import logging
from ..src.prompts import SELECTION_STAGE_SYSTEM_PROMPT, DISAMBIGUATION_STAGE_SYSTEM_PROMPT, DECOMPOSITION_STAGE_SYSTEM_PROMPT


logger = logging.getLogger("claimify")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_handler)

logger.setLevel(logging.DEBUG)

def _preview(value, max_len: int = 200) -> str:
    """Safe, concise single-line preview for debug logging."""
    try:
        s = "" if value is None else str(value)
    except Exception:
        s = repr(value)
    s = s.replace("\n", " ").strip()
    return s if len(s) <= max_len else s[:max_len] + "..."

class Claimify:
    def __init__(self, llm_function):
        """
        Initialize Claimify with an LLM function
        
        Args:
            llm_function: Function that takes (prompt, temperature) and returns response
        """
        self.llm = llm_function
        
        # Hyperparameters from Appendix D
        self.max_retries = 2
        self.max_preceding_sentences = 5
        self.max_following_sentences_selection = 5
        self.max_following_sentences_disambiguation = 0
        self.max_following_sentences_decomposition = 0
        
        # Stage-specific hyperparameters
        self.selection_completions = 3
        self.selection_min_successes = 2
        self.disambiguation_completions = 3
        self.disambiguation_min_successes = 2
        self.decomposition_completions = 1
        self.decomposition_min_successes = 1
    
    def extract_claims(self, question: str, answer: str) -> List[str]:
        """
        Main pipeline to extract claims from question-answer pair

        Returns:
            List of extracted factual claims
        """
        logger.info(
            "extract_claims: start | question_len=%d | answer_len=%d",
            len(question) if question is not None else 0,
            len(answer) if answer is not None else 0,
        )
        logger.debug(
            "extract_claims: inputs | question_preview='%s' | answer_preview='%s'",
            _preview(question),
            _preview(answer),
        )

        # Input validation warnings
        if not question or not str(question).strip():
            logger.warning("extract_claims: empty_or_blank_question")
        if not answer or not str(answer).strip():
            logger.warning("extract_claims: empty_or_blank_answer")

        # Step 1: Sentence Splitting and Context Creation (Section 3.1)
        sentences = self._split_sentences(answer or "")
        logger.info("extract_claims: sentence_split_complete | sentences=%d", len(sentences))
        if len(sentences) == 0:
            logger.warning("extract_claims: no_sentences_after_split")

        extracted_claims: List[str] = []

        for i, sentence in enumerate(sentences):
            logger.info("sentence[%d/%d]: selection_stage_begin", i + 1, len(sentences))
            try:
                # Step 2: Selection (Section 3.2)
                selection_result = self._selection_stage(sentence, sentences, i, question)
            except Exception:
                logger.error("sentence[%d]: selection_stage_exception", i + 1, exc_info=True)
                continue

            logger.info(
                "sentence[%d]: selection_stage_end | status=%s",
                i + 1,
                selection_result.get("status"),
            )
            if selection_result.get("status") == "no_verifiable_claims":
                continue

            selected_sentence = selection_result.get("sentence", "")

            logger.info("sentence[%d]: disambiguation_stage_begin", i + 1)
            try:
                # Step 3: Disambiguation (Section 3.3)
                disambiguation_result = self._disambiguation_stage(selected_sentence, sentences, i, question)
            except Exception:
                logger.error("sentence[%d]: disambiguation_stage_exception", i + 1, exc_info=True)
                continue

            logger.info(
                "sentence[%d]: disambiguation_stage_end | status=%s",
                i + 1,
                disambiguation_result.get("status"),
            )
            if disambiguation_result.get("status") == "cannot_disambiguate":
                continue

            disambiguated_sentence = disambiguation_result.get("sentence", "")

            logger.info("sentence[%d]: decomposition_stage_begin", i + 1)
            try:
                # Step 4: Decomposition (Section 3.4)
                claims = self._decomposition_stage(disambiguated_sentence, sentences, i, question)
            except Exception:
                logger.error("sentence[%d]: decomposition_stage_exception", i + 1, exc_info=True)
                claims = []

            if not claims:  # Labeled "No verifiable claims"
                logger.info("sentence[%d]: decomposition_stage_no_claims", i + 1)
                continue

            logger.info("sentence[%d]: decomposition_stage_claims | count=%d", i + 1, len(claims))
            extracted_claims.extend(claims)

        logger.info("extract_claims: complete | total_claims=%d", len(extracted_claims))
        logger.debug(
            "extract_claims: outputs | claims_count=%d | first_claim_preview='%s'",
            len(extracted_claims),
            _preview(extracted_claims[0]) if extracted_claims else "",
        )
        return extracted_claims
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK's sentence tokenizer (Section 3.1). Per Appendix C.1, first split answers into paragraphs by newlines, then tokenize each paragraph.
        Note: Paper reports NLTK version 3.9.1 for sentence tokenization."""
        paragraphs = [p.strip() for p in (text or "").split('\n')]
        logger.debug(
            "_split_sentences: input | text_len=%d | preview='%s'",
            len(text or ""),
            _preview(text),
        )
        sentences: List[str] = []

        for p in paragraphs:
            if not p:
                continue
            try:
                sentences.extend(nltk.sent_tokenize(p))
            except LookupError:
                # Attempt to download tokenizer resources on first failure
                logger.info("NLTK 'punkt' tokenizer not found. Attempting download...")
                try:
                    nltk.download("punkt", quiet=True)
                except Exception:
                    logger.error("Download of NLTK resource 'punkt' failed.", exc_info=True)
                # Newer NLTK versions may use 'punkt_tab' as well
                try:
                    nltk.download("punkt_tab", quiet=True)
                except Exception:
                    # Best-effort; ignore if unavailable
                    pass
                try:
                    sentences.extend(nltk.sent_tokenize(p))
                except Exception:
                    logger.error("NLTK sentence tokenization failed after resource download. Falling back to regex.", exc_info=True)
                    sentences.extend([s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()])
            except Exception:
                logger.error("Unexpected error during sentence tokenization. Falling back to regex.", exc_info=True)
                sentences.extend([s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()])

        logger.debug(
            "_split_sentences: outputs | sentences_count=%d | first_sentence_preview='%s'",
            len(sentences),
            _preview(sentences[0]) if sentences else "",
        )
        return sentences
    
    def _create_context(self, sentences: List[str], current_idx: int, stage: str) -> str:
        """Create context based on stage-specific requirements (Section 3.1 & Appendix E)"""
        start_idx = max(0, current_idx - self.max_preceding_sentences)
        logger.debug(
            "_create_context: input | stage=%s | idx=%d | total_sentences=%d",
            stage,
            current_idx,
            len(sentences),
        )
        
        if stage == "selection":
            end_idx = min(len(sentences), current_idx + self.max_following_sentences_selection + 1)
        elif stage == "disambiguation":
            end_idx = min(len(sentences), current_idx + self.max_following_sentences_disambiguation + 1)
        elif stage == "decomposition":
            end_idx = min(len(sentences), current_idx + self.max_following_sentences_decomposition + 1)
        else:
            logger.warning("create_context: unknown stage '%s'; defaulting to 'selection'", stage)
            end_idx = min(len(sentences), current_idx + self.max_following_sentences_selection + 1)

        # Create excerpt with [...] markers if not seeing all sentences
        excerpt_sentences = []
        if start_idx > 0:
            excerpt_sentences.append("[...]")
        
        excerpt_sentences.extend(sentences[start_idx:end_idx])
        
        if end_idx < len(sentences):
            excerpt_sentences.append("[...]")
        
        excerpt = " ".join(excerpt_sentences)
        logger.debug(
            "_create_context: output | context_chars=%d | preview='%s'",
            len(excerpt),
            _preview(excerpt),
        )
        return excerpt
    
    def _selection_stage(self, sentence: str, sentences: List[str], current_idx: int, question: str) -> Dict:
        """
        Selection stage implementation following Section 3.2 and Appendix N.1.1
        """
        context = self._create_context(sentences, current_idx, "selection")
        
        # Exact prompt from Appendix N.1.1 (including the contextual guidance and examples)
       
        user_prompt = f"""Question:
{question}
Excerpt:
{context}
Sentence:
{sentence}"""
        logger.debug(
            "_selection_stage: input | idx=%d | q_preview='%s' | sent_preview='%s' | ctx_chars=%d",
            current_idx + 1,
            _preview(question),
            _preview(sentence),
            len(context),
        )

        # Multiple completions with voting (Appendix D)
        for attempt in range(self.max_retries + 1):
            try:
                completions = []
                temperature = 0.2 if self.selection_completions > 1 else 0
                
                for _ in range(self.selection_completions):
                    response = self.llm(SELECTION_STAGE_SYSTEM_PROMPT+ "\n\n" + user_prompt, temperature)
                    completions.append(response)
                
                # Parse responses and count successes
                parsed_completions = []
                for completion in completions:
                    parsed = self._parse_selection_response(completion, sentence)
                    if parsed["valid"]:
                        parsed_completions.append(parsed)

                # Count successful (verifiable) outputs per Appendix D
                verifiable_count = sum(1 for p in parsed_completions if p["contains_verifiable"])
                if verifiable_count >= self.selection_min_successes:
                    # Return the first valid verifiable response
                    for parsed in parsed_completions:
                        if parsed["contains_verifiable"]:
                            logger.debug(
                                "_selection_stage: output | status=%s | sentence_preview='%s'",
                                "contains_verifiable",
                                _preview(parsed["sentence"]),
                            )
                            return {
                                "status": "contains_verifiable",
                                "sentence": parsed["sentence"]
                            }
                # Otherwise, allow retry attempts; failure will be returned after exhausting retries

            except Exception as e:
                if attempt == self.max_retries:
                    logger.warning("selection_stage: all attempts failed for sentence index %d; returning no_verifiable_claims (%s)", current_idx + 1, type(e).__name__)
                    return {"status": "no_verifiable_claims", "sentence": ""}
                else:
                    logger.warning("selection_stage: attempt %d/%d failed for sentence index %d (%s)", attempt + 1, self.max_retries + 1, current_idx + 1, type(e).__name__)
                    continue
        
        logger.debug(
            "_selection_stage: output | status=%s | sentence_preview='%s'",
            "no_verifiable_claims",
            "",
        )
        return {"status": "no_verifiable_claims", "sentence": ""}
    
    def _parse_selection_response(self, response: str, original_sentence: str) -> Dict:
        """Parse selection stage response"""
        try:
            logger.debug(
                "_parse_selection_response: input | response_len=%d | preview='%s'",
                len(response or ""),
                _preview(response),
            )
            lines = response.strip().split('\n')
            final_submission = ""
            sentence_info = ""
            
            for i, line in enumerate(lines):
                if line.strip().startswith("Final submission:"):
                    if i + 1 < len(lines):
                        final_submission = lines[i + 1].strip()
                elif line.strip().startswith("Sentence with only verifiable information:"):
                    if i + 1 < len(lines):
                        sentence_info = lines[i + 1].strip()
            
            contains_verifiable = "contains a specific and verifiable proposition" in final_submission.lower()
            
            if contains_verifiable and sentence_info and sentence_info.lower() != "none":
                if sentence_info.lower() == "remains unchanged":
                    logger.debug(
                        "_parse_selection_response: output | valid=%s | contains_verifiable=%s | sentence_preview='%s'",
                        True,
                        True,
                        _preview(original_sentence),
                    )
                    return {
                        "valid": True,
                        "contains_verifiable": True,
                        "sentence": original_sentence
                    }
                else:
                    logger.debug(
                        "_parse_selection_response: output | valid=%s | contains_verifiable=%s | sentence_preview='%s'",
                        True,
                        True,
                        _preview(sentence_info),
                    )
                    return {
                        "valid": True,
                        "contains_verifiable": True,
                        "sentence": sentence_info
                    }
            
            logger.debug(
                "_parse_selection_response: output | valid=%s | contains_verifiable=%s | sentence_preview='%s'",
                True,
                False,
                "",
            )
            return {
                "valid": True,
                "contains_verifiable": False,
                "sentence": ""
            }
            
        except Exception:
            logger.debug(
                "_parse_selection_response: output | valid=%s | contains_verifiable=%s | sentence_preview='%s'",
                False,
                False,
                "",
            )
            return {"valid": False, "contains_verifiable": False, "sentence": ""}
    
    def _disambiguation_stage(self, sentence: str, sentences: List[str], current_idx: int, question: str) -> Dict:
        """
        Disambiguation stage implementation following Section 3.3 and Appendix N.1.2
        """
        context = self._create_context(sentences, current_idx, "disambiguation")
        
        # Exact prompt structure from Appendix N.1.2
       
        user_prompt = f"""Question:
{question}
Excerpt:
{context}
Sentence:
{sentence}"""
        logger.debug(
            "_disambiguation_stage: input | idx=%d | q_preview='%s' | sent_preview='%s' | ctx_chars=%d",
            current_idx + 1,
            _preview(question),
            _preview(sentence),
            len(context),
        )

        # Multiple completions with voting
        for attempt in range(self.max_retries + 1):
            try:
                completions = []
                temperature = 0.2 if self.disambiguation_completions > 1 else 0
                
                for _ in range(self.disambiguation_completions):
                    response = self.llm(DISAMBIGUATION_STAGE_SYSTEM_PROMPT + "\n\n" + user_prompt, temperature)
                    completions.append(response)
                
                # Parse responses
                parsed_completions = []
                for completion in completions:
                    parsed = self._parse_disambiguation_response(completion)
                    if parsed["valid"]:
                        parsed_completions.append(parsed)
                
                # Count successful disambiguations per Appendix D
                disambiguated_count = sum(1 for p in parsed_completions if p["can_disambiguate"])
                if disambiguated_count >= self.disambiguation_min_successes:
                    # Return first successful disambiguation
                    for parsed in parsed_completions:
                        if parsed["can_disambiguate"]:
                            logger.debug(
                                "_disambiguation_stage: output | status=%s | sentence_preview='%s'",
                                "disambiguated",
                                _preview(parsed["sentence"]),
                            )
                            return {
                                "status": "disambiguated",
                                "sentence": parsed["sentence"]
                            }
                # Otherwise, allow retry attempts; failure will be returned after exhausting retries
            
            except Exception as e:
                if attempt == self.max_retries:
                    logger.warning("disambiguation_stage: all attempts failed for sentence index %d; returning cannot_disambiguate (%s)", current_idx + 1, type(e).__name__)
                    return {"status": "cannot_disambiguate", "sentence": ""}
                else:
                    logger.warning("disambiguation_stage: attempt %d/%d failed for sentence index %d (%s)", attempt + 1, self.max_retries + 1, current_idx + 1, type(e).__name__)
                    continue
        
        logger.debug(
            "_disambiguation_stage: output | status=%s | sentence_preview='%s'",
            "cannot_disambiguate",
            "",
        )
        return {"status": "cannot_disambiguate", "sentence": ""}
    
    def _parse_disambiguation_response(self, response: str) -> Dict:
        """Parse disambiguation stage response"""
        try:
            logger.debug(
                "_parse_disambiguation_response: input | response_len=%d | preview='%s'",
                len(response or ""),
                _preview(response),
            )
            if "Cannot be decontextualized" in response:
                logger.debug(
                    "_parse_disambiguation_response: output | valid=%s | can_disambiguate=%s | sentence_preview='%s'",
                    True,
                    False,
                    "",
                )
                return {
                    "valid": True,
                    "can_disambiguate": False,
                    "sentence": ""
                }
            
            # Look for DecontextualizedSentence
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("DecontextualizedSentence:") and "Cannot be decontextualized" not in line:
                    if i + 1 < len(lines):
                        sentence = lines[i + 1].strip()
                        logger.debug(
                            "_parse_disambiguation_response: output | valid=%s | can_disambiguate=%s | sentence_preview='%s'",
                            True,
                            True,
                            _preview(sentence),
                        )
                        return {
                            "valid": True,
                            "can_disambiguate": True,
                            "sentence": sentence
                        }
                    else:
                        # Sentence might be on same line
                        sentence = line.replace("DecontextualizedSentence:", "").strip()
                        if sentence:
                            logger.debug(
                                "_parse_disambiguation_response: output | valid=%s | can_disambiguate=%s | sentence_preview='%s'",
                                True,
                                True,
                                _preview(sentence),
                            )
                            return {
                                "valid": True,
                                "can_disambiguate": True,
                                "sentence": sentence
                            }
            
            logger.debug(
                "_parse_disambiguation_response: output | valid=%s | can_disambiguate=%s | sentence_preview='%s'",
                False,
                False,
                "",
            )
            return {"valid": False, "can_disambiguate": False, "sentence": ""}
            
        except Exception:
            logger.debug(
                "_parse_disambiguation_response: output | valid=%s | can_disambiguate=%s | sentence_preview='%s'",
                False,
                False,
                "",
            )
            return {"valid": False, "can_disambiguate": False, "sentence": ""}
    
    def _decomposition_stage(self, sentence: str, sentences: List[str], current_idx: int, question: str) -> List[str]:
        """
        Decomposition stage implementation following Section 3.4 and Appendix N.1.3
        """
        context = self._create_context(sentences, current_idx, "decomposition")
        
        # Exact prompt structure from Appendix N.1.3
       

        user_prompt = f"""Question:
{question}
Excerpt:
{context}
Sentence:
{sentence}"""
        logger.debug(
            "_decomposition_stage: input | idx=%d | q_preview='%s' | sent_preview='%s' | ctx_chars=%d",
            current_idx + 1,
            _preview(question),
            _preview(sentence),
            len(context),
        )

        # Single completion for decomposition (Appendix D)
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm(DECOMPOSITION_STAGE_SYSTEM_PROMPT + "\n\n" + user_prompt, temperature=0)
                claims = self._parse_decomposition_response(response)
                
                if claims is not None:  # Valid response
                    logger.debug(
                        "_decomposition_stage: output | claims_count=%d | first_claim_preview='%s'",
                        len(claims),
                        _preview(claims[0]) if claims else "",
                    )
                    return claims
                    
            except Exception as e:
                if attempt == self.max_retries:
                    logger.warning("decomposition_stage: all attempts failed for sentence index %d; returning no claims (%s)", current_idx + 1, type(e).__name__)
                    return []  # No verifiable claims
                else:
                    logger.warning("decomposition_stage: attempt %d/%d failed for sentence index %d (%s)", attempt + 1, self.max_retries + 1, current_idx + 1, type(e).__name__)
                    continue
        
        logger.debug(
            "_decomposition_stage: output | claims_count=%d | first_claim_preview='%s'",
            0,
            "",
        )
        return []
    
    def _parse_decomposition_response(self, response: str) -> Optional[List[str]]:
        """Parse decomposition stage response with robust header and list parsing."""
        try:
            logger.debug(
                "_parse_decomposition_response: input | response_len=%d | preview='%s'",
                len(response or ""),
                _preview(response),
            )
            lines = response.split('\n')

            def header_is_final(line: str) -> bool:
                # Handles variants like: "Specific , Verifiable , and Decontextualized Propositions with Essential Context/ Clarifications:"
                return re.search(
                    r'specific\s*,?\s*verifiable\s*,?\s*and\s*decontextualized\s*propositions.*essential\s*context.*clarifications\s*:',
                    line,
                    re.IGNORECASE,
                ) is not None

            def header_is_regular(line: str) -> bool:
                # Handles variants like: "Specific , Verifiable , and Decontextualized Propositions:"
                return re.search(
                    r'specific\s*,?\s*verifiable\s*,?\s*and\s*decontextualized\s*propositions\s*:',
                    line,
                    re.IGNORECASE,
                ) is not None

            def extract_quoted_claims_from_line(line: str) -> List[str]:
                # Extract any quoted strings on the line and clean trailing "- true or false?" (case/space tolerant)
                quoted = re.findall(r'"([^"]+)"', line)
                cleaned: List[str] = []
                for q in quoted:
                    q2 = re.sub(r'\s*-\s*true\s+or\s+false\?\s*$', '', q, flags=re.IGNORECASE).strip()
                    cleaned.append(q2)
                return cleaned

            # Pass 1: Prefer the "with Essential Context/Clarifications" section, if present
            in_section = False
            section_claims: List[str] = []
            for line in lines:
                if not in_section and header_is_final(line):
                    in_section = True
                    continue
                if in_section:
                    if line.strip().startswith(']'):
                        break
                    section_claims.extend(extract_quoted_claims_from_line(line))
            if section_claims:
                logger.debug(
                    "_parse_decomposition_response: output | claims_count=%d | first_claim_preview='%s'",
                    len(section_claims),
                    _preview(section_claims[0]) if section_claims else "",
                )
                return section_claims

            # Pass 2: Fallback to the regular propositions section
            in_section = False
            section_claims = []
            for line in lines:
                if not in_section and header_is_regular(line):
                    in_section = True
                    continue
                if in_section:
                    if line.strip().startswith(']'):
                        break
                    section_claims.extend(extract_quoted_claims_from_line(line))

            logger.debug(
                "_parse_decomposition_response: output | claims_count=%d | first_claim_preview='%s'",
                len(section_claims),
                _preview(section_claims[0]) if section_claims else "",
            )
            return section_claims if section_claims else []
        except Exception:
            logger.error("Failed to parse decomposition response.", exc_info=True)
            return None


def example_usage():
    from src.claimify import Claimify

    def run_my_llm(text, temperature): return "Use any llm but make sure you have temerature set. I tested with GPt-5 API (Temp=1) s it gives errors on others"
    
    claimify = Claimify(llm_function=run_my_llm)

    question = "What is the population of Delhi?"
    answer = "The largest city in India is the captial itself accomodating more than a Billion people over an area of 5000Sq Kms. It has more than 5 Lakhs people migrating each month from all over the world"

    claims = claimify.extract_claims(question, answer)

    print("Extracted Claims:")
    for i, claim in enumerate(claims, 1):
        print(f"{i}. {claim}")

if __name__ == "__main__":
    pass
# TODO: Definitely need to parallelize the calls for sure