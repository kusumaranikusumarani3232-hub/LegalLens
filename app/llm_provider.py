"""Unified LLM Provider abstraction supporting Gemini, Groq, OpenAI, and smart Offline Demo."""

import json
import re
from typing import Dict, Any, Optional
from app.config import config
from app.prompts import SYSTEM_LEGAL_SAFETY_PROMPT


class LLMError(Exception):
    """Raised when LLM call fails."""
    pass


class LLMProvider:
    """Manages requests across AI providers (Gemini, Groq, OpenAI, Mock)."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = (provider or config.get_active_provider(override_provider=provider, override_key=api_key)).lower()
        self.api_key = api_key or getattr(config, f"{self.provider.upper()}_API_KEY", None)

    def generate(self, prompt: str, system_prompt: str = SYSTEM_LEGAL_SAFETY_PROMPT, temperature: float = 0.2, json_mode: bool = False, max_tokens: int = 8192) -> str:
        """Dispatches generation to configured provider."""
        if self.provider == "gemini":
            return self._call_gemini(prompt, system_prompt, temperature, max_tokens)
        elif self.provider == "groq":
            return self._call_groq(prompt, system_prompt, temperature, json_mode, max_tokens)
        elif self.provider == "openai":
            return self._call_openai(prompt, system_prompt, temperature, json_mode, max_tokens)
        else:
            return self._call_mock(prompt, system_prompt)

    def generate_json(self, prompt: str, system_prompt: str = SYSTEM_LEGAL_SAFETY_PROMPT) -> Dict[str, Any]:
        """Generates text and parses validated JSON."""
        raw_text = self.generate(prompt, system_prompt, temperature=0.1, json_mode=True, max_tokens=8192)
        return self._clean_and_parse_json(raw_text)

    def _call_gemini(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int = 8192) -> str:
        key = self.api_key or config.GEMINI_API_KEY
        if not key:
            raise LLMError("Gemini API key is not configured. Please set GEMINI_API_KEY in .env or enter it in the sidebar.")

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=key)
            model_name = config.GEMINI_MODEL

            full_prompt = f"{system_prompt}\n\n{prompt}"
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text or ""
        except Exception as e:
            # Check for legacy google.generativeai fallback if needed
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=key)
                model = legacy_genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=system_prompt
                )
                res = model.generate_content(prompt)
                return res.text or ""
            except Exception:
                raise LLMError(f"Gemini API request failed: {str(e)}")

    def _call_groq(self, prompt: str, system_prompt: str, temperature: float, json_mode: bool = False, max_tokens: int = 8192) -> str:
        key = self.api_key or config.GROQ_API_KEY
        if not key:
            raise LLMError("Groq API key is not configured. Please set GROQ_API_KEY in .env or enter it in the sidebar.")

        try:
            from groq import Groq
            client = Groq(api_key=key)
            model_name = config.GROQ_MODEL

            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                response = client.chat.completions.create(**kwargs)
            except Exception:
                # If model or API rejected response_format, retry without it
                if json_mode and "response_format" in kwargs:
                    del kwargs["response_format"]
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(f"Groq API request failed: {str(e)}")

    def _call_openai(self, prompt: str, system_prompt: str, temperature: float, json_mode: bool = False, max_tokens: int = 8192) -> str:
        key = self.api_key or config.OPENAI_API_KEY
        if not key:
            raise LLMError("OpenAI API key is not configured. Please set OPENAI_API_KEY in .env or enter it in the sidebar.")

        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            model_name = config.OPENAI_MODEL

            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                response = client.chat.completions.create(**kwargs)
            except Exception:
                if json_mode and "response_format" in kwargs:
                    del kwargs["response_format"]
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(f"OpenAI API request failed: {str(e)}")

    def _call_mock(self, prompt: str, system_prompt: str) -> str:
        """
        Smart deterministic legal processor when running in offline/demo mode or without API keys.
        Extracts structured legal information using heuristic text inspection.
        """
        lower_prompt = prompt.lower()

        # Handle QA prompt
        if "user question:" in lower_prompt and "retrieved context" in lower_prompt:
            return self._mock_qa(prompt)

        # Handle Comparison prompt
        if "document a (original):" in lower_prompt and "document b (revised):" in lower_prompt:
            return self._mock_comparison(prompt)

        # Handle Document Analysis prompt
        return self._mock_document_analysis(prompt)

    def _repair_truncated_json(self, text: str) -> Dict[str, Any]:
        """Attempts to repair truncated or partially incomplete JSON text."""
        start = text.find('{')
        if start == -1:
            raise ValueError("No '{' found in JSON text.")
        json_body = text[start:]

        stack = []
        in_string = False
        escape = False
        cleaned_chars = []

        for char in json_body:
            if escape:
                escape = False
                cleaned_chars.append(char)
                continue
            if char == '\\' and in_string:
                escape = True
                cleaned_chars.append(char)
                continue
            if char == '"':
                in_string = not in_string
                cleaned_chars.append(char)
                continue

            if not in_string:
                if char in '{[':
                    stack.append(char)
                elif char in '}]':
                    if stack:
                        stack.pop()
            cleaned_chars.append(char)

        repaired = ''.join(cleaned_chars)
        if in_string:
            repaired += '"'

        repaired = re.sub(r',\s*$', '', repaired)
        repaired = re.sub(r',\s*"[^"]*":?\s*"?[^"]*$', '', repaired)

        stack = []
        in_string = False
        escape = False
        for char in repaired:
            if escape:
                escape = False
                continue
            if char == '\\' and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char in '{[':
                    stack.append(char)
                elif char in '}]':
                    if stack:
                        stack.pop()

        for open_char in reversed(stack):
            if open_char == '{':
                repaired += '}'
            elif open_char == '[':
                repaired += ']'

        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        return json.loads(repaired)

    def _clean_and_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Extracts JSON object from text, handling markdown fences, whitespace, and truncation."""
        text = raw_text.strip()
        # Strip markdown code blocks
        if "```" in text:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try locating first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_substr = text[start:end + 1]
            try:
                return json.loads(json_substr)
            except json.JSONDecodeError:
                # Remove trailing commas before closing braces
                fixed = re.sub(r',\s*([}\]])', r'\1', json_substr)
                try:
                    return json.loads(fixed)
                except Exception:
                    pass

        # Try repairing truncated JSON
        try:
            return self._repair_truncated_json(text)
        except Exception:
            pass

        raise LLMError(f"Failed to parse structured model response as JSON. Raw output: {text[:200]}...")

    def _mock_document_analysis(self, prompt: str) -> str:
        """Generates realistic structured legal analysis based on document keywords."""
        text = prompt.split("DOCUMENT TEXT:")[-1] if "DOCUMENT TEXT:" in prompt else prompt
        lower = text.lower()

        is_lease = any(w in lower for w in ["lease", "rent", "tenant", "landlord", "premises"])
        is_employment = any(w in lower for w in ["employee", "employment", "salary", "employer", "duties"])
        is_nda = any(w in lower for w in ["confidential", "non-disclosure", "disclosing party", "receiving party"])

        if is_lease:
            doc_type = "Residential Lease Agreement"
            parties = "Landlord: Property Owner / Tenant: Occupant (Identified in agreement)"
            effective = "Beginning of lease term stated in Section 1"
            expiration = "12 months from commencement date"
            jurisdiction = "State/Local jurisdiction specified in governing law clause"
            summary_what = "This is a residential rental lease agreement establishing occupancy rights, monthly rental rates, security deposit handling, and property maintenance obligations."
            agree_user = "Pay monthly rent on or before the due date, maintain the premises in good order, and notify the landlord prior to lease termination."
            agree_other = "Provide peaceful possession of the premises, maintain essential structural utilities, and return the security deposit in accordance with applicable notice terms."
            attention = "Review security deposit refund conditions, early termination penalties, and required notice periods."
            clauses = [
                {
                    "clause_title": "Monthly Rent & Payment Schedule",
                    "category": "Payment",
                    "what_the_document_says": "Rent is due on the 1st day of each calendar month with a grace period and late fee for delayed payments.",
                    "in_simple_words": "You must pay rent promptly at the beginning of each month to avoid incurring late fees.",
                    "why_you_may_want_to_pay_attention": "Late payment fees accumulate quickly if payments are missed.",
                    "source": "Section 2 — Rent and Fees"
                },
                {
                    "clause_title": "Security Deposit Handling",
                    "category": "Deposit",
                    "what_the_document_says": "A security deposit is held to cover damages beyond reasonable wear and tear, returnable within specified statutory days.",
                    "in_simple_words": "Your deposit covers property damage but should be refunded after you vacate if no damage occurred.",
                    "why_you_may_want_to_pay_attention": "Ensure you perform a walkthrough inspection and document existing damages to protect your refund.",
                    "source": "Section 3 — Security Deposit"
                },
                {
                    "clause_title": "Termination Notice Period",
                    "category": "Notice period",
                    "what_the_document_says": "Written notice of at least 30 to 60 days is mandatory prior to vacation or non-renewal.",
                    "in_simple_words": "You cannot simply leave when the lease ends; you must submit advance written notice.",
                    "why_you_may_want_to_pay_attention": "Failing to give proper advance notice can lead to forfeiture of deposit or rent continuation.",
                    "source": "Section 6 — Termination and Renewal"
                }
            ]
            obligations = [
                {"party": "Your obligations", "obligation": "Pay agreed monthly rent on the first of each month.", "deadline_or_timeline": "1st day of every month", "source": "Section 2.1"},
                {"party": "Your obligations", "obligation": "Provide written notice before vacating the property.", "deadline_or_timeline": "30 to 60 days before expiration", "source": "Section 6.2"},
                {"party": "Other party's obligations", "obligation": "Deliver possession of the leased premises in habitable condition.", "deadline_or_timeline": "Effective start date", "source": "Section 1.3"},
                {"party": "Other party's obligations", "obligation": "Return unused portion of the security deposit following move-out inspection.", "deadline_or_timeline": "Within 21-30 days post move-out", "source": "Section 3.4"}
            ]
            attentions = [
                {
                    "item": "Advance Notice Requirement for Termination",
                    "observation": "Agreement requires formal written notice 60 days prior to lease expiration to prevent auto-renewal or month-to-month conversion.",
                    "why_it_matters": "If you forget to provide notice, you could remain liable for additional rent cycles.",
                    "source": "Section 6 — Renewal & Notice"
                },
                {
                    "item": "Maintenance and Repair Threshold",
                    "observation": "Minor maintenance tasks under a specified monetary amount may be assigned to the occupant.",
                    "why_it_matters": "Clarify exactly what qualifies as landlord structural repair versus tenant routine maintenance.",
                    "source": "Section 5 — Maintenance"
                }
            ]
            checklist = [
                {"category": "Before Signing", "task": "Verify exact monthly rent amount and grace period", "detail": "Confirm no hidden administrative or utility surcharges exist.", "completed": False},
                {"category": "Before Signing", "task": "Inspect the premises and record photos of pre-existing wear", "detail": "Protects your security deposit upon move-out.", "completed": False},
                {"category": "Before Signing", "task": "Check notice period requirements for non-renewal", "detail": "Note the exact calendar deadline for your 30 or 60 day notice.", "completed": False},
                {"category": "After Signing / Ongoing", "task": "Set calendar reminder 60 days before lease end date", "detail": "Avoid unintended automatic renewal.", "completed": False}
            ]
            lawyer_questions = [
                {"category": "Deposit Refund", "question": "What documentation is required to ensure full return of the security deposit?", "context": "Document specifies deposit deductions for repairs; clarify inventory documentation expectations."},
                {"category": "Early Termination", "question": "What are the exact financial liabilities if I need to terminate early due to job relocation?", "context": "The lease contains early termination restrictions; determine if mitigating circumstances apply."}
            ]
        elif is_nda:
            doc_type = "Non-Disclosure Agreement (NDA)"
            parties = "Disclosing Party and Receiving Party as designated in the introductory clause"
            effective = "Date of signature or designated effective date"
            expiration = "2 to 3 years from disclosure date"
            jurisdiction = "Governing jurisdiction stated in dispute section"
            summary_what = "A bilateral or unilateral non-disclosure agreement designed to protect proprietary information, business data, and trade secrets shared between parties."
            agree_user = "Maintain strict confidentiality regarding disclosed proprietary information and refrain from unauthorized use or dissemination."
            agree_other = "Provide clear identification of confidential materials and limit access to designated personnel."
            attention = "Review definition of confidential information, carve-outs (public knowledge), and duration of non-disclosure obligations."
            clauses = [
                {"clause_title": "Confidential Information Scope", "category": "Confidentiality", "what_the_document_says": "Covers all proprietary technical, financial, and business data disclosed.", "in_simple_words": "You cannot share or misuse any private business information disclosed to you.", "why_you_may_want_to_pay_attention": "Ensure standard exclusions (information already public or independently developed) are present.", "source": "Section 1 — Definitions"},
                {"clause_title": "Duration of Obligation", "category": "Term", "what_the_document_says": "Confidentiality obligations survive termination for a specified number of years.", "in_simple_words": "You must keep the secrets safe even after the business relationship ends.", "why_you_may_want_to_pay_attention": "Indefinite terms should be avoided except for statutory trade secrets.", "source": "Section 4 — Term"}
            ]
            obligations = [
                {"party": "Your obligations", "obligation": "Use confidential data solely for the evaluated business purpose.", "deadline_or_timeline": "Continuous during and after relationship", "source": "Section 2.1"},
                {"party": "Your obligations", "obligation": "Promptly return or destroy all confidential records upon written request.", "deadline_or_timeline": "Within 14 days of request", "source": "Section 3.2"}
            ]
            attentions = [
                {"item": "Survival Period", "observation": "Confidentiality obligations persist after relationship termination.", "why_it_matters": "You will be bound to silence even if you work elsewhere.", "source": "Section 4"}
            ]
            checklist = [
                {"category": "Before Signing", "task": "Confirm standard carve-outs are included (public domain, prior knowledge)", "detail": "Prevents liability for information you already knew.", "completed": False}
            ]
            lawyer_questions = [
                {"category": "Scope", "question": "Does the definition of confidential info include reasonable standard exceptions?", "context": "Check that publicly known information is expressly excluded."}
            ]
        else:
            doc_type = "Legal Agreement"
            parties = "Parties named in agreement or 'Not found in the document.'"
            effective = "Effective date specified in document"
            expiration = "Term expiration specified in document"
            jurisdiction = "Jurisdiction specified in governing law clause"
            summary_what = "This is a binding legal contract outlining mutual covenants, payment terms, and operational obligations between the signing parties."
            agree_user = "Perform stated obligations, adhere to scheduled timelines, and comply with stipulated terms."
            agree_other = "Fulfill reciprocal covenants and provide agreed goods, services, or accommodations."
            attention = "Review termination clauses, payment terms, limitations of liability, and dispute mechanisms."
            clauses = [
                {"clause_title": "Terms and Conditions", "category": "Term", "what_the_document_says": "Agreement governs relations for the duration specified.", "in_simple_words": "Defines how long both sides must follow the contract.", "why_you_may_want_to_pay_attention": "Check renewal and termination provisions.", "source": "Section 1"}
            ]
            obligations = [
                {"party": "Your obligations", "obligation": "Comply with stated terms and deliverables.", "deadline_or_timeline": "As specified in contract", "source": "Section 2"}
            ]
            attentions = [
                {"item": "Limitation of Liability", "observation": "Review liability provisions and dispute resolution clauses.", "why_it_matters": "Affects your remedies in case of disagreement.", "source": "Section 8"}
            ]
            checklist = [
                {"category": "Before Signing", "task": "Review all dates, names, and financial commitments", "detail": "Ensure terms match your oral understanding.", "completed": False}
            ]
            lawyer_questions = [
                {"category": "Dispute Resolution", "question": "What dispute resolution process is mandated before court action?", "context": "Verify if binding arbitration is required."}
            ]

        data = {
            "metadata": {
                "document_type": doc_type,
                "approximate_length": f"~{max(1, len(text.split()) // 400)} pages, {len(text.split())} words",
                "parties_involved": parties,
                "effective_date": effective,
                "expiration_date": expiration,
                "governing_jurisdiction": jurisdiction
            },
            "summary": {
                "what_is_this": summary_what,
                "what_am_i_agreeing_to": agree_user,
                "what_does_other_party_agree_to": agree_other,
                "what_should_i_pay_attention_to": attention
            },
            "clauses": clauses,
            "obligations": obligations,
            "attention_points": attentions,
            "checklist": checklist,
            "lawyer_questions": lawyer_questions
        }
        return json.dumps(data)

    def _mock_qa(self, prompt: str) -> str:
        """Grounds Q&A directly in retrieved text or declares not found."""
        match_q = re.search(r'USER QUESTION:\s*(.*?)(?:\n|$)', prompt, re.IGNORECASE)
        question = match_q.group(1).strip() if match_q else ""

        match_ctx = re.search(r'RETRIEVED CONTEXT FROM DOCUMENT:\s*([\s\S]*?)(?:USER QUESTION:|$)', prompt, re.IGNORECASE)
        context = match_ctx.group(1).strip() if match_ctx else ""

        if not context or "not found" in context.lower() or len(context) < 30:
            return "I couldn't find this information in the uploaded document.\n\n*Note: LegalLens strictly provides grounded facts from your document. If you have questions regarding unstated terms, please consult a qualified legal professional.*"

        q_lower = question.lower()
        ctx_lower = context.lower()

        # Check if question keywords actually occur in context
        q_words = [w for w in re.sub(r'[^\w\s]', '', q_lower).split() if len(w) > 3 and w not in ['what', 'when', 'where', 'which', 'does', 'have', 'this', 'that', 'with']]
        matching_words = [w for w in q_words if w in ctx_lower]

        if not matching_words and len(q_words) > 0:
            return "I couldn't find this information in the uploaded document.\n\n*Note: LegalLens strictly relies on your document text. If this term was discussed verbally, verify if it appears in an addendum or consult an attorney.*"

        # Extract citation from context header
        citation = "Page 1, Section 1"
        header_match = re.search(r'\[Source:\s*([^\]]+)\]', context)
        if header_match:
            citation = header_match.group(1)

        # Build concise grounded response
        snippet = context[:400].replace('\n', ' ')
        return (
            f"Based on the document ({citation}):\n\n"
            f"The document states: \"{snippet}...\"\n\n"
            f"**In simple words:** The agreement outlines specific provisions concerning this matter. Review the referenced section carefully to confirm all conditions.\n\n"
            f"**Source:** [{citation}]\n\n"
            f"*Disclaimer: This answer provides document assistance and does not constitute formal legal advice.*"
        )

    def _mock_comparison(self, prompt: str) -> str:
        """Produces realistic comparative analysis between Document A and Document B."""
        data = {
            "document_a_name": "Document A (Original)",
            "document_b_name": "Document B (Revised)",
            "summary_of_differences": "The revised agreement introduces several material changes including adjusted payment schedules, modified termination notice periods, and added operational responsibilities.",
            "changes": [
                {
                    "category": "Payment & Fees",
                    "what_changed": "Adjustment to compensation / monthly payment terms or fee structure.",
                    "previous_version": "Original payment baseline specified in Document A.",
                    "new_version": "Updated financial terms or revised payment rate in Document B.",
                    "what_this_could_mean": "Direct financial impact on your cash flow or cost of the agreement.",
                    "source_previous": "Doc A, Section 2",
                    "source_new": "Doc B, Section 2"
                },
                {
                    "category": "Notice Period",
                    "what_changed": "Changed notification timeline required before termination or non-renewal.",
                    "previous_version": "30-day prior written notice required.",
                    "new_version": "Extended to 60-day prior written notice.",
                    "what_this_could_mean": "You must plan decisions earlier to avoid unwanted contract renewal or penalty.",
                    "source_previous": "Doc A, Section 5",
                    "source_new": "Doc B, Section 5"
                },
                {
                    "category": "Responsibilities & Penalties",
                    "what_changed": "Introduction of updated maintenance conditions or late penalty fee provisions.",
                    "previous_version": "Standard or relaxed compliance provisions.",
                    "new_version": "Stricter penalty structure or assigned responsibilities.",
                    "what_this_could_mean": "Increases your exposure to fees if deadlines or duties are missed.",
                    "source_previous": "Doc A, Section 7",
                    "source_new": "Doc B, Section 7"
                }
            ],
            "overall_takeaway": "Pay close attention to the revised notice period and any financial rate changes before signing the revised version."
        }
        return json.dumps(data)
