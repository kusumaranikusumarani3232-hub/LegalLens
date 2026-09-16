# ⚖️ LegalLens

**AI-powered legal document understanding and comparison for everyday users.**

> *Understand it before you sign it.*

---

## 🧩 The Problem

Legal documents — lease agreements, employment contracts, NDAs, freelance agreements, loan documents, and Terms & Conditions — are written by lawyers, for lawyers. For ordinary people:

- Dense legal jargon creates confusion and anxiety
- Important clauses (automatic renewals, penalty fees, notice periods) are buried in fine print
- It is expensive and slow to consult a lawyer for every document
- People routinely sign agreements they don't fully understand

The result: missed obligations, unexpected costs, and preventable disputes.

---

## 💡 The Solution — LegalLens

LegalLens is a GenAI-powered legal document assistant that transforms complicated legal documents into plain-language understanding, actionable checklists, and practical questions — so you know what you're agreeing to *before* you sign.

Upload a legal document and within seconds you receive:

| Feature | Description |
|---|---|
| 📘 Document Overview | Document type, parties, dates, jurisdiction |
| 📝 Plain-Language Summary | What the document is, what you agree to, what the other party agrees to |
| 🔍 Important Clauses | Extracted clauses with plain-English explanations and source citations |
| ⚠️ Attention Points | Neutral, non-alarmist identification of provisions that deserve scrutiny |
| 📌 Obligations & Deadlines | Actionable responsibilities per party with source references |
| ⚖️ Document Comparison | Side-by-side meaningful change analysis between two versions |
| 💬 Grounded Q&A | Ask questions grounded 100% in the document, with source citations |
| ✅ Action Checklist | Pre-signing and ongoing checklist personalized to the document |
| 👨‍⚖️ Consultation Questions | What to ask a lawyer or the counterparty before signing |

---

## ✨ Key Features in Detail

### 🔒 Grounding & Hallucination Prevention

LegalLens uses a custom **RAG (Retrieval-Augmented Generation)** pipeline:

```
Document
   ↓
Text Extraction (PDF / DOCX / TXT)
   ↓
Semantic Chunking (page-preserving, section-aware)
   ↓
TF-IDF + n-gram Vector Index
   ↓
Similarity Retrieval (top-k relevant chunks)
   ↓
Grounded LLM Prompt (strict context injection)
   ↓
Cited, grounded answer with source references
```

If information is absent from the document, LegalLens states:
> *"Not found in the document."*

It **never** invents facts, statutes, case law, or legal opinions.

### ⚖️ Safety-First Design

- All outputs pass a **SafetyGuardrail** check: prohibits alarmist language ("dangerous trap", "you will definitely lose"), definitive legal advice, or claims of legal validity
- Every screen shows a clear, non-intrusive **Legal Disclaimer**
- Neutral language enforced: "Worth reviewing" — not "Dangerous clause"
- LegalLens never claims to be a lawyer or guarantees legal outcomes

### 🧪 Offline / Demo Mode

**No API key required** to evaluate the full UI. Select *Offline / Demo Engine* in the sidebar and load any sample contract for an immediate hands-on demonstration.

---

## 🏗️ Architecture

```
LegalLens/
│
├── main.py                      ← Streamlit entry point
├── app/
│   ├── config.py                ← Environment & provider config
│   ├── models.py                ← Pydantic data schemas
│   ├── document_processor.py    ← PDF/DOCX/TXT extraction & chunking
│   ├── embeddings.py            ← Lightweight TF-IDF vectorizer
│   ├── rag.py                   ← RAG index & retrieval engine
│   ├── prompts.py               ← System & task prompts
│   ├── llm_provider.py          ← Gemini/Groq/OpenAI/Mock unified client
│   ├── legal_analyzer.py        ← Single-document analysis pipeline
│   ├── comparator.py            ← Two-document comparison engine
│   ├── safety.py                ← Safety guardrails & disclaimers
│   └── ui.py                    ← Streamlit UI components & pages
│
├── data/                        ← Sample legal documents (bundled)
│   ├── residential_lease_v1.txt
│   ├── residential_lease_v2_revised.txt
│   └── freelance_consulting_agreement.txt
│
├── tests/                       ← Unit test suite
│   ├── test_document_processor.py
│   ├── test_rag.py
│   ├── test_legal_analyzer.py
│   ├── test_comparator.py
│   ├── test_safety.py
│   └── test_qa_grounding.py
│
├── .env.example                 ← API key template
├── requirements.txt             ← Python dependencies
└── README.md
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit 1.35+ with custom CSS |
| **Backend** | Python 3.10+ |
| **LLM: Gemini** | `google-genai` (gemini-2.0-flash / gemini-1.5-flash) |
| **LLM: Groq** | `groq` SDK (llama-3.3-70b-versatile) |
| **LLM: OpenAI** | `openai` SDK (gpt-4o-mini) |
| **PDF Parsing** | `pypdf` |
| **DOCX Parsing** | `python-docx` |
| **Embeddings** | Lightweight TF-IDF + bigram n-gram vectorizer (no external service) |
| **Vector Index** | Custom cosine similarity retrieval (no Chroma/FAISS dependency) |
| **Config** | `python-dotenv` |
| **Schema Validation** | `pydantic` |
| **Testing** | `pytest` |

---

## 🚀 Installation & Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-org/legallens.git
cd legallens
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Copy the template and fill in your API key(s):
```bash
cp .env.example .env
```

Edit `.env`:
```env
DEFAULT_PROVIDER=gemini         # gemini | groq | openai | mock
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

> **Note:** All API key fields are optional. You can run the full demo UI without any API key using the **Offline / Demo Engine** (select `mock` as provider, or leave all keys blank).

### 5. Run the application
```bash
streamlit run main.py
```

Then open **http://localhost:8501** in your browser.

---

## 🔑 Environment Variables Reference

| Variable | Description | Default |
|---|---|---|
| `DEFAULT_PROVIDER` | `gemini`, `groq`, `openai`, or `mock` | `gemini` |
| `GEMINI_API_KEY` | Google AI Studio API key | — |
| `GROQ_API_KEY` | Groq platform API key | — |
| `OPENAI_API_KEY` | OpenAI platform API key | — |
| `GEMINI_MODEL` | Gemini model name override | `gemini-2.0-flash` |
| `GROQ_MODEL` | Groq model name override | `llama-3.3-70b-versatile` |
| `OPENAI_MODEL` | OpenAI model name override | `gpt-4o-mini` |

---

## 📖 Example Workflow

1. **Launch** `streamlit run main.py` → open http://localhost:8501
2. **Upload** a rental lease, NDA, employment contract, or freelance agreement (PDF / DOCX / TXT)  
   — OR — click **"🏠 Lease v1"** in the sidebar to load the bundled sample
3. Click **"🚀 Analyze This Document"**
4. Review the **Document Overview** panel: type, parties, dates, jurisdiction
5. Read the **Plain-English Summary** to understand what you're agreeing to
6. Expand **Key Clauses** to see each term in simple words with source citations
7. Check **Obligations & Deadlines** — know what actions you are responsible for
8. Review **Attention Points** — neutral flags on terms that deserve closer look
9. Use **💬 Ask Questions** to ask grounded questions about the document
10. Work through the **✅ Action Checklist** before signing
11. Generate **👨‍⚖️ Questions for Lawyer** to prepare for a consultation

### Document Comparison Workflow

1. Click the **⚖️ Compare Versions** tab
2. Upload or load two versions (e.g. Original Lease and Revised Lease)
3. Click **"🔍 Compare Versions Now"**
4. Review side-by-side meaningful change cards with plain-language implications

---

## 🏃 Running Tests

```bash
pytest -v tests/
```

18 tests covering:
- Document extraction (PDF, DOCX, TXT, empty file, scanned PDF, unsupported format)
- RAG chunking, retrieval grounding, and unrelated query threshold
- Legal analyzer structured output and missing information handling
- Document comparator meaningful change detection
- Safety guardrails, disclaimer enforcement, and neutral language
- Q&A grounding / hallucination prevention and JSON parsing robustness

---

## ☁️ Deployment

### Streamlit Community Cloud (Free, Recommended)

1. Fork/push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository and select `main.py` as the entry point
4. Add secrets under **App Settings → Secrets**:
   ```toml
   GEMINI_API_KEY = "your_key"
   DEFAULT_PROVIDER = "gemini"
   ```
5. Click **Deploy**

---

## ⚠️ Limitations

| Limitation | Details |
|---|---|
| Scanned PDFs | Scanned image PDFs with no embedded text are not supported. Use a text-based or OCR-processed PDF. |
| Very large documents | Documents beyond ~60,000 characters are trimmed to stay within model context limits. Key sections should still be analyzed. |
| Language | Currently optimized for English-language documents |
| LLM accuracy | AI-generated summaries and clause extractions may contain errors or omissions. Always verify with the source document. |
| Offline Mode | The Offline/Demo Engine uses heuristic keyword detection. It is suitable for demonstration only and is not as accurate as a real LLM. |

---

## 🤝 Responsible AI

LegalLens is built with responsible AI principles at its core:

- **Educational assistance only**: LegalLens provides document understanding and general information, not legal advice or legal opinions.
- **Grounding enforced**: All answers are strictly grounded in the uploaded document. The system is explicitly instructed to say "Not found in the document" rather than guessing.
- **No alarmist language**: Safety guardrails reject language claiming a clause is "illegal", "a trap", or that the user will "definitely win or lose".
- **Explicit disclaimer**: Every page displays a non-intrusive but clear legal disclaimer.
- **Temporary session storage**: Documents are processed in memory only and never persisted to disk.
- **Privacy-first**: No document contents are logged. API calls are made directly to the chosen AI provider.
- **Encourages professional advice**: Every relevant section recommends consulting a qualified legal professional.

> ⚖️ **LegalLens provides general information and document assistance, not legal advice. AI-generated explanations may be incomplete or incorrect. For important legal decisions or disputes, consult a qualified legal professional.**

---

## 📜 License

MIT License — see [LICENSE](LICENSE)
