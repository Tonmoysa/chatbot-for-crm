# RAG সিস্টেম যাচাই (HR Policy Knowledge Base)

এই ডকুমেন্টে বলা আছে **আপনার RAG ঠিকমতো চলছে কিনা** কীভাবে ধাপে ধাপে পরীক্ষা করবেন — ইনফ্রা, ইনজেশন, চ্যাট API, লগ আর অটোমেটেড টেস্ট।

---

## সংক্ষেপ (কী কী লাগবে)

| জিনিস | কেন |
|--------|-----|
| **Qdrant** চালু (`QDRANT_URL`, ডিফল্ট `http://localhost:6333`) | ভেক্টর সার্চ এখানেই |
| **`LLM_API_KEY`** (বা `OPENAI_API_KEY`) | চ্যাট / RAG JSON উত্তর (গ্রাউন্ডিং) |
| **`EMBEDDING_BACKEND`** `openai` \| `local` | `local` হলে এম্বেড API লাগে না (`sentence-transformers`) |
| **`KB_RAG_ENABLED=true`** (ডিফল্ট true) | RAG বন্ধ থাকলে সবসময় স্ট্যাটিক হ্যান্ডবুক ফallback |
| কিছু **পলিসি টেক্সট ইনজেস্ট** | খালি কালেকশনে রিট্রিভ হবে না |

টেস্ট/CI তে `conftest.py` এ `KB_RAG_ENABLED=false` থাকতে পারে — লোকালি RAG চেক করতে `.env` এ **`KB_RAG_ENABLED=true`** রাখুন।

---

## ১. Qdrant চালু আছে কিনা

প্রথমে রুট URL চেক করুন (API জীবন্ত কিনা):

```text
GET http://localhost:6333/
```

উত্তরে JSON পেলে (যেমন `"title":"qdrant"` বা `version`) — **সার্ভার চলছে**। চ্যাটবটের RAG এই API দিয়েই চলে; ড্যাশবোর্ড লাগবে না।

### `/dashboard` এ ৪০৪ (Windows `qdrant.exe`)

**Windows-এ ডাউনলোড করা স্ট্যান্ডঅলোন বাইনারিতে** ওয়েব UI বান্ডল হয় না। তাই `http://localhost:6333/dashboard` অনেক সময় **৪০৪** দেখায় — এটা স্বাভাবিক।

**করণীয় (একটা বেছে নিন):**

1. **Docker (সবচেয়ে সহজ)** — ইমেজে UI থাকে:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
   তারপর `http://localhost:6333/dashboard` খুলুন।

2. **নেটিভ `qdrant.exe` রেখেই UI চান** — `qdrant.exe` যে ফোল্ডার থেকে চালান, সেখানে **`static`** ফোল্ডার বানিয়ে [qdrant-web-ui](https://github.com/qdrant/qdrant-web-ui/releases) এর `dist` কনটেন্ট আনজিপ করুন (অফিসিয়াল গাইড অনুযায়ী)।

কালেকশন লিস্ট API (ড্যাশবোর্ড ছাড়াই):

```text
GET http://localhost:6333/collections
```

নির্দিষ্ট কালেকশন (`.env` এর `QDRANT_COLLECTION`):

```text
GET http://localhost:6333/collections/hr_policies_local
```

---

## ১-ক. লোকাল এম্বেডিং (`EMBEDDING_BACKEND=local`)

`pip install sentence-transformers torch` (প্রজেক্ট `requirements.txt` এ আছে)।

`.env` উদাহরণ:

```env
EMBEDDING_BACKEND=local
LOCAL_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# GPU থাকলে: LOCAL_EMBED_DEVICE=cuda
```

- ডিফল্ট মডেলের ভেক্টর সাইজ **৩৮৪**। `QDRANT_VECTOR_SIZE` সেট না করলে Django `local` মোডে **৩৮৪** ধরে নেয়।
- আগে **১৫৩৬** ডাইমেনশনে (OpenAI) ইনডেক্স করা কালেকশন আছে → একই কালেকশনে লোকাল ভেক্টর দেবেন না। **`QDRANT_COLLECTION` বদলান** (যেমন `hr_policies_local`) অথবা Qdrant থেকে পুরনো কালেকশন ডিলিট করে আবার `ingest_policies` চালান।
- প্রথম রানে মডেল ডাউনলোড হতে পারে (কিছু মিনিট); লগে `local_embed_model_loaded` দেখবেন।

---

## ২. এনভায়রনমেন্ট ভেরিয়েবল (`.env`)

নিচেরগুলো মিলিয়ে নিন (`config/settings.py` এর সাথে):

```env
# বাধ্যতামূলক / প্রায় সবসময়
LLM_API_KEY=sk-...
# অথবা একই মান OPENAI_API_KEY=

# RAG / Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=hr_policies
OPENAI_EMBED_MODEL=text-embedding-3-small
KB_RAG_ENABLED=true

# লোকাল এম্বেডিং (API ছাড়া): Groq চ্যাট + লোকাল ভেক্টর
# EMBEDDING_BACKEND=local
# LOCAL_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# QDRANT_VECTOR_SIZE=384
# আগে OpenAI ১৫৩৬ দিয়ে কালেকশন বানালে: নতুন কালেকশন নাম দিন বা Qdrant থেকে পুরনো ডিলিট করুন
# QDRANT_COLLECTION=hr_policies_local

# ঐচ্ছিক টিউনিং
RAG_TOP_K=8
RAG_SCORE_THRESHOLD=0.62
```

API কলের জন্য চ্যাট সার্ভিসে:

```env
HR_SERVICE_API_KEY=your-secret-key
```

---

## ৩. নলেজ ইনজেস্ট (ডাটা ছাড়া RAG চলবে না)

### অপশন A — ফোল্ডার থেকে বাল্ক

```powershell
cd d:\chatbot\backend
$env:KB_POLICY_DIRS="d:\policies"
python manage.py ingest_policies --trace-id=manual-1
```

নির্দিষ্ট ফোল্ডার:

```powershell
python manage.py ingest_policies --dir d:\policies --trace-id=manual-2
```

আবার ইনডেক্স (চেকসাম ভিন্ন হলে নতুন; `--reindex` পুরনো চেকসাম রিলোড):

```powershell
python manage.py ingest_policies --dir d:\policies --reindex --trace-id=manual-3
```

সফল হলে লগে দেখবেন: `kb_ingest_done`, আর API রেসপন্সে `status: indexed`।

### অপশন B — API তে আপলোড

```http
POST /api/kb/upload-policy/
Content-Type: multipart/form-data
X-API-Key: <HR_SERVICE_API_KEY>

file=<your.pdf or .md>
title=Optional Title
department=OptionalDept
```

উত্তরের উদাহরণ:

```json
{
  "document_id": "1",
  "chunks_created": 12,
  "status": "indexed"
}
```

---

## ৪. চ্যাট API দিয়ে RAG চালু কিনা (সবচেয়ে গুরুত্বপূর্ণ)

**শুধু তখনই RAG চালে** যখন ইনটেন্ট **`HR_POLICY`** — অথবা **`UNKNOWN`** কিন্তু বার্তাটা পলিসি-সদৃশ (`is_rules_query` / শক্ত HR পলিসি হিউরিস্টিক)।

### রিকোয়েস্ট উদাহরণ (PowerShell)

`BASE` আর `KEY` নিজের মতো বসান।

```powershell
$BASE = "http://127.0.0.1:8000"
$KEY = "your-secret-key"

$body = @{
  message = "What is our policy on confidentiality and client data?"
  employee_id = "demo-employee"
} | ConvertTo-Json

Invoke-RestMethod -Uri "$BASE/api/chat/" -Method POST -Headers @{
  "X-API-Key" = $KEY
  "Content-Type" = "application/json"
} -Body $body
```

### সফল RAG এর চিহ্ন (রেসপন্স JSON এ)

1. **`intent`**: সাধারণত `"HR_POLICY"` (পলিসি প্রশ্ন হলে)।
2. **`sources`**: নন-এম্পটি অ্যারে — প্রতিটি আইটেমে অন্তত:
   - `document`, `section`, `snippet`, `score`
3. **`response.message`**: রিট্রিভ করা কনটেক্সট থেকে গ্রাউন্ডেড উত্তর; প্রমাণ না থাকলে ঠিক এই বার্তা হতে পারে:  
   `"I could not find this policy in the handbook."`
4. স্ট্যাটিক হ্যান্ডবুক সেকশন ম্যাচ হলে পুরনো আচরণ থাকে; RAG হিট হলে লগে **`rag_hr_policy_hit`** দেখা যাবে।

`UNKNOWN` + পলিসি-স্টাইল প্রশ্নে RAG হলে লগে **`rag_unknown_policy_hit`**।

---

## ৫. সার্ভার লগ (`trace_id` দিয়ে)

`runserver` কনসোলে লগার: **`hr_chatbot`** (INFO)।

খুঁজে দেখুন (ক্রম নয়, কাজের উপর নির্ভর করে):

| লগ স্টেপ | অর্থ |
|-----------|------|
| `rag_skip_full_handbook` | ইউজার পুরো হ্যান্ডবুক চাইছে — RAG স্কিপ (ইচ্ছাকৃত) |
| `rag_hr_policy_hit` | HR_POLICY পথে RAG উত্তর ব্যবহার |
| `rag_unknown_policy_hit` | UNKNOWN কিন্তু পলিসি-সদৃশ — RAG |
| `rag_no_hits` | Qdrant এ স্কোর থ্রেশহোল্ডের উপর হিট নেই → স্ট্যাটিক `rules_handbook` |
| `rag_qdrant_search_failed` | Qdrant ডাউন / নেটওয়ার্ক |
| `rag_embed_query_failed` | এম্বেডিং API ফেল |
| `kb_ingest_done` / `kb_ingest_qdrant_failed` | ইনজেশন OK / Qdrant আপসার্ট ফেল |

একই `trace_id` দিয়ে এক রিকোয়েস্টের পুরো পাইপলাইন ফলো করা সহজ।

---

## ৬. অটোমেটেড টেস্ট (রিগ্রেশন)

```powershell
cd d:\chatbot\backend
python -m pytest tests/test_kb_chunker.py tests/test_kb_sanitization.py tests/test_kb_rag_pipeline.py tests/test_kb_orchestrator.py -v
```

পুরো স্যুট:

```powershell
python -m pytest tests/ -q
```

নোট: ডিফল্ট `conftest.py` এ **`KB_RAG_ENABLED=false`** থাকতে পারে যাতে CI তে Qdrant না লাগে; RAG ইউনিট টেস্টগুলো নিজেরা `settings.KB_RAG_ENABLED = True` করে মক দিয়ে চালায়।

---

## ৭. সমস্যা সারণী (ট্রাবলশুটিং)

| লক্ষণ | সম্ভাব্য কারণ | করণীয় |
|--------|----------------|---------|
| সবসময় স্ট্যাটিক হ্যান্ডবুক উত্তর | `KB_RAG_ENABLED=false` বা Qdrant/এম্বেড ফেল | env + Qdrant + `LLM_API_KEY` |
| `sources` সবসময় `[]` | RAG পথে যাচ্ছে না (ইনটেন্ট) বা হিট নেই | উপযুক্ত পলিসি প্রশ্ন + ইনজেশ্ট কনটেন্ট |
| `I could not find this policy...` | রিট্রিভ দুর্বল বা LLM insufficient | থ্রেশহোল্ড কমান (`RAG_SCORE_THRESHOLD`), আরও রিলেভ্যান্ট ডক ইনজেস্ট |
| ইনজেশন `failed` / `qdrant_upsert_failed` | Qdrant বন্ধ বা কালেকশন | Qdrant চালু, লগ দেখুন |

---

## ৮. দ্রুত চেকলিস্ট

- [ ] Qdrant রানিং, `GET /collections/<QDRANT_COLLECTION>` OK  
- [ ] `LLM_API_KEY` সেট, এম্বেডিং মডেল মিলিয়েছে (`OPENAI_EMBED_MODEL`)  
- [ ] `KB_RAG_ENABLED=true` (লোকাল পরীক্ষার জন্য)  
- [ ] অন্তত একটি ডক `ingest_policies` বা `upload-policy` দিয়ে `indexed`  
- [ ] `POST /api/chat/` এ পলিসি প্রশ্ন → `sources` নন-এম্পটি বা যৌক্তিক “not found”  
- [ ] লগে `rag_hr_policy_hit` বা `rag_retrieval_done` (hits > 0)

---

## ফাইল রেফারেন্স (কোড ধরতে চাইলে)

| বিষয় | পথ |
|--------|-----|
| অর্কেস্ট্রেটরে RAG হুক | `chat/services/orchestrator.py` |
| RAG এক এন্ট্রি | `knowledge_base/services/rag_pipeline.py` |
| রিট্রিভাল | `knowledge_base/services/retriever.py` |
| ইনজেশন | `knowledge_base/services/ingest.py` |
| Qdrant | `knowledge_base/services/qdrant_service.py` |
| এম্বেডিং | `chat/services/llm_client.py` → `embed_texts` |

প্রশ্ন থাকলে এই ফাইলটির নাম বলে জিজ্ঞেস করলেই হবে: **`backend/docs/RAG_VERIFICATION.md`**.
