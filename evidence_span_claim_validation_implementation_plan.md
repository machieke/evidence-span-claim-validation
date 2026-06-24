# Evidence-Span → Claim → Validation Pipeline: Full Implementation Plan

**Version:** 1.0  
**Date:** 2026-06-24  
**Scope:** Chat messages, PDFs, audio conversations, and images  
**Primary design goal:** Extract source-faithful, attributed, atomic claims from multimodal sources while preserving verifiable evidence provenance and quarantining unsupported or risky outputs.

---

## 1. Executive summary

The system should not treat “chunks” as knowledge. Chunks are only temporary context windows used to help models interpret source material. The durable knowledge unit should be a validated, evidence-grounded claim.

The core pipeline is:

```text
source
  → modality-specific ingestion
  → evidence substrate
  → safe context chunks
  → claim-bearing span or region detection
  → source-faithful atomic claim extraction
  → deterministic validation
  → optional semantic validation / repair
  → normalized semantic claim
  → graph / search / reasoning export
```

The required conceptual separation is:

```text
chunk    = temporary context window
span     = exact evidence anchor
claim    = source-faithful extracted assertion or observation
normal   = normalized semantic representation of the claim
atom     = downstream reasoning object derived from the validated claim
```

For text-like modalities, including chat, PDF text, OCR, and audio transcripts, the evidence anchor is an exact text span. For images, the evidence anchor is a visual region, crop, bounding box, mask, patch, or feature cluster. Image-derived claims must be stored as model observations or visual hypotheses unless human-confirmed.

The invariant across all modalities is:

```text
Every accepted claim must point back to one or more evidence records.
Every evidence record must point back to the source object and provenance coordinates.
No extracted claim should be promoted to truth merely because a model generated it.
```

---

## 2. Core principles

### 2.1 Preserve provenance before extraction

The first priority is not claim generation; it is evidence anchoring. Every source object must be decomposed into evidence-bearing units with stable IDs and provenance.

Examples:

| Modality | Evidence anchor | Provenance |
|---|---|---|
| Chat | message span or quoted message span | conversation ID, message ID, sender, timestamp, character offsets |
| PDF | sentence/block span | document ID, page, block ID, bounding box, section path, character offsets |
| Audio | utterance span | source ID, speaker label, start/end timestamps, ASR confidence, diarization confidence |
| Image | region, crop, mask, patch, OCR text region, feature cluster | image ID, bbox/mask/crop path, proposal model, embedding model, classifier/VLM metadata |

### 2.2 Claims are attributed, not verified

The extraction stage should not decide whether a statement is true in the world. It should decide what the source says, shows, reports, asks, denies, or visually suggests.

For example:

```text
Bob: The engine was replaced last week.
```

Do not store this as:

```text
The engine was replaced last week.
```

Store it as:

```text
Bob asserted that the engine was replaced last week.
```

with an optional nested proposition:

```json
{
  "subject": "engine",
  "predicate": "was_replaced",
  "object": "last week",
  "attribution": "Bob",
  "truth_status": "speaker_asserted_unverified"
}
```

### 2.3 Exact evidence is mandatory for text-like modalities

For chat, PDF text, OCR text, and audio transcripts, `evidence_text` must be an exact substring of the evidence span or the primary source text unit.

The validator must reject, repair, or quarantine any claim whose evidence text is not an exact substring.

### 2.4 Image claims are model observations, not direct text claims

Images do not contain explicit linguistic claims unless they contain readable text. A visual model can classify or cluster regions, but those outputs should be represented as model observations.

Use:

```text
Model M classified region R as mast with score S.
```

or:

```text
Regions R1, R2, and R3 were clustered as visually similar under embedding model E.
```

Do not immediately store:

```text
The image contains a mast.
```

unless the label has passed a stronger validation or human review policy.

### 2.5 Validation is a first-class pipeline stage

The LLM extractor is not trusted. Validation is not optional. At minimum, the system must check:

1. Required fields exist.
2. Enum values are valid.
3. Confidence values are bounded.
4. Evidence text is exact for text-like modalities.
5. Speaker, page, block, region, or message provenance exists.
6. Negation is preserved.
7. Uncertainty is preserved.
8. Attribution is preserved.
9. Quantities and dates are preserved.
10. New unsupported entities are not introduced.

---

## 3. System architecture

### 3.1 High-level service layout

```text
                 ┌───────────────────────┐
                 │  Source Ingestion API  │
                 └───────────┬───────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Chat Ingest  │      │ PDF Ingest   │      │ Audio Ingest │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       ▼                     ▼                     ▼
┌──────────────────────────────────────────────────────────┐
│              Evidence Registry / JSONL Store              │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                    Context Chunkers                       │
│  message/thread chunks · section chunks · utterance chunks │
│  visual region groups / cluster contexts                  │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                Claim-Bearing Span Detector                │
│  rule-based v1 · classifier v2 · semantic highlighter v3  │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                  Claim Extraction Layer                   │
│     LLM strict JSON · schema repair · model routing       │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                    Validation Layer                       │
│ deterministic · modality-specific · semantic · quarantine │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                  Normalization Layer                      │
│ entity canonicalization · predicate mapping · dedupe      │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                Downstream Storage / Export                │
│ JSONL · SQL · vector index · graph DB · AtomSpace export  │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Recommended implementation stack

Use a modular Python stack first. Avoid over-engineering the first version.

| Layer | Recommended v1 |
|---|---|
| Language | Python 3.11+ |
| CLI | Typer or Click |
| Schemas | Pydantic v2 |
| Local storage | JSONL + SQLite or DuckDB |
| Production metadata store | PostgreSQL |
| Large artifacts | Local filesystem first; S3-compatible object storage later |
| Embeddings / vector search | FAISS locally; Qdrant/Milvus later if needed |
| Job orchestration | Simple resumable CLI first; Celery/RQ/Prefect/Dagster later |
| LLM abstraction | Provider-neutral adapter: `extract_json(prompt, schema)` |
| Logging | structured JSON logs |
| Reports | HTML or Markdown summary generated from JSONL |

---

## 4. Repository and artifact layout

Recommended repository layout:

```text
project/
  README.md
  pyproject.toml
  .env.example

  configs/
    pipeline.yaml
    models.yaml
    validation.yaml
    modalities.yaml

  prompts/
    extract_claims.core.md
    extract_claims.chat.md
    extract_claims.pdf.md
    extract_claims.audio.md
    extract_claims.image_region.md
    validate_claims.md
    repair_schema.md
    normalize_claim.md

  src/
    evidence_pipeline/
      __init__.py
      cli.py
      config.py
      ids.py
      jsonl.py
      logging.py

      schemas/
        base.py
        sources.py
        evidence.py
        chunks.py
        spans.py
        claims.py
        validation.py
        image.py
        audio.py
        pdf.py
        chat.py

      ingest/
        chat.py
        pdf.py
        audio.py
        image.py

      chunking/
        chat_chunker.py
        pdf_chunker.py
        audio_chunker.py
        image_region_grouper.py

      spans/
        rule_highlighter.py
        semantic_highlighter.py
        sentence_splitter.py
        image_region_selector.py

      extraction/
        llm_client.py
        claim_extractor.py
        schema_repair.py
        batching.py

      validation/
        deterministic.py
        text_support.py
        modality.py
        image_validation.py
        quarantine.py

      normalization/
        entities.py
        predicates.py
        dedupe.py
        graph_export.py

      reports/
        summary.py
        gold_eval.py

  data/
    raw/
      chat/
      pdf/
      audio/
      images/
    work/
      normalized_audio/
      normalized_images/
      crops/
      masks/
      vectors/
    jsonl/
      sources.jsonl
      chat_messages.jsonl
      pdf_blocks.jsonl
      audio_utterances.jsonl
      image_regions.jsonl
      chunks.jsonl
      spans.jsonl
      claims.raw.jsonl
      claims.validated.jsonl
      claims.normalized.jsonl
      validations.jsonl
      errors.jsonl
      quarantine.jsonl
    reports/
      extraction_summary.md
      extraction_summary.html
```

---

## 5. Common data model

### 5.1 Source record

Every input object should be registered as a source.

```json
{
  "source_id": "src_01JABC...",
  "source_modality": "chat|pdf|audio|image",
  "source_file": "raw/pdf/report.pdf",
  "source_uri": null,
  "sha256": "...",
  "created_at": "2026-06-24T08:10:00Z",
  "ingested_at": "2026-06-24T08:12:00Z",
  "metadata": {
    "title": "Inspection Report",
    "language": "en"
  },
  "schema_version": "source.v1"
}
```

### 5.2 Evidence record

The evidence record is the common abstraction across modalities.

```json
{
  "evidence_id": "ev_01JABC...",
  "source_id": "src_01JABC...",
  "source_modality": "pdf",
  "evidence_type": "text_span|utterance_span|message_span|visual_region|visual_cluster|ocr_text_span",
  "text": "The vessel Hope appears to have an older diesel engine.",
  "provenance": {
    "page": 17,
    "block_id": "pdf_001_p17_b03",
    "bbox": [72, 155, 510, 198],
    "char_start": 120,
    "char_end": 178
  },
  "risk_flags": [],
  "schema_version": "evidence.v1"
}
```

For image evidence, `text` may be null unless the evidence is OCR text.

```json
{
  "evidence_id": "ev_img_001_r003",
  "source_id": "img_001",
  "source_modality": "image",
  "evidence_type": "visual_region",
  "text": null,
  "provenance": {
    "image_id": "img_001",
    "region_id": "img_001_r003",
    "bbox": [120, 40, 30, 260],
    "mask_path": "work/masks/img_001_r003.png",
    "crop_path": "work/crops/img_001_r003.png",
    "proposal_model": "grid_224_stride112",
    "proposal_score": null
  },
  "risk_flags": ["low_resolution"],
  "schema_version": "evidence.v1"
}
```

### 5.3 Chunk record

Chunks are context windows. They are not final knowledge units.

```json
{
  "chunk_id": "chunk_01JABC...",
  "source_id": "src_01JABC...",
  "source_modality": "pdf",
  "evidence_ids": ["ev_001", "ev_002", "ev_003"],
  "primary_evidence_ids": ["ev_002", "ev_003"],
  "overlap_evidence_ids": ["ev_001"],
  "text": "...",
  "provenance_summary": {
    "pages": [16, 17],
    "section_path": ["Inspection", "Engine Condition"]
  },
  "chunking_policy": {
    "strategy": "section_paragraph_token_fallback",
    "target_tokens": 1200,
    "overlap_tokens": 150
  },
  "schema_version": "chunk.v1"
}
```

### 5.4 Span record

Span records identify claim-bearing evidence. For text-like modalities, spans should be exact text ranges.

```json
{
  "span_id": "span_01JABC...",
  "chunk_id": "chunk_01JABC...",
  "source_id": "src_01JABC...",
  "source_modality": "chat",
  "evidence_id": "ev_msg_001",
  "text": "I saw the boat Hope yesterday.",
  "char_start": 0,
  "char_end": 32,
  "context_text": "Previous message: Did you see Hope?",
  "label": "claim_bearing",
  "score": 0.91,
  "detector": {
    "name": "rules_v1",
    "version": "0.1.0"
  },
  "risk_flags": ["context_dependent_coreference"],
  "schema_version": "span.v1"
}
```

For images, the equivalent “span” is a visual region or cluster.

```json
{
  "span_id": "span_img_001_r003",
  "chunk_id": null,
  "source_id": "img_001",
  "source_modality": "image",
  "evidence_id": "ev_img_001_r003",
  "text": null,
  "label": "visual_region_candidate",
  "score": 0.82,
  "detector": {
    "name": "grid_patch_proposal",
    "version": "0.1.0"
  },
  "risk_flags": ["partial_object"],
  "schema_version": "span.v1"
}
```

### 5.5 Raw claim record

Raw claims are produced by the extractor and should not be trusted until validated.

```json
{
  "claim_id": "claim_01JABC...",
  "source_id": "src_01JABC...",
  "source_modality": "pdf",
  "span_id": "span_01JABC...",
  "evidence_id": "ev_001",
  "claim_type": "attributed_text_claim",
  "source_faithful_claim": "The report says the vessel Hope appears to have an older diesel engine.",
  "subject": "vessel Hope",
  "predicate": "appears_to_have_condition",
  "object": "older diesel engine",
  "quantity": null,
  "attributes": {
    "hedge": "appears"
  },
  "modality": "uncertain_observation",
  "evidence_text": "appears to have an older diesel engine",
  "attribution": {
    "type": "document",
    "agent": "report"
  },
  "truth_status": "source_asserted_unverified",
  "confidence": 0.65,
  "model": {
    "provider": "llm_provider",
    "model": "model_name",
    "prompt_version": "extract_claims.pdf.v1"
  },
  "support_status": "raw_extracted",
  "risk_flags": [],
  "schema_version": "claim.raw.v1"
}
```

### 5.6 Validated claim record

Validated claims are accepted or rejected with explicit reasons.

```json
{
  "claim_id": "claim_01JABC...",
  "source_id": "src_01JABC...",
  "source_modality": "pdf",
  "span_id": "span_01JABC...",
  "evidence_id": "ev_001",
  "source_faithful_claim": "The report says the vessel Hope appears to have an older diesel engine.",
  "evidence_text": "appears to have an older diesel engine",
  "normalized_claim": {
    "subject": "vessel Hope",
    "predicate": "appears_condition",
    "object": "older diesel engine"
  },
  "modality": "uncertain_observation",
  "truth_status": "source_asserted_unverified",
  "support_status": "accepted_extracted",
  "validation": {
    "deterministic_valid": true,
    "evidence_exact_match": true,
    "negation_preserved": true,
    "uncertainty_preserved": true,
    "attribution_preserved": true,
    "quantities_preserved": true,
    "introduced_entities": [],
    "validator_version": "deterministic.v1"
  },
  "risk_flags": [],
  "schema_version": "claim.validated.v1"
}
```

---

## 6. Pipeline stages in detail

### 6.1 Stage A — Ingest source

Responsibilities:

1. Compute source hash.
2. Assign stable `source_id`.
3. Store source metadata.
4. Preserve original file or message export unchanged.
5. Produce modality-specific base records.

Output files:

```text
sources.jsonl
chat_messages.jsonl
pdf_blocks.jsonl
audio_sources.jsonl
audio_utterances.jsonl
images.jsonl
image_regions.jsonl
```

### 6.2 Stage B — Build evidence substrate

This stage converts each modality into evidence anchors.

```text
chat message       → message_span evidence
PDF page/block     → text_span evidence
ASR utterance      → utterance_span evidence
image region       → visual_region evidence
image OCR text     → ocr_text_span evidence
visual cluster     → visual_cluster evidence
```

The output is `evidence.jsonl`.

### 6.3 Stage C — Build context chunks

The chunker groups evidence records into model-friendly context windows.

Rules:

- Chunks may include overlap for context.
- Chunks must distinguish primary evidence from overlap context.
- Claim extraction should normally be allowed only from primary evidence.
- Context evidence may help resolve references, but it should not be silently converted into evidence for claims.

Output is `chunks.jsonl`.

### 6.4 Stage D — Detect claim-bearing spans or regions

The span detector reduces extraction cost and improves precision.

V1 approach:

- Rule-based detector for chat and audio transcripts.
- Sentence and clause splitting for PDFs.
- Region proposal and clustering for images.

V2 approach:

- Train a binary classifier: `claim_bearing` vs `not_claim_bearing`.
- Use manually reviewed accepted/rejected spans as training data.

V3 approach:

- Use semantic highlighting or query-conditioned relevance scoring when extraction is performed for a specific task or goal.

Output is `spans.jsonl`.

### 6.5 Stage E — Extract source-faithful claims

The LLM extractor receives:

1. System instructions.
2. Modality-specific rules.
3. A chunk for local context.
4. One or more target spans.
5. Strict JSON schema.

The extractor returns one or more atomic claims per span.

Output is `claims.raw.jsonl`.

### 6.6 Stage F — Validate claims

Validation should run in layers.

```text
schema validation
  → deterministic evidence validation
  → modality-specific validation
  → semantic support validation
  → repair or quarantine
```

Output files:

```text
claims.validated.jsonl
validations.jsonl
quarantine.jsonl
errors.jsonl
```

### 6.7 Stage G — Normalize claims

Normalization should only operate on accepted claims.

Tasks:

- Canonicalize entities.
- Canonicalize predicates.
- Normalize units and dates where safe.
- Deduplicate semantically equivalent claims.
- Preserve the original `source_faithful_claim` unchanged.
- Record all transformations as derived data.

Output is `claims.normalized.jsonl`.

---

## 7. Chat message pipeline

### 7.1 Chat input assumptions

Chat data may arrive as:

- JSON export.
- Database rows.
- Slack/Discord/Telegram/WhatsApp exports.
- Plain text logs.
- Application-native conversation records.

The pipeline should normalize all formats into `chat_messages.jsonl`.

### 7.2 Chat message schema

```json
{
  "message_id": "msg_001",
  "source_id": "chat_src_001",
  "conversation_id": "conv_001",
  "thread_id": "thread_001",
  "sender_id": "user_001",
  "sender_display_name": "Kevin",
  "sender_role": "user|assistant|system|external",
  "timestamp": "2026-06-24T08:12:43Z",
  "text": "I saw the boat Hope yesterday. It had three masts.",
  "reply_to_message_id": null,
  "quoted_message_ids": [],
  "edit_history": [],
  "attachments": [],
  "metadata": {
    "platform": "chat_export"
  },
  "risk_flags": []
}
```

### 7.3 Chat evidence generation

Each message should become a message evidence record. Long messages can be split into sentence or paragraph spans, but the source message ID must remain available.

```json
{
  "evidence_id": "ev_msg_001_s001",
  "source_modality": "chat",
  "source_id": "chat_src_001",
  "evidence_type": "message_span",
  "text": "I saw the boat Hope yesterday.",
  "provenance": {
    "conversation_id": "conv_001",
    "thread_id": "thread_001",
    "message_id": "msg_001",
    "sender_id": "user_001",
    "timestamp": "2026-06-24T08:12:43Z",
    "char_start": 0,
    "char_end": 32
  }
}
```

### 7.4 Chat chunking

Chat is context-dependent. Use thread-aware chunking.

Policy:

- Primary unit: message.
- Context window: previous 2–6 messages, bounded by token count.
- Preserve message boundaries and speaker labels.
- For direct replies, include the parent message even if outside the recent window.
- Mark context messages separately from primary messages.

Example chunk:

```json
{
  "chunk_id": "chat_conv_001_c0042",
  "source_modality": "chat",
  "source_id": "chat_src_001",
  "evidence_ids": ["ev_msg_099", "ev_msg_100", "ev_msg_101"],
  "primary_evidence_ids": ["ev_msg_101"],
  "overlap_evidence_ids": ["ev_msg_099", "ev_msg_100"],
  "text": "USER_A: Did Hope have masts?\nUSER_B: I saw it yesterday. It had three masts.",
  "chunking_policy": {
    "strategy": "thread_window",
    "previous_messages": 2,
    "max_tokens": 1200
  }
}
```

### 7.5 Chat span detection

A chat span should be marked `claim_bearing` if it includes:

- factual assertion,
- direct observation,
- report of what someone said,
- negation,
- uncertainty,
- quantity,
- date or temporal claim,
- commitment or plan,
- causal explanation,
- comparison,
- definition.

Skip or deprioritize:

- greetings,
- filler,
- pure reactions,
- emoji-only messages,
- backchannels,
- acknowledgements,
- rhetorical statements,
- unsupported extracted claims from system/developer instructions unless the system is explicitly part of the analyzed corpus.

Questions require special handling. Do not extract presupposed content as fact. For example:

```text
Did Hope still have three masts?
```

Extract at most:

```text
The speaker asked whether Hope still had three masts.
```

### 7.6 Chat extraction rules

Prompt wrapper:

```text
You extract source-faithful claims from chat messages.

Rules:
- Extract claims made by the message sender; do not verify them.
- Every claim must preserve the sender attribution.
- Do not treat a speaker's statement as a world fact.
- If the message says "I saw X", represent it as a reported observation by the sender.
- Preserve uncertainty, hedging, negation, quantities, dates, and temporal markers.
- Use previous messages only as context; evidence_text must be copied exactly from the target message span.
- If the claim depends on context, set context_dependent=true and include context_used.
- Do not resolve pronouns unless context explicitly supports the resolution.
- Return JSON only.
```

### 7.7 Chat validation

Deterministic checks:

1. `evidence_text` is exact substring of the target message span.
2. `sender_id` exists.
3. `conversation_id` and `message_id` exist.
4. The claim is attributed to the sender or marked as a message act.
5. Context-derived references are flagged with `context_dependent=true`.
6. No context-only fact becomes a primary claim without evidence.
7. Negation, uncertainty, quantities, and dates are preserved.

Risk flags:

```json
[
  "context_dependent_coreference",
  "quoted_message_ambiguity",
  "edited_message",
  "deleted_parent_message",
  "sarcasm_possible",
  "assistant_generated_text",
  "low_confidence_sender_identity"
]
```

---

## 8. PDF pipeline

### 8.1 PDF ingestion strategy

Start with PyMuPDF for the MVP because it can extract page text blocks and bounding boxes. Add GROBID for scientific papers and Unstructured for mixed business documents where layout elements are important.

Recommended extractor selection:

| PDF type | First choice | Fallback |
|---|---|---|
| Digitally generated reports | PyMuPDF | Unstructured |
| Scientific papers | GROBID | PyMuPDF + section detection |
| Scanned PDFs | OCR layer first | Unstructured or Tesseract/PaddleOCR |
| Mixed forms/tables | Unstructured | PyMuPDF table extraction + OCR |
| Highly complex academic PDFs | GROBID + OCR fallback | Marker/Nougat-style conversion if acceptable |

### 8.2 PDF block schema

```json
{
  "block_id": "pdf_001_p017_b003",
  "source_id": "pdf_001",
  "source_file": "raw/pdf/report.pdf",
  "page": 17,
  "block_no": 3,
  "block_type": "text|image|table|caption|header|footer|unknown",
  "text": "The vessel Hope appears to have an older diesel engine.",
  "bbox": [72.0, 155.0, 510.0, 198.0],
  "char_start_document": 18320,
  "char_end_document": 18378,
  "section_path": ["Inspection", "Engine Condition"],
  "risk_flags": []
}
```

### 8.3 PDF preprocessing

PDF text is often noisy. Implement cleanup before chunking.

Required cleanup:

- Remove repeated headers and footers using page frequency detection.
- Remove page numbers and repeated boilerplate.
- Repair hyphenated line breaks.
- Join wrapped lines inside paragraphs.
- Preserve list boundaries.
- Preserve table/caption markers.
- Mark references, acknowledgements, footnotes, and appendices.
- Detect headings and section hierarchy.
- Preserve original block text separately from cleaned text.

Store both:

```json
{
  "original_text": "The ves-\nsel Hope appears old.",
  "cleaned_text": "The vessel Hope appears old.",
  "cleanup_actions": ["repair_hyphenation", "join_wrapped_lines"]
}
```

### 8.4 PDF chunking

Preferred chunking order:

1. Section/subsection boundary.
2. Paragraph boundary.
3. Page/block boundary.
4. Token limit fallback.

Target:

- `target_tokens`: 800–1800.
- `overlap_tokens`: 100–200.
- Keep tables/captions as separate evidence types where possible.
- Do not merge references into main body chunks unless the domain requires bibliographic claims.

PDF chunk schema:

```json
{
  "chunk_id": "pdf_001_c042",
  "source_id": "pdf_001",
  "source_modality": "pdf",
  "pages": [16, 17],
  "section_path": ["Inspection", "Engine Condition"],
  "block_ids": ["pdf_001_p016_b009", "pdf_001_p017_b001"],
  "primary_evidence_ids": ["ev_pdf_001_p017_b003"],
  "overlap_evidence_ids": ["ev_pdf_001_p016_b009"],
  "char_start_document": 18320,
  "char_end_document": 20144,
  "text": "...",
  "chunking_policy": {
    "strategy": "section_paragraph_token_fallback",
    "target_tokens": 1200,
    "overlap_tokens": 150
  }
}
```

### 8.5 PDF span detection

V1:

- Sentence splitter.
- Clause splitter for semicolon-heavy or compound sentences.
- Rule highlighter for claim-bearing language.

Claim-bearing signals:

- `is`, `was`, `has`, `contains`, `uses`, `requires`, `causes`, `indicates`, `reports`, `found`, `observed`, `measured`, `increased`, `decreased`.
- numbers, dates, measurements, comparisons.
- negation and uncertainty.
- citations or attribution phrases.

Skip:

- table of contents,
- repeated headers/footers,
- references unless configured,
- boilerplate disclaimers unless configured,
- standalone captions unless useful.

### 8.6 PDF extraction prompt wrapper

```text
You extract source-faithful, attributed claims from PDF text spans.

Rules:
- Extract claims made by the document; do not verify them.
- Every claim must quote exact evidence_text from the target span.
- Split compound statements into atomic claims.
- Preserve uncertainty words such as appears, seems, likely, may, allegedly.
- Preserve negation.
- Preserve quantities, units, dates, and comparisons.
- Do not introduce entities absent from the span or explicit local context.
- If a claim is attributed to a study, author, report, speaker, figure, or table, preserve that attribution.
- Use section and nearby text only for context; evidence_text must come from the target span.
- Return JSON only.
```

### 8.7 PDF validation

Deterministic checks:

1. `evidence_text` exact substring of span or block cleaned text.
2. Span maps to page and block.
3. Page number exists.
4. Block ID exists.
5. Bounding box exists for digitally extracted text where available.
6. Claim attribution preserved.
7. Units and quantities preserved.
8. No evidence extracted from references/boilerplate unless policy allows.

Quarantine reasons:

```json
[
  "evidence_not_exact_substring",
  "missing_page_provenance",
  "missing_block_provenance",
  "unsupported_entity_introduced",
  "uncertainty_dropped",
  "negation_dropped",
  "quantity_changed",
  "table_fragment_ambiguous",
  "ocr_low_confidence"
]
```

---

## 9. Audio pipeline

### 9.1 Audio architecture

Audio is not chunked like a document. It must first become speaker-attributed, timestamped utterance evidence.

```text
audio file
  → audio normalization
  → voice activity detection
  → speaker diarization
  → speech-to-text with timestamps
  → ASR/diarization alignment
  → speaker-attributed utterances
  → conversation-safe chunks
  → claim-bearing utterance spans
  → speaker-attributed claim extraction
  → validation / quarantine
```

### 9.2 Audio source schema

```json
{
  "source_id": "audio_001",
  "source_modality": "audio",
  "source_file": "raw/audio/meeting.mp3",
  "normalized_file": "work/normalized_audio/meeting_16khz_mono.wav",
  "duration_seconds": 3720.4,
  "sample_rate": 16000,
  "channels": 1,
  "language": "en",
  "sha256": "...",
  "ingested_at": "2026-06-24T08:20:00Z"
}
```

### 9.3 Audio normalization

Normalize files to a consistent format:

```text
16 kHz
mono
WAV
```

Example command:

```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 work/normalized_audio/input.wav
```

### 9.4 Voice activity detection

VAD separates speech from silence/noise.

Output:

```json
{
  "segment_id": "audio_001_vad_000123",
  "source_id": "audio_001",
  "start": 183.42,
  "end": 190.18,
  "kind": "speech",
  "confidence": 0.92,
  "model": "silero_vad|webrtc_vad|pyannote_vad|asr_internal"
}
```

### 9.5 Speaker diarization

Diarization answers “who spoke when?” It does not identify real-world identities by itself.

Output:

```json
{
  "turn_id": "turn_001",
  "source_id": "audio_001",
  "speaker": "SPEAKER_00",
  "start": 183.42,
  "end": 190.18,
  "confidence": 0.82,
  "model": "pyannote_audio",
  "risk_flags": []
}
```

Identity resolution must be separate:

```json
{
  "speaker_label": "SPEAKER_00",
  "identity": "Alice",
  "source_id": "audio_001",
  "basis": "human_review|metadata|self_identification",
  "confidence": 0.95,
  "evidence_id": "ev_..."
}
```

### 9.6 Speech-to-text with timestamps

Required outputs:

- segment timestamps,
- word timestamps where possible,
- language,
- confidence if available,
- no-speech probability or equivalent risk metric if available.

ASR segment schema:

```json
{
  "asr_segment_id": "asr_001",
  "source_id": "audio_001",
  "start": 183.42,
  "end": 190.18,
  "text": "I saw the boat Hope yesterday. It had three masts.",
  "words": [
    {"word": "I", "start": 183.42, "end": 183.50, "confidence": 0.93},
    {"word": "saw", "start": 183.51, "end": 183.70, "confidence": 0.91}
  ],
  "asr_confidence": 0.91,
  "model": "whisperx|faster_whisper|cloud_asr"
}
```

### 9.7 Align ASR and speakers

Combine ASR segments and diarization turns into speaker-attributed utterances.

```json
{
  "utterance_id": "utt_001",
  "source_id": "audio_001",
  "speaker": "SPEAKER_02",
  "start": 183.42,
  "end": 190.18,
  "text": "I saw the boat Hope yesterday. It had three masts.",
  "asr_segment_ids": ["asr_001"],
  "turn_ids": ["turn_092"],
  "asr_confidence": 0.91,
  "diarization_confidence": 0.82,
  "risk_flags": []
}
```

Risk flags:

```json
[
  "overlapping_speech",
  "low_asr_confidence",
  "speaker_uncertain",
  "background_noise",
  "music",
  "cross_talk",
  "language_switch",
  "possible_asr_hallucination"
]
```

### 9.8 Audio evidence generation

Each utterance or utterance clause becomes an evidence span.

```json
{
  "evidence_id": "ev_audio_001_utt_001",
  "source_id": "audio_001",
  "source_modality": "audio",
  "evidence_type": "utterance_span",
  "text": "I saw the boat Hope yesterday.",
  "provenance": {
    "utterance_id": "utt_001",
    "speaker": "SPEAKER_02",
    "start": 183.42,
    "end": 186.10,
    "asr_confidence": 0.91,
    "diarization_confidence": 0.82
  },
  "risk_flags": []
}
```

### 9.9 Audio chunking

Use speaker/time/message structure, not arbitrary token windows.

Policy:

- 30–120 seconds per chunk, or 8–30 utterances.
- Include previous 1–2 utterances as overlap.
- Preserve speaker labels.
- Primary evidence is the target utterance or span.
- Context evidence can resolve pronouns but should not automatically support claims.

Chunk example:

```json
{
  "chunk_id": "audio_001_c0042",
  "source_id": "audio_001",
  "source_modality": "audio",
  "start": 180.0,
  "end": 240.0,
  "utterance_ids": ["utt_100", "utt_101", "utt_102"],
  "primary_evidence_ids": ["ev_audio_001_utt_102"],
  "overlap_evidence_ids": ["ev_audio_001_utt_101"],
  "text": "SPEAKER_01: Did Hope have an engine?\nSPEAKER_02: Yeah, but it seemed old.",
  "chunking_policy": {
    "unit": "utterance",
    "max_seconds": 90,
    "max_tokens": 1200,
    "overlap_utterances": 1
  }
}
```

### 9.10 Audio extraction prompt wrapper

```text
You extract source-faithful claims from timestamped conversation spans.

Rules:
- Every claim must preserve speaker attribution.
- Do not treat speaker statements as verified facts.
- If a speaker says "I saw X", represent it as that speaker's reported observation.
- Preserve uncertainty, hedging, negation, quantities, dates, and temporal markers.
- If the span depends on previous context, set context_dependent=true and include context_used.
- Do not resolve pronouns unless the context explicitly supports it.
- evidence_text must be copied exactly from the target utterance span.
- Return JSON only.
```

### 9.11 Audio validation

Deterministic checks:

1. `evidence_text` exact substring of utterance text.
2. `speaker` exists and is not silently resolved to a real identity.
3. Start/end timestamps exist and fall within source duration.
4. Claim includes speaker attribution or is a clearly marked speech act.
5. ASR confidence and diarization confidence propagate to claim risk flags.
6. Overlapping speech is flagged or quarantined.
7. Context-dependent claims are flagged.
8. Negation, uncertainty, quantities, and temporal references are preserved.

Policy examples:

```yaml
audio_validation:
  quarantine_if:
    asr_confidence_lt: 0.55
    diarization_confidence_lt: 0.50
    overlapping_speech: true
  needs_review_if:
    context_dependent: true
    speaker_uncertain: true
    asr_confidence_lt: 0.75
```

---

## 10. Image pipeline

### 10.1 Image architecture

Images require a visual evidence layer before claim extraction.

```text
image file
  → metadata extraction and normalization
  → region / patch / mask proposal
  → crop and mask persistence
  → visual embeddings
  → similarity clustering or matching
  → optional OCR text extraction
  → optional VLM/detector labeling
  → visual evidence records
  → visual claims
  → validation / review / downstream reasoning
```

### 10.2 Image source schema

```json
{
  "image_id": "img_001",
  "source_id": "img_001",
  "source_modality": "image",
  "source_file": "raw/images/boat_001.jpg",
  "normalized_file": "work/normalized_images/boat_001.jpg",
  "width": 1920,
  "height": 1080,
  "exif_datetime": "2026-06-24T10:30:00Z",
  "sha256": "...",
  "perceptual_hash": "...",
  "ingested_at": "2026-06-24T10:35:00Z",
  "risk_flags": []
}
```

### 10.3 Image normalization

Required:

- Preserve original image.
- Normalize orientation using EXIF.
- Store normalized copy.
- Compute SHA-256 hash.
- Compute perceptual hash for duplicate/near-duplicate detection.
- Store width, height, color mode, EXIF timestamp, GPS metadata if available and policy allows.

### 10.4 Region proposal

You need candidate visual units. Use a staged approach.

#### V1: dense patch grid

- Patch size: 224×224 or 336×336.
- Stride: 50% overlap, for example 112 for 224 patches.
- Simple and robust for early implementation.
- Good for discovering recurring patterns.

Region schema:

```json
{
  "region_id": "img_001_patch_00042",
  "image_id": "img_001",
  "source_id": "img_001",
  "region_type": "patch",
  "bbox": [448, 224, 224, 224],
  "crop_path": "work/crops/img_001_patch_00042.jpg",
  "mask_path": null,
  "proposal_method": "grid_224_stride112",
  "proposal_score": null,
  "risk_flags": []
}
```

#### V2: segmentation proposals

Use segmentation models for object-like regions.

```json
{
  "region_id": "img_001_r003",
  "image_id": "img_001",
  "region_type": "segmentation_mask",
  "bbox": [120, 40, 30, 260],
  "mask_path": "work/masks/img_001_r003.png",
  "crop_path": "work/crops/img_001_r003.png",
  "proposal_method": "sam2",
  "proposal_score": 0.91,
  "risk_flags": ["thin_region"]
}
```

#### V3: detector proposals

Use detector proposals for known categories.

Examples:

- open-vocabulary detector for known labels,
- domain detector trained for domain entities,
- OCR text boxes,
- face/person/object detectors if permitted by policy.

### 10.5 Visual embeddings

Compute embeddings for each region/crop.

Recommended split:

- DINOv2-like embeddings for unnamed visual similarity and feature discovery.
- CLIP/SigLIP/OpenCLIP-like embeddings for language-aligned label scoring.

Embedding record:

```json
{
  "region_id": "img_001_r003",
  "image_id": "img_001",
  "embedding_model": "dinov2_vitl14",
  "embedding_path": "work/vectors/dinov2/img_001_r003.npy",
  "embedding_dim": 1024,
  "preprocessing": {
    "crop_resize": 224,
    "normalize": true
  }
}
```

### 10.6 Visual feature clustering

For unnamed feature discovery:

```text
region embeddings
  → nearest-neighbor graph
  → thresholding or HDBSCAN
  → feature clusters
  → representative crops
```

Start with either:

- FAISS nearest neighbors + thresholded connected components, or
- HDBSCAN on DINOv2 embeddings.

Cluster schema:

```json
{
  "feature_cluster_id": "vf_017",
  "embedding_model": "dinov2_vitl14",
  "clustering_method": "hdbscan",
  "member_region_ids": ["img_001_r003", "img_002_r014", "img_007_r002"],
  "cluster_size": 12,
  "cohesion_score": 0.81,
  "nearest_neighbor_margin": 0.18,
  "representative_region_ids": ["img_001_r003", "img_002_r014"],
  "status": "unnamed",
  "risk_flags": []
}
```

### 10.7 Image claim types

#### Type A — named visual classification

```json
{
  "claim_id": "claim_img_001",
  "claim_type": "named_visual_classification",
  "source_modality": "image",
  "source_id": "img_001",
  "evidence_id": "ev_img_001_r003",
  "region_id": "img_001_r003",
  "source_faithful_claim": "Model vlm_x classified region img_001_r003 as a mast.",
  "subject": "img_001_r003",
  "predicate": "classified_as",
  "object": "mast",
  "classifier": {
    "model": "vlm_x",
    "confidence": 0.82,
    "basis": "crop_only"
  },
  "truth_status": "model_observation_unverified",
  "support_status": "raw_extracted",
  "risk_flags": []
}
```

#### Type B — unnamed visual feature cluster

```json
{
  "claim_id": "claim_vf_017",
  "claim_type": "unnamed_visual_feature_cluster",
  "source_modality": "image",
  "source_id": "image_collection_001",
  "evidence_id": "ev_vf_017",
  "feature_cluster_id": "vf_017",
  "source_faithful_claim": "Regions img_001_r003, img_002_r014, and img_007_r002 were clustered as visually similar under dinov2_vitl14.",
  "subject": "vf_017",
  "predicate": "has_member_regions",
  "object": ["img_001_r003", "img_002_r014", "img_007_r002"],
  "discovery_method": {
    "embedding_model": "dinov2_vitl14",
    "clustering_method": "hdbscan",
    "cohesion_score": 0.81
  },
  "truth_status": "model_observation_unverified",
  "support_status": "raw_extracted"
}
```

#### Type C — OCR text claim from image

If an image contains text, create OCR evidence and route it through the text claim extractor.

```json
{
  "evidence_id": "ev_img_001_ocr_004",
  "source_modality": "image",
  "source_id": "img_001",
  "evidence_type": "ocr_text_span",
  "text": "Engine replaced 2024",
  "provenance": {
    "image_id": "img_001",
    "bbox": [910, 640, 220, 40],
    "ocr_model": "paddleocr|tesseract|cloud_ocr",
    "ocr_confidence": 0.88
  },
  "risk_flags": []
}
```

OCR-derived claims must include OCR confidence risk and must not be treated as cleaner than the OCR evidence.

### 10.8 Image validation

There is no exact substring check for visual classification. Use visual validation policies.

For named classifications, require one or more of:

- high classifier/VLM confidence,
- agreement between multiple models,
- detector and VLM agreement,
- stable label under crop jitter,
- label consistency across cluster members,
- human confirmation.

For unnamed clusters, validate by:

- cluster size,
- intra-cluster similarity,
- nearest-neighbor margin,
- cross-image recurrence,
- robustness under crop jitter,
- avoiding trivial artifacts.

Risk flags:

```json
[
  "low_resolution",
  "partial_occlusion",
  "background_artifact",
  "same_image_duplicates",
  "near_duplicate_images",
  "weak_cluster_margin",
  "crop_contains_multiple_objects",
  "vlm_label_low_confidence",
  "ocr_low_confidence"
]
```

Quarantine examples:

```yaml
image_validation:
  named_visual_classification:
    accept_if:
      human_reviewed: true
    needs_review_if:
      classifier_confidence_lt: 0.85
      no_model_agreement: true
    reject_if:
      region_area_ratio_lt: 0.001
      region_blur_score_lt: threshold
  unnamed_visual_feature_cluster:
    accept_if:
      cluster_size_gte: 5
      cohesion_score_gte: 0.75
      cross_source_images_gte: 3
    needs_review_if:
      weak_cluster_margin: true
      mostly_same_image_duplicates: true
```

---

## 11. Claim extraction prompts

### 11.1 Core text extraction prompt

```text
You extract attributed claims from source evidence.

Rules:
- Extract claims made in the source; do not verify them.
- Preserve provenance by quoting exact evidence_text.
- Split compound descriptions into atomic claims.
- Preserve uncertainty words: seems, maybe, likely, allegedly, appears, may.
- Preserve negation.
- Preserve quantities, units, dates, comparisons, and temporal markers.
- Do not infer beyond the target evidence and explicitly provided context.
- Do not introduce entities absent from the target evidence or explicit local context.
- If the text says "I saw X", represent it as an observation by the speaker/source.
- If the source reports someone else's claim, preserve that nested attribution.
- Return JSON only.
```

### 11.2 Core output schema

```json
{
  "claims": [
    {
      "claim_text": "string",
      "source_faithful_claim": "string",
      "subject": "string",
      "predicate": "string",
      "object": "string|array|object|null",
      "quantity": "number|null",
      "attributes": "object|null",
      "modality": "asserted|direct_observation|uncertain_observation|reported|reported_direct_observation|reported_uncertain|negated|hypothetical|question_asked|model_observation",
      "confidence": "number from 0 to 1",
      "evidence_text": "exact quote from target evidence",
      "context_dependent": "boolean",
      "context_used": "string|null",
      "attribution": {
        "type": "speaker|document|model|human_reviewer|unknown",
        "agent": "string|null"
      }
    }
  ]
}
```

### 11.3 Validation prompt

Use this only after deterministic checks or for high-value claims. Do not rely on it as the only validator.

```text
You validate whether a proposed claim is supported by its evidence.

Return JSON only.

Check:
- Is evidence_text an exact excerpt of the evidence? If not, invalid.
- Is the claim atomic?
- Is the claim entailed by the evidence as a source-faithful statement?
- Does the claim preserve attribution?
- Does the claim preserve uncertainty and modality?
- Does the claim preserve negation?
- Does the claim preserve quantities, units, dates, and comparisons?
- Did the claim introduce unsupported entities?
- Should the claim be accepted, repaired, or quarantined?

Do not verify whether the claim is true in the world.
```

### 11.4 Schema repair prompt

```text
Repair the following JSON so it conforms exactly to the provided schema.
Do not add new claims.
Do not change evidence_text.
Do not invent missing evidence.
If a field cannot be repaired from the input, set it to null and add a repair warning.
Return JSON only.
```

---

## 12. Validation architecture

### 12.1 Validation status machine

```text
raw_extracted
  → schema_valid
  → deterministic_valid
  → semantic_valid
  → accepted_extracted
  → normalized
  → exported
```

Failure routes:

```text
raw_extracted
  → schema_invalid
  → repair_attempted
  → repaired or quarantined

raw_extracted
  → evidence_invalid
  → repair_attempted or quarantined

raw_extracted
  → modality_risky
  → needs_review or quarantined
```

### 12.2 Deterministic validator pseudocode

```python
def validate_text_claim(claim, span):
    errors = []
    warnings = []

    if not claim.evidence_text:
        errors.append("missing_evidence_text")
    elif claim.evidence_text not in span.text:
        errors.append("evidence_not_exact_substring")

    if claim.confidence is None or not (0 <= claim.confidence <= 1):
        errors.append("invalid_confidence")

    if claim.modality not in ALLOWED_MODALITIES:
        errors.append("invalid_modality")

    if contains_negation(span.text) and not preserves_negation(claim):
        errors.append("negation_dropped")

    if contains_hedge(span.text) and not preserves_uncertainty(claim):
        errors.append("uncertainty_dropped")

    if quantities(span.text) != quantities_or_subset(claim):
        errors.append("quantity_mismatch")

    introduced = unsupported_entities(claim, span.text, claim.context_used)
    if introduced:
        errors.append("unsupported_entities_introduced")

    if errors:
        return ValidationResult(status="quarantined", errors=errors, warnings=warnings)

    return ValidationResult(status="accepted_extracted", errors=[], warnings=warnings)
```

### 12.3 Evidence exact-match strategy

Use strict matching first:

```python
claim.evidence_text in span.text
```

If strict matching fails:

1. Normalize whitespace and retry.
2. Normalize Unicode quotes and dashes and retry.
3. Fuzzy locate only as a repair suggestion.
4. Never silently accept fuzzy evidence.
5. Store repaired evidence only if the repaired text is an exact substring after repair.

### 12.4 Entity introduction check

Approximate v1:

1. Extract capitalized named entities from evidence and context.
2. Extract named entities from claim.
3. Allow pronouns and generic roles.
4. Flag entities that appear only in the claim.

This should be a warning first, then tightened after reviewing false positives.

### 12.5 Modality and uncertainty checks

Maintain lexicons:

```yaml
uncertainty_markers:
  - appears
  - seems
  - may
  - might
  - likely
  - allegedly
  - reportedly
  - possibly
  - suggests

negation_markers:
  - not
  - no
  - never
  - without
  - neither
  - failed to
  - lacks
```

If evidence contains uncertainty or negation markers and the claim omits uncertainty or negation, quarantine or mark `needs_review`.

### 12.6 Quantity preservation

Extract numbers, units, dates, percentages, durations, and ranges from evidence and claim. Require claim quantities to be equal to or a subset of evidence quantities.

Examples:

```text
Evidence: "The engine was replaced in 2024."
Claim: "The engine was replaced in 2023."
Status: reject, quantity/date mismatch.
```

```text
Evidence: "The vessel had three masts and two engines."
Claim A: "The vessel had three masts."
Claim B: "The vessel had two engines."
Status: accept if both evidence_text values are exact or the full sentence supports both atomic claims.
```

---

## 13. Normalization layer

### 13.1 Purpose

Normalization converts accepted source-faithful claims into stable semantic forms for retrieval, graph reasoning, or AtomSpace ingestion.

It must never replace the source-faithful claim. It creates a derived record.

### 13.2 Normalized claim schema

```json
{
  "normalized_claim_id": "nclaim_001",
  "claim_id": "claim_001",
  "source_id": "src_001",
  "evidence_id": "ev_001",
  "normalized_claim": {
    "subject": "entity:vessel_hope",
    "predicate": "appears_condition",
    "object": "condition:older_diesel_engine",
    "qualifiers": {
      "modality": "uncertain_observation",
      "attribution": "source:report_001",
      "truth_status": "source_asserted_unverified"
    }
  },
  "normalization": {
    "entity_resolution": [
      {
        "surface": "vessel Hope",
        "canonical_id": "entity:vessel_hope",
        "confidence": 0.88,
        "basis": "exact_name_match"
      }
    ],
    "predicate_mapping": {
      "surface": "appears_to_have_condition",
      "canonical": "appears_condition"
    }
  },
  "schema_version": "claim.normalized.v1"
}
```

### 13.3 Predicate registry

Create a controlled predicate registry.

Example:

```yaml
predicates:
  asserts:
    description: Speaker or source asserts proposition.
  reports_observation:
    description: Speaker reports direct observation.
  classified_as:
    description: Model or reviewer labels a visual region.
  belongs_to_visual_cluster:
    description: Region belongs to feature cluster.
  appears_condition:
    description: Source expresses uncertain condition.
  has_quantity:
    description: Subject has measured quantity.
  negates:
    description: Source denies proposition.
```

### 13.4 Deduplication

Deduplicate only after validation.

Deduping levels:

1. Exact duplicate evidence and claim.
2. Same source span, same normalized claim.
3. Different evidence, same normalized proposition.
4. Cross-source corroboration candidate.

Do not collapse cross-source claims into one fact. Preserve each source assertion and create a separate aggregation record.

---

## 14. Storage strategy

### 14.1 JSONL first

JSONL should be the canonical v1 artifact format because it is:

- append-friendly,
- resumable,
- easy to diff,
- easy to validate,
- easy to stream,
- compatible with batch processing.

Canonical files:

```text
sources.jsonl
evidence.jsonl
chunks.jsonl
spans.jsonl
claims.raw.jsonl
validations.jsonl
claims.validated.jsonl
claims.normalized.jsonl
errors.jsonl
quarantine.jsonl
```

### 14.2 SQL later

Use PostgreSQL when you need:

- multi-user review,
- job orchestration,
- queryable status,
- complex joins,
- incremental processing,
- audit logs.

Core tables:

```text
sources
evidence
chunks
spans
claims_raw
validations
claims_validated
claims_normalized
jobs
artifacts
review_decisions
identity_mappings
```

### 14.3 Object storage

Store large artifacts outside SQL:

```text
normalized audio
image crops
image masks
embeddings
OCR page images
PDF page renders
```

Use content-addressed paths where practical:

```text
work/crops/sha256_prefix/region_id.png
work/vectors/model_name/region_id.npy
```

---

## 15. CLI interface

Start with a resumable CLI.

```bash
# Register sources
python -m evidence_pipeline ingest-chat data/raw/chat/export.json
python -m evidence_pipeline ingest-pdf data/raw/pdf/report.pdf
python -m evidence_pipeline ingest-audio data/raw/audio/meeting.mp3
python -m evidence_pipeline ingest-images data/raw/images/

# Build evidence and chunks
python -m evidence_pipeline build-evidence --modality all
python -m evidence_pipeline chunk --modality chat
python -m evidence_pipeline chunk --modality pdf
python -m evidence_pipeline chunk --modality audio
python -m evidence_pipeline propose-image-regions

# Detect spans or visual regions
python -m evidence_pipeline detect-spans --modality chat
python -m evidence_pipeline detect-spans --modality pdf
python -m evidence_pipeline detect-spans --modality audio
python -m evidence_pipeline cluster-image-regions

# Extract and validate
python -m evidence_pipeline extract-claims --modality all --model cheap
python -m evidence_pipeline validate-claims
python -m evidence_pipeline repair-claims --only evidence_not_exact_substring
python -m evidence_pipeline normalize-claims

# Review and report
python -m evidence_pipeline report
python -m evidence_pipeline export-graph --format jsonl
```

Every command should be idempotent. If a record with the same source hash, stage, config hash, and model version already exists, skip or require `--force`.

---

## 16. Job orchestration and idempotency

### 16.1 Job record

```json
{
  "job_id": "job_001",
  "stage": "extract_claims",
  "source_id": "pdf_001",
  "input_record_ids": ["span_001", "span_002"],
  "config_hash": "...",
  "model_hash": "...",
  "prompt_hash": "...",
  "status": "pending|running|succeeded|failed|skipped",
  "attempts": 0,
  "created_at": "...",
  "updated_at": "...",
  "error": null
}
```

### 16.2 Idempotency key

Use:

```text
stage + source_id + input_record_id + config_hash + model_id + prompt_hash
```

This prevents accidental duplicate extraction when rerunning the pipeline.

### 16.3 Model routing

Use a cheap model for normal extraction and a stronger model for:

- schema repair failures,
- low-confidence claims,
- high-value sources,
- ambiguous claims,
- validation disagreements,
- human-selected review batches.

Example config:

```yaml
models:
  extraction_default: cheap_structured_json_model
  extraction_strong: strong_reasoning_model
  validation_default: cheap_validator_model
  validation_strong: strong_validator_model

routing:
  use_strong_extractor_if:
    span_score_lt: 0.60
    source_priority: high
    previous_schema_failure: true
  use_strong_validator_if:
    raw_claim_confidence_lt: 0.65
    modality: image
    risk_flags_any:
      - context_dependent_coreference
      - speaker_uncertain
      - weak_cluster_margin
```

---

## 17. Evaluation and QA

### 17.1 Gold set

Build a small manually annotated set before scaling.

Recommended initial gold set:

| Modality | Gold set size |
|---|---:|
| Chat | 50–100 messages across several threads |
| PDF | 20–50 chunks, including tables and uncertain claims |
| Audio | 20–50 utterance chunks with speaker labels |
| Images | 100–300 regions/clusters with human labels or review decisions |

### 17.2 Metrics

Track:

- claim precision,
- claim recall,
- evidence exact-match rate,
- attribution correctness,
- uncertainty preservation,
- negation preservation,
- quantity preservation,
- unsupported entity rate,
- duplicate rate,
- quarantine rate,
- repair success rate,
- review disagreement rate,
- cost per accepted claim,
- latency per source.

### 17.3 Review UI requirements

A minimal review interface should show:

- source view,
- highlighted evidence span or region crop,
- raw claim,
- validation result,
- risk flags,
- normalized claim,
- accept/reject/edit controls,
- reason code selection,
- reviewer notes.

For PDFs, show page and bounding box. For audio, provide playable clip from start/end timestamps. For images, show crop, full image context, and cluster representatives.

---

## 18. Security, privacy, and auditability

### 18.1 Audit trail

Every derived record should include:

- source ID,
- input record IDs,
- model ID,
- prompt version,
- config hash,
- validation version,
- timestamp,
- status,
- error or warning reasons.

### 18.2 Privacy

Implement:

- PII detection in chat and transcripts,
- optional redaction before external LLM calls,
- local-only mode for sensitive sources,
- encrypted storage for raw sources,
- configurable retention policies,
- audit logs for human review actions.

### 18.3 Data lineage

Lineage must answer:

```text
Which source file/message/audio segment/image generated this claim?
Which evidence span or visual region supports it?
Which model extracted it?
Which validator accepted it?
Which normalizer transformed it?
Who reviewed it, if anyone?
```

---

## 19. Implementation roadmap

### Milestone 0 — Foundation

Deliverables:

- Repository structure.
- Pydantic schemas.
- JSONL append/read utilities.
- Stable ID generation.
- Config loader.
- Basic CLI skeleton.
- Logging.

Acceptance criteria:

- Can create valid `sources.jsonl`, `evidence.jsonl`, and `errors.jsonl`.
- Schema validation runs from CLI.
- All records include schema version and source IDs.

### Milestone 1 — Chat pipeline

Deliverables:

- Chat export ingestor.
- Message evidence generator.
- Thread-aware chunker.
- Rule-based span detector.
- Chat claim extraction prompt.
- Deterministic validator.

Acceptance criteria:

- Process 100 chat messages end-to-end.
- Accepted claims have exact `evidence_text` from message spans.
- Sender attribution is preserved.
- Context-dependent claims are flagged.

### Milestone 2 — PDF pipeline

Deliverables:

- PyMuPDF extractor.
- PDF block schema.
- Header/footer cleanup.
- Section/paragraph chunker.
- PDF span detector.
- PDF claim prompt.
- PDF validation.

Acceptance criteria:

- Process 3 PDFs end-to-end.
- Every accepted claim has `source_id`, page, block ID, and exact evidence text.
- Claims with missing evidence substrings are quarantined.
- Extraction summary report is generated.

### Milestone 3 — Shared extraction and validation hardening

Deliverables:

- LLM provider abstraction.
- Batch extraction.
- Schema repair.
- Validator result records.
- Negation/uncertainty/quantity preservation checks.
- Gold set evaluation harness.

Acceptance criteria:

- Evidence exact-match rate for accepted text claims is 100%.
- Validation reason codes are recorded for every rejected claim.
- Evaluation report computes precision and recall on gold set.

### Milestone 4 — Audio pipeline

Deliverables:

- Audio normalization.
- ASR integration.
- Optional diarization integration.
- ASR/diarization alignment.
- Utterance evidence generation.
- Audio chunker.
- Audio claim prompt.
- Audio-specific validator.

Acceptance criteria:

- Process one 30–60 minute audio file.
- Claims include speaker labels and timestamps.
- Low ASR/diarization confidence propagates to risk flags.
- Overlapping speech claims are flagged or quarantined.

### Milestone 5 — Image evidence and claims

Deliverables:

- Image ingestion and normalization.
- Patch region proposal.
- Crop persistence.
- DINOv2/OpenCLIP embedding integration.
- Region clustering.
- Visual claim emission for unnamed clusters.
- Optional VLM cluster label prompt.
- Image validation policies.

Acceptance criteria:

- Process a folder of images.
- Emit region evidence records.
- Emit cluster claims with embedding/clustering provenance.
- Named visual labels are marked as hypotheses unless validated.
- Human review can accept/reject labels.

### Milestone 6 — Normalization and graph export

Deliverables:

- Entity canonicalization.
- Predicate registry.
- Normalized claim records.
- Deduplication.
- JSON graph export.
- Optional AtomSpace/MeTTa export.

Acceptance criteria:

- Every normalized claim links to a validated source-faithful claim.
- Cross-source duplicate claims are grouped but not collapsed into unsupported facts.
- Export preserves attribution and truth status.

---

## 20. Testing strategy

### 20.1 Unit tests

Test:

- schema parsing,
- ID stability,
- JSONL append/read,
- exact evidence matching,
- whitespace normalization repair,
- quantity extraction,
- negation detection,
- chunk boundary logic,
- audio timestamp bounds,
- image bbox bounds.

### 20.2 Integration tests

Create fixtures:

```text
tests/fixtures/chat/simple_thread.json
tests/fixtures/pdf/simple_report.pdf
tests/fixtures/audio/simple_meeting.wav
tests/fixtures/images/simple_boat.jpg
```

Each fixture should have expected outputs.

### 20.3 Regression tests

Keep rejected examples. Every hallucinated, unsupported, or wrongly attributed claim should become a regression test.

---

## 21. Key risks and mitigations

| Risk | Where | Mitigation |
|---|---|---|
| LLM invents evidence | all text-like modalities | exact substring validation |
| LLM converts speaker assertion into fact | chat/audio | mandatory attribution checks |
| Uncertainty dropped | all | uncertainty marker validator |
| Negation dropped | all | negation marker validator |
| PDF layout corrupts reading order | PDF | block coordinates, section detection, manual QA |
| OCR errors become claims | PDF/image OCR | OCR confidence flags and quarantine policy |
| ASR hallucination | audio | VAD, no-speech thresholds, confidence checks |
| Wrong speaker assigned | audio | diarization confidence, overlap flags, human correction |
| Context-only entity introduced | chat/audio | context-dependent flag and entity introduction check |
| Image model confabulates label | image | store as hypothesis, require agreement or human review |
| Visual clusters capture artifacts | image | filter borders/backgrounds, require cross-source recurrence |
| Duplicate explosion | all | post-validation dedupe |
| Cost explosion | all | span detector before extraction, cheap/strong model routing |

---

## 22. Definition of done

A source is fully processed when:

1. Source is registered with hash and metadata.
2. Modality-specific evidence records are created.
3. Context chunks are created and linked to evidence.
4. Claim-bearing spans or regions are detected.
5. Raw claims are extracted with strict schema.
6. Deterministic validation has run.
7. Invalid claims are quarantined with reasons.
8. Accepted claims have complete provenance.
9. Accepted claims preserve attribution, negation, uncertainty, and quantities.
10. Normalized claims are derived only from accepted claims.
11. Reports summarize counts, failures, and quality metrics.
12. Re-running the same stage does not duplicate records.

---

## 23. Practical v1 build order

Recommended order:

```text
1. Core schemas + JSONL + CLI
2. Chat pipeline
3. PDF pipeline
4. Shared text claim extraction + validation
5. QA report and gold set evaluation
6. Audio pipeline
7. Image region evidence + clustering
8. Image label hypotheses + human review
9. Normalized claim graph export
```

Reasoning:

- Chat is the simplest text substrate and validates the attribution model early.
- PDF exercises layout provenance and exact evidence span validation.
- Audio reuses chat attribution patterns but adds ASR and diarization risk.
- Images require a different evidence substrate, so implement after the text validation core is stable.

---

## 24. Minimal working example target

A successful first demo should process:

```text
10 chat messages
3 PDFs
1 audio conversation
20 images
```

and produce:

```text
data/jsonl/sources.jsonl
data/jsonl/evidence.jsonl
data/jsonl/chunks.jsonl
data/jsonl/spans.jsonl
data/jsonl/claims.raw.jsonl
data/jsonl/claims.validated.jsonl
data/jsonl/claims.normalized.jsonl
data/jsonl/quarantine.jsonl
data/reports/extraction_summary.md
```

Minimum acceptance thresholds:

- 100% of accepted text-like claims have exact evidence substrings.
- 100% of accepted chat/audio claims have speaker/sender attribution.
- 100% of accepted PDF claims have page provenance.
- 100% of image claims are marked as model observations, hypotheses, or human-reviewed labels.
- No raw image model label is promoted to world truth by default.
- Every quarantined claim has at least one machine-readable reason code.

---

## 25. Tooling references checked for this plan

These references support tool choices and terminology. The pipeline should still pin exact package versions during implementation.

- PyMuPDF text block and bounding box extraction: https://pymupdf.readthedocs.io/en/latest/app1.html
- GROBID PDF-to-structured-TEI principles: https://grobid.readthedocs.io/en/latest/Principles/
- Unstructured partitioning into document elements: https://docs.unstructured.io/open-source/core-functionality/partitioning
- WhisperX word-level timestamps and diarization: https://github.com/m-bain/whisperx
- pyannote.audio speaker diarization toolkit: https://github.com/pyannote/pyannote-audio
- Segment Anything Model 2 for image/video segmentation: https://ai.meta.com/research/sam2/
- DINOv2 visual feature embeddings: https://github.com/facebookresearch/dinov2
- HDBSCAN clustering documentation: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html

---

## 26. Final architectural recommendation

Build the system around the following invariant:

```text
No claim without evidence.
No evidence without provenance.
No normalized atom without a validated source-faithful claim.
No model output promoted to truth without explicit validation policy.
```

For text-like modalities, the most important engineering feature is exact evidence substring validation. For audio, the most important feature is timestamped speaker attribution with confidence propagation. For images, the most important feature is avoiding naked visual facts: store visual model outputs as observations, clusters, or reviewable hypotheses.

This architecture gives you a durable path from messy multimodal sources to auditable claims and eventually to graph or reasoning-system ingestion.
