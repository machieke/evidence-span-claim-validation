# Evidence Span Claim Validation

This repository implements an evidence-span -> claim -> validation -> normalization pipeline for auditable multimodal knowledge extraction.

The current branch covers the full deterministic scaffold:

- Multimodal ingestion for chat, PDFs, audio transcripts/media registration, images, and image OCR.
- Durable evidence records with provenance for messages, PDF blocks, utterances, visual regions, visual clusters, and OCR spans.
- Context chunking and rule-based claim-bearing span detection.
- Source-faithful claim extraction through a provider-neutral JSON extraction adapter.
- Deterministic validation for exact evidence support, provenance, attribution, negation, uncertainty, quantities, timestamps, image labels, and review outcomes.
- Repair suggestions and audited repair application for evidence text mismatches.
- Human review decisions and review queue exports, including HTML previews for PDF pages, audio clips, image crops, and cluster representatives.
- Normalized claims, duplicate groups, JSON graph export, SQLite export, and MeTTa export.
- Confidence propagation from raw claims through validation, normalization, graph edges, duplicate groups, and MeTTa confidence atoms.
- Acceptance checks, gold evaluation, trace reports, PII/privacy/retention helpers, and run/finalize commands.

## Quick Start

```bash
PYTHONPATH=src python3 -m evidence_pipeline init
PYTHONPATH=src python3 -m evidence_pipeline register-source data/raw/chat/export.json --modality chat
PYTHONPATH=src python3 -m evidence_pipeline validate-artifacts
```

Demo acceptance dataset:

```bash
PYTHONPATH=src python3 -m evidence_pipeline seed-demo-artifacts
PYTHONPATH=src python3 -m evidence_pipeline finalize-run --gold data/reports/demo_gold.json
```

Chat pipeline:

```bash
PYTHONPATH=src python3 -m evidence_pipeline ingest-chat data/raw/chat/export.json
PYTHONPATH=src python3 -m evidence_pipeline build-chat-evidence
PYTHONPATH=src python3 -m evidence_pipeline chunk-chat
PYTHONPATH=src python3 -m evidence_pipeline detect-chat-spans
PYTHONPATH=src python3 -m evidence_pipeline route-models --stage extraction
PYTHONPATH=src python3 -m evidence_pipeline extract-claims --modality chat
PYTHONPATH=src python3 -m evidence_pipeline extract-claims --modality chat --batch-size 50
PYTHONPATH=src python3 -m evidence_pipeline validate-claims
PYTHONPATH=src python3 -m evidence_pipeline detect-pii
PYTHONPATH=src python3 -m evidence_pipeline redact-pii --artifact chat_messages
PYTHONPATH=src python3 -m evidence_pipeline check-privacy
PYTHONPATH=src python3 -m evidence_pipeline retention-plan
PYTHONPATH=src python3 -m evidence_pipeline repair-claims
PYTHONPATH=src python3 -m evidence_pipeline apply-repairs
PYTHONPATH=src python3 -m evidence_pipeline normalize-claims
PYTHONPATH=src python3 -m evidence_pipeline export-graph
PYTHONPATH=src python3 -m evidence_pipeline export-sqlite
PYTHONPATH=src python3 -m evidence_pipeline export-metta
PYTHONPATH=src python3 -m evidence_pipeline dedupe-claims
PYTHONPATH=src python3 -m evidence_pipeline report
PYTHONPATH=src python3 -m evidence_pipeline report --format html
PYTHONPATH=src python3 -m evidence_pipeline acceptance-check
PYTHONPATH=src python3 -m evidence_pipeline finalize-run
PYTHONPATH=src python3 -m evidence_pipeline review-claim claim_... --decision accept --reviewer-id reviewer_1
PYTHONPATH=src python3 -m evidence_pipeline review-queue
PYTHONPATH=src python3 -m evidence_pipeline review-queue --format html
PYTHONPATH=src python3 -m evidence_pipeline eval-gold tests/fixtures/gold/chat_claims.json
PYTHONPATH=src python3 -m evidence_pipeline trace-claim claim_...
PYTHONPATH=src python3 -m evidence_pipeline trace-claim claim_... --format html
PYTHONPATH=src python3 -m evidence_pipeline validate-artifacts --include-reports
```

PDF pipeline:

```bash
PYTHONPATH=src python3 -m evidence_pipeline ingest-pdf data/raw/pdf/report.pdf
PYTHONPATH=src python3 -m evidence_pipeline build-pdf-evidence
PYTHONPATH=src python3 -m evidence_pipeline chunk-pdf
PYTHONPATH=src python3 -m evidence_pipeline detect-pdf-spans
PYTHONPATH=src python3 -m evidence_pipeline extract-claims --modality pdf
PYTHONPATH=src python3 -m evidence_pipeline validate-claims
PYTHONPATH=src python3 -m evidence_pipeline normalize-claims
PYTHONPATH=src python3 -m evidence_pipeline export-graph
PYTHONPATH=src python3 -m evidence_pipeline report
PYTHONPATH=src python3 -m evidence_pipeline validate-artifacts
```

Audio transcript pipeline:

```bash
PYTHONPATH=src python3 -m evidence_pipeline normalize-audio data/raw/audio/meeting.mp3
PYTHONPATH=src python3 -m evidence_pipeline normalize-audio data/raw/audio/meeting.mp3 --execute
PYTHONPATH=src python3 -m evidence_pipeline ingest-audio-transcript data/raw/audio/transcript.json
PYTHONPATH=src python3 -m evidence_pipeline build-audio-evidence
PYTHONPATH=src python3 -m evidence_pipeline chunk-audio
PYTHONPATH=src python3 -m evidence_pipeline detect-audio-spans
PYTHONPATH=src python3 -m evidence_pipeline extract-claims --modality audio
PYTHONPATH=src python3 -m evidence_pipeline validate-claims
PYTHONPATH=src python3 -m evidence_pipeline normalize-claims
PYTHONPATH=src python3 -m evidence_pipeline export-graph
PYTHONPATH=src python3 -m evidence_pipeline report
PYTHONPATH=src python3 -m evidence_pipeline validate-artifacts
```

Image evidence pipeline:

```bash
PYTHONPATH=src python3 -m evidence_pipeline ingest-images data/raw/images/
PYTHONPATH=src python3 -m evidence_pipeline ingest-image-ocr data/raw/images/ocr.json
PYTHONPATH=src python3 -m evidence_pipeline chunk-image-ocr
PYTHONPATH=src python3 -m evidence_pipeline detect-image-ocr-spans
PYTHONPATH=src python3 -m evidence_pipeline propose-image-regions --patch-size 224 --stride 112
PYTHONPATH=src python3 -m evidence_pipeline build-image-evidence
PYTHONPATH=src python3 -m evidence_pipeline embed-image-regions
PYTHONPATH=src python3 -m evidence_pipeline classify-image-regions
PYTHONPATH=src python3 -m evidence_pipeline cluster-image-regions
PYTHONPATH=src python3 -m evidence_pipeline build-image-cluster-evidence
PYTHONPATH=src python3 -m evidence_pipeline extract-claims --modality image
PYTHONPATH=src python3 -m evidence_pipeline validate-claims
PYTHONPATH=src python3 -m evidence_pipeline normalize-claims
PYTHONPATH=src python3 -m evidence_pipeline export-graph
PYTHONPATH=src python3 -m evidence_pipeline report
PYTHONPATH=src python3 -m evidence_pipeline validate-artifacts
```

During local development:

```bash
python3 -m pytest
```

Convenience runners:

```bash
PYTHONPATH=src python3 -m evidence_pipeline run-chat data/raw/chat/export.json
PYTHONPATH=src python3 -m evidence_pipeline run-pdf data/raw/pdf/report.pdf
PYTHONPATH=src python3 -m evidence_pipeline run-audio-transcript data/raw/audio/transcript.json
PYTHONPATH=src python3 -m evidence_pipeline run-images data/raw/images/
PYTHONPATH=src python3 -m evidence_pipeline finalize-run
```

Generic stage commands:

```bash
PYTHONPATH=src python3 -m evidence_pipeline build-evidence --modality all
PYTHONPATH=src python3 -m evidence_pipeline chunk --modality chat
PYTHONPATH=src python3 -m evidence_pipeline detect-spans --modality chat
```

## Operator Workflow

Use the pipeline as an auditable promotion path:

1. Ingest or register sources.
2. Build evidence records with source-specific provenance.
3. Build chunks and detect claim-bearing spans.
4. Extract raw source-faithful claims.
5. Validate claims before normalization or export.
6. Repair evidence text issues only through `repair-claims` and `apply-repairs`.
7. Use `review-queue` and `review-claim` for quarantined or risky claims.
8. Normalize only accepted claims.
9. Export graph, SQLite, or MeTTa outputs.
10. Run `acceptance-check`, `report`, `trace-claim`, and `validate-artifacts --include-reports`.

The key invariant is that exported claims are not free-floating facts. They retain source IDs, evidence IDs, attribution, truth status, source-faithful text, confidence, and validation metadata.

## Confidence And MeTTa

Raw claim confidence is preserved into validated claims, normalized qualifiers, JSON graph edges, duplicate groups, and MeTTa export.

MeTTa output includes first-class confidence expressions:

```lisp
(claim-confidence "nclaim_..." 0.82)
(claim-confidence-basis "nclaim_..." "validated_claim_confidence")
```

Validation still acts as a gate: quarantined claims do not normalize or export. Accepted claims carry their confidence forward for downstream revision or reasoning.
