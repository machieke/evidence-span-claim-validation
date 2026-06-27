# Evidence Span Claim Validation

This repository implements the foundation for an evidence-span to claim to validation pipeline.

The initial slice covers Milestone 0 from the implementation plan:

- Pydantic schemas for sources, evidence, chunks, spans, claims, and validations.
- Stable ID helpers.
- JSONL read, append, and validation utilities.
- Basic configuration loading.
- A Typer CLI for initializing artifacts, registering sources, and validating JSONL files.

## Quick Start

```bash
PYTHONPATH=src python3 -m evidence_pipeline init
PYTHONPATH=src python3 -m evidence_pipeline register-source data/raw/chat/export.json --modality chat
PYTHONPATH=src python3 -m evidence_pipeline validate-artifacts
```

Chat pipeline:

```bash
PYTHONPATH=src python3 -m evidence_pipeline ingest-chat data/raw/chat/export.json
PYTHONPATH=src python3 -m evidence_pipeline build-chat-evidence
PYTHONPATH=src python3 -m evidence_pipeline chunk-chat
PYTHONPATH=src python3 -m evidence_pipeline detect-chat-spans
PYTHONPATH=src python3 -m evidence_pipeline extract-claims --modality chat
PYTHONPATH=src python3 -m evidence_pipeline validate-claims
PYTHONPATH=src python3 -m evidence_pipeline normalize-claims
PYTHONPATH=src python3 -m evidence_pipeline export-graph
PYTHONPATH=src python3 -m evidence_pipeline report
PYTHONPATH=src python3 -m evidence_pipeline eval-gold tests/fixtures/gold/chat_claims.json
PYTHONPATH=src python3 -m evidence_pipeline validate-artifacts
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
PYTHONPATH=src python3 -m evidence_pipeline propose-image-regions --patch-size 224 --stride 112
PYTHONPATH=src python3 -m evidence_pipeline build-image-evidence
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
```
