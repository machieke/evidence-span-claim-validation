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
PYTHONPATH=src python3 -m evidence_pipeline validate-artifacts
```

During local development:

```bash
python3 -m pytest
```
