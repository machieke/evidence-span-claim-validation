from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.ingest.pdf import (
    _ExtractedPDFBlock,
    classify_repeated_pdf_furniture,
    clean_pdf_text,
    infer_pdf_section_paths,
)
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.pdf import PDFBlockRecord


runner = CliRunner()


def _seed_pdf_blocks() -> None:
    blocks = [
        PDFBlockRecord(
            block_id="pdf_block_1",
            source_id="src_pdf_1",
            source_file="report.pdf",
            page=1,
            block_no=0,
            text="The vessel Hope appears to have an older diesel engine. It was replaced in 2024.",
            bbox=[72.0, 100.0, 500.0, 140.0],
            char_start_document=0,
            char_end_document=78,
            section_path=["Inspection", "Engine"],
            extractor="fixture",
        ),
        PDFBlockRecord(
            block_id="pdf_block_2",
            source_id="src_pdf_1",
            source_file="report.pdf",
            page=1,
            block_no=1,
            text="The surveyor found no active fuel leak.",
            bbox=[72.0, 160.0, 500.0, 190.0],
            char_start_document=79,
            char_end_document=118,
            section_path=["Inspection", "Engine"],
            extractor="fixture",
        ),
    ]
    for block in blocks:
        append_jsonl(Path("data/jsonl/pdf_blocks.jsonl"), block)


def test_clean_pdf_text_repairs_hyphenation_and_wrapped_lines():
    cleaned, actions = clean_pdf_text("The ves-\nsel Hope\nappears old.")

    assert cleaned == "The vessel Hope appears old."
    assert actions == ["repair_hyphenation", "join_wrapped_lines"]


def test_classify_repeated_pdf_page_furniture():
    blocks = [
        _ExtractedPDFBlock(1, 0, "Confidential Survey", None, "fixture", []),
        _ExtractedPDFBlock(1, 1, "The vessel Hope had three masts.", None, "fixture", []),
        _ExtractedPDFBlock(1, 2, "Page 1", None, "fixture", []),
        _ExtractedPDFBlock(2, 0, "Confidential Survey", None, "fixture", []),
        _ExtractedPDFBlock(2, 1, "The engine was replaced in 2024.", None, "fixture", []),
        _ExtractedPDFBlock(2, 2, "Page 2", None, "fixture", []),
    ]

    furniture = classify_repeated_pdf_furniture(blocks)

    assert furniture == {
        (1, 0): "header",
        (1, 2): "footer",
        (2, 0): "header",
        (2, 2): "footer",
    }


def test_infer_pdf_section_paths_from_numbered_headings():
    blocks = [
        _ExtractedPDFBlock(1, 0, "1. Inspection", None, "fixture", []),
        _ExtractedPDFBlock(1, 1, "The vessel Hope had three masts.", None, "fixture", []),
        _ExtractedPDFBlock(1, 2, "1.1 Engine Condition", None, "fixture", []),
        _ExtractedPDFBlock(1, 3, "The engine was replaced in 2024.", None, "fixture", []),
        _ExtractedPDFBlock(2, 0, "2. Recommendations", None, "fixture", []),
        _ExtractedPDFBlock(2, 1, "The surveyor recommended a fuel inspection.", None, "fixture", []),
        _ExtractedPDFBlock(2, 2, "THE ENGINE WAS REPLACED IN 2024.", None, "fixture", []),
    ]

    section_paths = infer_pdf_section_paths(blocks)

    assert section_paths[(1, 0)] == ["1. Inspection"]
    assert section_paths[(1, 1)] == ["1. Inspection"]
    assert section_paths[(1, 2)] == ["1. Inspection", "1.1 Engine Condition"]
    assert section_paths[(1, 3)] == ["1. Inspection", "1.1 Engine Condition"]
    assert section_paths[(2, 0)] == ["2. Recommendations"]
    assert section_paths[(2, 1)] == ["2. Recommendations"]
    assert section_paths[(2, 2)] == ["2. Recommendations"]


def test_ingest_pdf_infers_section_paths_into_evidence(monkeypatch, tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0
        Path("report.pdf").write_bytes(b"fixture pdf")
        monkeypatch.setattr(
            "evidence_pipeline.ingest.pdf._extract_blocks",
            lambda _: [
                _ExtractedPDFBlock(1, 0, "1. Inspection", None, "fixture", []),
                _ExtractedPDFBlock(1, 1, "1.1 Engine Condition", None, "fixture", []),
                _ExtractedPDFBlock(1, 2, "The engine was replaced in 2024.", None, "fixture", []),
            ],
        )

        ingest = runner.invoke(app, ["ingest-pdf", "report.pdf"])
        assert ingest.exit_code == 0, ingest.stdout
        assert "blocks_created=3" in ingest.stdout

        blocks = [payload for _, payload in read_jsonl(Path("data/jsonl/pdf_blocks.jsonl"))]
        assert blocks[0]["section_path"] == ["1. Inspection"]
        assert blocks[1]["section_path"] == ["1. Inspection", "1.1 Engine Condition"]
        assert blocks[2]["section_path"] == ["1. Inspection", "1.1 Engine Condition"]

        evidence_result = runner.invoke(app, ["build-pdf-evidence"])
        assert evidence_result.exit_code == 0, evidence_result.stdout

        evidence = [payload for _, payload in read_jsonl(Path("data/jsonl/evidence.jsonl"))]
        body_evidence = next(record for record in evidence if record["provenance"]["block_no"] == 2)
        assert body_evidence["provenance"]["section_path"] == ["1. Inspection", "1.1 Engine Condition"]


def test_pdf_header_footer_blocks_are_not_evidence(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        for block in [
            PDFBlockRecord(
                block_id="header_1",
                source_id="src_pdf_1",
                source_file="report.pdf",
                page=1,
                block_no=0,
                block_type="header",
                text="Confidential Survey",
            ),
            PDFBlockRecord(
                block_id="body_1",
                source_id="src_pdf_1",
                source_file="report.pdf",
                page=1,
                block_no=1,
                block_type="text",
                text="The vessel Hope had three masts.",
            ),
            PDFBlockRecord(
                block_id="footer_1",
                source_id="src_pdf_1",
                source_file="report.pdf",
                page=1,
                block_no=2,
                block_type="footer",
                text="Page 1",
            ),
        ]:
            append_jsonl(Path("data/jsonl/pdf_blocks.jsonl"), block)

        result = runner.invoke(app, ["build-pdf-evidence"])
        assert result.exit_code == 0, result.stdout
        assert "evidence_created=1" in result.stdout
        assert "evidence_skipped=2" in result.stdout

        evidence = [payload for _, payload in read_jsonl(Path("data/jsonl/evidence.jsonl"))]
        assert [record["text"] for record in evidence] == ["The vessel Hope had three masts."]


def test_pdf_chunking_respects_section_boundaries(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        for evidence in [
            EvidenceRecord(
                evidence_id="ev_pdf_1",
                source_id="src_pdf_1",
                source_modality="pdf",
                evidence_type="text_span",
                text="The vessel Hope had three masts.",
                provenance={
                    "page": 1,
                    "block_no": 0,
                    "block_id": "pdf_block_1",
                    "section_path": ["1. Inspection"],
                },
            ),
            EvidenceRecord(
                evidence_id="ev_pdf_2",
                source_id="src_pdf_1",
                source_modality="pdf",
                evidence_type="text_span",
                text="The engine was replaced in 2024.",
                provenance={
                    "page": 1,
                    "block_no": 1,
                    "block_id": "pdf_block_2",
                    "section_path": ["1. Inspection"],
                },
            ),
            EvidenceRecord(
                evidence_id="ev_pdf_3",
                source_id="src_pdf_1",
                source_modality="pdf",
                evidence_type="text_span",
                text="The surveyor recommended a fuel inspection.",
                provenance={
                    "page": 2,
                    "block_no": 0,
                    "block_id": "pdf_block_3",
                    "section_path": ["2. Recommendations"],
                },
            ),
        ]:
            append_jsonl(Path("data/jsonl/evidence.jsonl"), evidence)

        first = runner.invoke(app, ["chunk-pdf", "--target-tokens", "1000"])
        second = runner.invoke(app, ["chunk-pdf", "--target-tokens", "1000"])
        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "chunks_created=2" in first.stdout
        assert "chunks_skipped=2" in second.stdout

        chunks = [payload for _, payload in read_jsonl(Path("data/jsonl/chunks.jsonl"))]
        assert [chunk["primary_evidence_ids"] for chunk in chunks] == [
            ["ev_pdf_1", "ev_pdf_2"],
            ["ev_pdf_3"],
        ]
        assert chunks[1]["overlap_evidence_ids"] == []
        assert chunks[0]["provenance_summary"]["section_paths"] == [["1. Inspection"]]
        assert chunks[1]["provenance_summary"]["section_paths"] == [["2. Recommendations"]]
        assert chunks[0]["chunking_policy"]["strategy"] == "section_page_block_token_fallback"


def test_pdf_artifact_pipeline_is_idempotent(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0
        _seed_pdf_blocks()

        commands = [
            ["build-pdf-evidence"],
            ["chunk-pdf", "--target-tokens", "20"],
            ["detect-pdf-spans"],
            ["extract-claims", "--modality", "pdf"],
            ["validate-claims"],
            ["normalize-claims"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        assert len(list(read_jsonl(Path("data/jsonl/pdf_blocks.jsonl")))) == 2
        assert len(list(read_jsonl(Path("data/jsonl/evidence.jsonl")))) == 2
        assert len(list(read_jsonl(Path("data/jsonl/chunks.jsonl")))) >= 1

        spans = [payload for _, payload in read_jsonl(Path("data/jsonl/spans.jsonl"))]
        span_texts = [span["text"] for span in spans]
        assert "The vessel Hope appears to have an older diesel engine." in span_texts
        assert "It was replaced in 2024." in span_texts
        assert "The surveyor found no active fuel leak." in span_texts
        assert len(list(read_jsonl(Path("data/jsonl/claims.raw.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 3

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
