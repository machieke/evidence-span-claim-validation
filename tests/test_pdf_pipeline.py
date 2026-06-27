from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.ingest.pdf import (
    _ExtractedPDFBlock,
    classify_repeated_pdf_furniture,
    clean_pdf_text,
)
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
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
