import json
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import NormalizedClaimRecord, RawClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.sources import SourceRecord
from evidence_pipeline.schemas.validation import ValidationRecord


runner = CliRunner()


def test_review_claim_records_idempotent_decision_and_trace(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        evidence = EvidenceRecord(
            evidence_id="ev_img_1",
            source_id="src_img_1",
            source_modality="image",
            evidence_type="visual_region",
            text=None,
            provenance={"region_id": "region_1", "bbox": [0, 0, 16, 16]},
        )
        claim = RawClaimRecord(
            claim_id="claim_img_label_1",
            source_id="src_img_1",
            source_modality="image",
            evidence_id="ev_img_1",
            claim_type="named_visual_classification",
            source_faithful_claim="Model classifier_v1 classified region region_1 as red.",
            subject="region_1",
            predicate="classified_as",
            object="red",
            modality="model_observation",
            attribution={"type": "model", "agent": "classifier_v1"},
            truth_status="model_observation_unverified",
            confidence=0.9,
        )
        append_jsonl(Path("data/jsonl/evidence.jsonl"), evidence)
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), claim)

        first = runner.invoke(
            app,
            [
                "review-claim",
                "claim_img_label_1",
                "--decision",
                "accept",
                "--reviewer-id",
                "reviewer_1",
                "--reason-code",
                "human_confirmed_label",
            ],
        )
        second = runner.invoke(
            app,
            [
                "review-claim",
                "claim_img_label_1",
                "--decision",
                "accept",
                "--reviewer-id",
                "reviewer_1",
                "--reason-code",
                "human_confirmed_label",
            ],
        )
        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "created=True" in first.stdout
        assert "created=False" in second.stdout

        reviews = [payload for _, payload in read_jsonl(Path("data/jsonl/review_decisions.jsonl"))]
        assert len(reviews) == 1
        assert reviews[0]["claim_id"] == "claim_img_label_1"
        assert reviews[0]["decision"] == "accept"
        assert reviews[0]["reason_codes"] == ["human_confirmed_label"]

        audit_events = [payload for _, payload in read_jsonl(Path("data/jsonl/audit_events.jsonl"))]
        assert len(audit_events) == 2
        assert [event["status"] for event in audit_events] == ["created", "skipped"]
        assert all(event["action"] == "review_claim" for event in audit_events)
        assert all(event["actor_id"] == "reviewer_1" for event in audit_events)
        assert all(event["claim_id"] == "claim_img_label_1" for event in audit_events)
        assert audit_events[1]["details"]["skip_reason"] == "duplicate_review"

        trace = runner.invoke(app, ["trace-claim", "claim_img_label_1"])
        assert trace.exit_code == 0, trace.stdout
        trace_payload = json.loads(trace.stdout)
        assert trace_payload["review_decisions"][0]["review_id"] == reviews[0]["review_id"]
        assert [event["status"] for event in trace_payload["audit_events"]] == ["created", "skipped"]

        html_trace = runner.invoke(app, ["trace-claim", "claim_img_label_1", "--format", "html"])
        assert html_trace.exit_code == 0, html_trace.stdout
        assert "data/reports/claim_img_label_1.trace.html" in html_trace.stdout
        html_trace_text = Path("data/reports/claim_img_label_1.trace.html").read_text(encoding="utf-8")
        assert "<h1>Claim Trace: claim_img_label_1</h1>" in html_trace_text
        assert "<h2>Review Decisions</h2>" in html_trace_text
        assert "human_confirmed_label" in html_trace_text

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| review_decisions | 1 |" in report_text
        assert "| audit_events | 2 |" in report_text
        assert "| accept | 1 |" in report_text
        assert "| review_claim | 2 |" in report_text

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout

        invalid = runner.invoke(app, ["review-claim", "claim_img_label_1", "--decision", "maybe"])
        assert invalid.exit_code != 0
        assert "review decision must be one of" in invalid.stdout

        invalid_trace = runner.invoke(app, ["trace-claim", "claim_img_label_1", "--format", "xml"])
        assert invalid_trace.exit_code != 0
        assert "trace format must be json or html" in invalid_trace.stdout


def test_report_tracks_review_disagreement_rate(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            RawClaimRecord(
                claim_id="claim_img_label_1",
                source_id="src_img_1",
                source_modality="image",
                evidence_id="ev_img_1",
                claim_type="named_visual_classification",
                source_faithful_claim="Model classifier_v1 classified region region_1 as red.",
                subject="region_1",
                predicate="classified_as",
                object="red",
                modality="model_observation",
                attribution={"type": "model", "agent": "classifier_v1"},
                truth_status="model_observation_unverified",
                confidence=0.9,
            ),
        )

        accept = runner.invoke(
            app,
            [
                "review-claim",
                "claim_img_label_1",
                "--decision",
                "accept",
                "--reviewer-id",
                "reviewer_1",
            ],
        )
        reject = runner.invoke(
            app,
            [
                "review-claim",
                "claim_img_label_1",
                "--decision",
                "reject",
                "--reviewer-id",
                "reviewer_2",
            ],
        )
        assert accept.exit_code == 0, accept.stdout
        assert reject.exit_code == 0, reject.stdout

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| Review disagreement rate | 100.0% |" in report_text


def test_review_queue_exports_unreviewed_quarantined_claims(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/sources.jsonl"),
            SourceRecord(
                source_id="src_img_1",
                source_modality="image",
                source_file="image.png",
            ),
        )
        append_jsonl(
            Path("data/jsonl/evidence.jsonl"),
            EvidenceRecord(
                evidence_id="ev_img_1",
                source_id="src_img_1",
                source_modality="image",
                evidence_type="visual_region",
                provenance={
                    "region_id": "region_1",
                    "bbox": [0, 0, 16, 16],
                    "crop_path": "data/work/crops/region_1.png",
                },
                risk_flags=["low_visual_contrast"],
            ),
        )
        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            RawClaimRecord(
                claim_id="claim_img_label_1",
                source_id="src_img_1",
                source_modality="image",
                evidence_id="ev_img_1",
                claim_type="named_visual_classification",
                source_faithful_claim="Model classifier_v1 classified region region_1 as red.",
                subject="region_1",
                predicate="classified_as",
                object="red",
                modality="model_observation",
                attribution={"type": "model", "agent": "classifier_v1"},
                truth_status="model_observation_unverified",
                confidence=0.4,
                risk_flags=["color_only_classification"],
            ),
        )
        append_jsonl(
            Path("data/jsonl/claims.normalized.jsonl"),
            NormalizedClaimRecord(
                normalized_claim_id="nclaim_img_label_1",
                claim_id="claim_img_label_1",
                source_id="src_img_1",
                evidence_id="ev_img_1",
                normalized_claim={
                    "subject": "region_1",
                    "predicate": "classified_as",
                    "object": "red",
                    "qualifiers": {
                        "modality": "model_observation",
                        "truth_status": "model_observation_unverified",
                    },
                },
            ),
        )

        validation = runner.invoke(app, ["validate-claims"])
        assert validation.exit_code == 0, validation.stdout
        assert "claims_quarantined=1" in validation.stdout
        append_jsonl(
            Path("data/jsonl/validations.jsonl"),
            ValidationRecord(
                validation_id="val_claim_img_label_1_review_context",
                claim_id="claim_img_label_1",
                record_id="claim_img_label_1",
                stage="validate_claims",
                status="quarantined",
                errors=["image_label_low_confidence"],
                warnings=["needs_visual_double_check"],
            ),
        )

        queue = runner.invoke(app, ["review-queue"])
        assert queue.exit_code == 0, queue.stdout
        assert "data/reports/review_queue.jsonl review_items=1" in queue.stdout
        items = [payload for _, payload in read_jsonl(Path("data/reports/review_queue.jsonl"))]
        assert len(items) == 1
        assert items[0]["claim_id"] == "claim_img_label_1"
        assert items[0]["source_file"] == "image.png"
        assert items[0]["validation_status"] == "quarantined"
        assert items[0]["reason_codes"] == ["image_label_low_confidence"]
        assert items[0]["warnings"] == ["needs_visual_double_check"]
        assert items[0]["risk_flags"] == ["color_only_classification", "low_visual_contrast"]
        assert items[0]["review_state"] == "unreviewed"
        assert items[0]["evidence"]["provenance"]["bbox"] == [0, 0, 16, 16]
        assert items[0]["evidence_anchor"] == {
            "source_modality": "image",
            "source_id": "src_img_1",
            "source_file": "image.png",
            "evidence_id": "ev_img_1",
            "evidence_type": "visual_region",
            "region_id": "region_1",
            "bbox": [0, 0, 16, 16],
            "crop_path": "data/work/crops/region_1.png",
        }
        assert items[0]["normalized_claims"][0]["normalized_claim"]["predicate"] == "classified_as"
        assert (
            items[0]["review_commands"]["accept"]
            == "python3 -m evidence_pipeline review-claim claim_img_label_1 --decision accept "
            "--reviewer-id human_reviewer --reason-code image_label_low_confidence"
        )
        assert "review-claim claim_img_label_1 --decision reject" in items[0]["review_commands"]["reject"]
        assert (
            "--reason-code image_label_low_confidence"
            in items[0]["review_commands"]["needs_review"]
        )

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert [job["stage"] for job in jobs] == ["validate_claims", "review_queue"]
        assert jobs[1]["model_id"] == "review.queue.v1"
        assert jobs[1]["input_record_ids"] == [
            "claims_normalized",
            "claims_raw",
            "format:jsonl",
            "include_reviewed:False",
            "review_decisions",
            "validations",
        ]
        assert jobs[1]["metrics"] == {"review_items": 1}

        trace = runner.invoke(app, ["trace-claim", "claim_img_label_1"])
        assert trace.exit_code == 0, trace.stdout
        trace_payload = json.loads(trace.stdout)
        assert trace_payload["review_queue"][0]["review_queue_id"] == items[0]["review_queue_id"]
        assert trace_payload["review_queue"][0]["normalized_claims"][0]["normalized_claim"]["object"] == "red"
        assert trace_payload["review_queue"][0]["evidence_anchor"]["bbox"] == [0, 0, 16, 16]
        assert (
            trace_payload["review_queue"][0]["review_commands"]["accept"]
            == items[0]["review_commands"]["accept"]
        )

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 2 |" in report_text
        assert "## Jobs By Stage" in report_text
        assert "| review_queue | 1 |" in report_text
        assert "## Review Queue By State" in report_text
        assert "| unreviewed | 1 |" in report_text
        assert "## Review Queue Reasons" in report_text
        assert "| image_label_low_confidence | 1 |" in report_text
        assert "## Review Queue Warnings" in report_text
        assert "| needs_visual_double_check | 1 |" in report_text
        assert "## Review Queue Risk Flags" in report_text
        assert "| color_only_classification | 1 |" in report_text
        assert "| low_visual_contrast | 1 |" in report_text

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
        assert "data/reports/review_queue.jsonl: checked 1 records" in artifact_check.stdout

        html_queue = runner.invoke(app, ["review-queue", "--format", "html"])
        assert html_queue.exit_code == 0, html_queue.stdout
        assert "data/reports/review_queue.html review_items=1" in html_queue.stdout
        html_text = Path("data/reports/review_queue.html").read_text(encoding="utf-8")
        assert "<!doctype html>" in html_text
        assert "<h1>Claim Review Queue</h1>" in html_text
        assert "image_label_low_confidence" in html_text
        assert "classified_as" in html_text
        assert "<th>Anchor</th>" in html_text
        assert "<th>Preview</th>" in html_text
        assert "<th>Warnings</th>" in html_text
        assert "<th>Risk Flags</th>" in html_text
        assert "region_id" in html_text
        assert "region_1" in html_text
        assert "[0, 0, 16, 16]" in html_text
        assert '<img loading="lazy" src="../work/crops/region_1.png"' in html_text
        assert 'alt="Evidence crop for claim_img_label_1"' in html_text
        assert "needs_visual_double_check" in html_text
        assert "color_only_classification" in html_text
        assert "low_visual_contrast" in html_text
        assert '<select name="decision"' in html_text
        assert '<input name="reason_code"' in html_text
        assert '<textarea name="notes"' in html_text
        assert 'data-decision="accept"' in html_text
        assert 'data-decision="reject"' in html_text
        assert 'data-decision="needs_review"' in html_text
        assert "review-claim claim_img_label_1 --decision accept" in html_text
        assert "--reason-code image_label_low_confidence" in html_text
        assert "Model classifier_v1 classified region region_1 as red." in html_text

        html_trace = runner.invoke(app, ["trace-claim", "claim_img_label_1", "--format", "html"])
        assert html_trace.exit_code == 0, html_trace.stdout
        trace_html = Path("data/reports/claim_img_label_1.trace.html").read_text(encoding="utf-8")
        assert "<h2>Review Queue</h2>" in trace_html
        assert items[0]["review_queue_id"] in trace_html

        needs_review = runner.invoke(
            app,
            [
                "review-claim",
                "claim_img_label_1",
                "--decision",
                "needs_review",
                "--reviewer-id",
                "reviewer_1",
                "--reason-code",
                "needs_second_reviewer",
            ],
        )
        assert needs_review.exit_code == 0, needs_review.stdout

        unresolved_queue = runner.invoke(app, ["review-queue"])
        assert unresolved_queue.exit_code == 0, unresolved_queue.stdout
        unresolved_items = [
            payload for _, payload in read_jsonl(Path("data/reports/review_queue.jsonl"))
        ]
        assert len(unresolved_items) == 1
        assert unresolved_items[0]["review_state"] == "needs_review"

        review = runner.invoke(
            app,
            [
                "review-claim",
                "claim_img_label_1",
                "--decision",
                "accept",
                "--reviewer-id",
                "reviewer_1",
                "--reason-code",
                "human_confirmed_label",
            ],
        )
        assert review.exit_code == 0, review.stdout

        reviewed_queue = runner.invoke(app, ["review-queue"])
        assert reviewed_queue.exit_code == 0, reviewed_queue.stdout
        assert "review_items=0" in reviewed_queue.stdout

        include_reviewed = runner.invoke(app, ["review-queue", "--include-reviewed"])
        assert include_reviewed.exit_code == 0, include_reviewed.stdout
        reviewed_items = [
            payload for _, payload in read_jsonl(Path("data/reports/review_queue.jsonl"))
        ]
        assert len(reviewed_items) == 1
        assert reviewed_items[0]["review_state"] == "accept"
        assert reviewed_items[0]["latest_review"]["reason_codes"] == ["human_confirmed_label"]

        invalid_format = runner.invoke(app, ["review-queue", "--format", "pdf"])
        assert invalid_format.exit_code != 0
        assert "review queue format must be jsonl or html" in invalid_format.stdout
