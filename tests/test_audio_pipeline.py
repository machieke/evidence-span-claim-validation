import json
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def _write_audio_transcript(path: Path) -> None:
    payload = {
        "source_file": "meeting.wav",
        "language": "en",
        "utterances": [
            {
                "id": "utt_1",
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 2.0,
                "text": "Did Hope have an engine?",
                "asr_confidence": 0.95,
                "diarization_confidence": 0.9,
            },
            {
                "id": "utt_2",
                "speaker": "SPEAKER_01",
                "start": 2.1,
                "end": 6.0,
                "text": "I saw it yesterday. It had two engines.",
                "asr_confidence": 0.92,
                "diarization_confidence": 0.86,
            },
            {
                "id": "utt_3",
                "speaker": "SPEAKER_01",
                "start": 6.2,
                "end": 8.0,
                "text": "It did not leak.",
                "asr_confidence": 0.91,
                "diarization_confidence": 0.86,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_overlapping_audio_transcript(path: Path) -> None:
    payload = {
        "source_file": "overlap.wav",
        "language": "en",
        "utterances": [
            {
                "id": "utt_overlap_1",
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 3.0,
                "text": "Hope had three masts.",
                "asr_confidence": 0.95,
                "diarization_confidence": 0.9,
            },
            {
                "id": "utt_overlap_2",
                "speaker": "SPEAKER_01",
                "start": 2.5,
                "end": 4.0,
                "text": "The engine was replaced.",
                "asr_confidence": 0.96,
                "diarization_confidence": 0.88,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_low_confidence_audio_transcript(path: Path) -> None:
    payload = {
        "source_file": "low_confidence.wav",
        "language": "en",
        "utterances": [
            {
                "id": "utt_low_asr",
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 2.0,
                "text": "Hope had three masts.",
                "asr_confidence": 0.5,
                "diarization_confidence": 0.9,
            },
            {
                "id": "utt_speaker_uncertain",
                "speaker": "SPEAKER_01",
                "start": 3.0,
                "end": 5.0,
                "text": "The engine was replaced.",
                "asr_confidence": 0.95,
                "diarization_confidence": 0.4,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audio_transcript_pipeline_is_idempotent(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        transcript = Path("transcript.json")
        _write_audio_transcript(transcript)

        commands = [
            ["ingest-audio-transcript", "transcript.json"],
            ["build-audio-evidence"],
            ["chunk-audio", "--previous-utterances", "1"],
            ["detect-audio-spans"],
            ["extract-claims", "--modality", "audio"],
            ["validate-claims"],
            ["normalize-claims"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        assert len(list(read_jsonl(Path("data/jsonl/sources.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/audio_utterances.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/evidence.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/chunks.jsonl")))) == 3

        spans = [payload for _, payload in read_jsonl(Path("data/jsonl/spans.jsonl"))]
        span_texts = [span["text"] for span in spans]
        assert span_texts == [
            "Did Hope have an engine?",
            "I saw it yesterday.",
            "It had two engines.",
            "It did not leak.",
        ]
        assert "question_speech_act" in spans[0]["risk_flags"]
        assert "context_dependent_coreference" in spans[2]["risk_flags"]

        assert len(list(read_jsonl(Path("data/jsonl/claims.raw.jsonl")))) == 4
        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 4
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 4

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| audio_utterances | 3 |" in report_text
        assert "| audio | 4 |" in report_text

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout


def test_overlapping_audio_speech_is_quarantined(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        transcript = Path("overlap.json")
        _write_overlapping_audio_transcript(transcript)

        commands = [
            ["ingest-audio-transcript", "overlap.json"],
            ["build-audio-evidence"],
            ["chunk-audio"],
            ["detect-audio-spans"],
            ["extract-claims", "--modality", "audio"],
            ["validate-claims"],
        ]
        for command in commands:
            result = runner.invoke(app, command)
            assert result.exit_code == 0, result.stdout

        utterances = [payload for _, payload in read_jsonl(Path("data/jsonl/audio_utterances.jsonl"))]
        assert all("overlapping_speech" in utterance["risk_flags"] for utterance in utterances)

        raw_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert len(raw_claims) == 2
        assert all("overlapping_speech" in claim["risk_flags"] for claim in raw_claims)

        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]
        assert all(validation["status"] == "quarantined" for validation in validations)
        assert len(quarantined) == 2
        assert all(record["reason_codes"] == ["overlapping_speech"] for record in quarantined)


def test_low_confidence_audio_is_quarantined(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        transcript = Path("low_confidence.json")
        _write_low_confidence_audio_transcript(transcript)

        commands = [
            ["ingest-audio-transcript", "low_confidence.json"],
            ["build-audio-evidence"],
            ["chunk-audio"],
            ["detect-audio-spans"],
            ["extract-claims", "--modality", "audio"],
            ["validate-claims"],
        ]
        for command in commands:
            result = runner.invoke(app, command)
            assert result.exit_code == 0, result.stdout

        raw_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert [claim["risk_flags"] for claim in raw_claims] == [
            ["low_asr_confidence"],
            ["speaker_uncertain"],
        ]

        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]
        assert [record["reason_codes"] for record in quarantined] == [
            ["low_asr_confidence"],
            ["speaker_uncertain"],
        ]
