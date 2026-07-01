import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.cli import app
from evidence_pipeline.ingest import audio as audio_ingest
from evidence_pipeline.ingest.audio import normalize_audio_source
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def test_audio_media_normalization_registers_planned_source(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        media = Path("meeting.mp3")
        media.write_bytes(b"fake audio bytes")

        first = runner.invoke(
            app,
            [
                "normalize-audio",
                "meeting.mp3",
                "--sample-rate",
                "8000",
                "--channels",
                "1",
                "--metadata",
                "collection=fixture",
            ],
        )
        second = runner.invoke(
            app,
            ["normalize-audio", "meeting.mp3", "--sample-rate", "8000", "--channels", "1"],
        )

        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "source_created=True" in first.stdout
        assert "source_created=False" in second.stdout
        assert "executed=False" in first.stdout

        sources = [payload for _, payload in read_jsonl(Path("data/jsonl/sources.jsonl"))]
        assert len(sources) == 1
        assert sources[0]["source_modality"] == "audio"
        assert sources[0]["source_file"] == "meeting.mp3"
        metadata = sources[0]["metadata"]
        assert metadata["collection"] == "fixture"
        assert metadata["media_kind"] == "audio_source"
        assert metadata["normalization_status"] == "planned"
        assert metadata["normalized_file"] == "data/work/normalized_audio/meeting_8khz_mono.wav"
        assert metadata["target_sample_rate"] == 8000
        assert metadata["target_channels"] == 1
        assert metadata["normalizer"] == "audio.normalization.ffmpeg_plan.v1"
        assert metadata["audio_normalizations"] == [
            {
                "normalization_command": [
                    "ffmpeg",
                    "-y",
                    "-i",
                    "meeting.mp3",
                    "-ac",
                    "1",
                    "-ar",
                    "8000",
                    "data/work/normalized_audio/meeting_8khz_mono.wav",
                ],
                "normalization_policy_id": metadata["normalization_policy_id"],
                "normalization_status": "planned",
                "normalized_file": "data/work/normalized_audio/meeting_8khz_mono.wav",
                "normalizer": "audio.normalization.ffmpeg_plan.v1",
                "target_channels": 1,
                "target_sample_rate": 8000,
            }
        ]
        assert metadata["normalization_command"] == [
            "ffmpeg",
            "-y",
            "-i",
            "meeting.mp3",
            "-ac",
            "1",
            "-ar",
            "8000",
            "data/work/normalized_audio/meeting_8khz_mono.wav",
        ]

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["stage"] == "normalize_audio"
        assert jobs[0]["model_id"] == "audio.normalization.ffmpeg_plan.v1"
        assert jobs[0]["input_record_ids"] == [
            "audio:meeting.mp3",
            "execute:0",
            f"normalization_policy:{metadata['normalization_policy_id']}",
        ]
        assert jobs[0]["metrics"] == {"executed": 0, "source_created": 1, "source_updated": 0}
        assert jobs[0]["metadata"]["command"] == metadata["normalization_command"]
        assert jobs[0]["metadata"]["normalization_policy_id"] == metadata["normalization_policy_id"]


def test_audio_media_normalization_jobs_are_policy_specific(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        media = Path("meeting.mp3")
        media.write_bytes(b"fake audio bytes")

        first = runner.invoke(app, ["normalize-audio", "meeting.mp3", "--sample-rate", "8000"])
        second = runner.invoke(
            app,
            ["normalize-audio", "meeting.mp3", "--sample-rate", "16000", "--channels", "2"],
        )
        duplicate_second = runner.invoke(
            app,
            ["normalize-audio", "meeting.mp3", "--sample-rate", "16000", "--channels", "2"],
        )

        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert duplicate_second.exit_code == 0, duplicate_second.stdout

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 2
        assert [job["metadata"]["normalized_file"] for job in jobs] == [
            "data/work/normalized_audio/meeting_8khz_mono.wav",
            "data/work/normalized_audio/meeting_16khz_stereo.wav",
        ]
        assert jobs[1]["metrics"] == {"executed": 0, "source_created": 0, "source_updated": 1}
        assert jobs[0]["metadata"]["normalization_policy_id"] != jobs[1]["metadata"]["normalization_policy_id"]

        sources = [payload for _, payload in read_jsonl(Path("data/jsonl/sources.jsonl"))]
        assert len(sources) == 1
        assert sources[0]["metadata"]["normalized_file"] == "data/work/normalized_audio/meeting_16khz_stereo.wav"
        assert len(sources[0]["metadata"]["audio_normalizations"]) == 2


def test_audio_media_execute_updates_planned_source(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    media = Path("meeting.mp3")
    media.write_bytes(b"fake audio bytes")
    config = PipelineConfig()
    calls = []

    monkeypatch.setattr(audio_ingest.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr(audio_ingest.subprocess, "run", fake_run)

    planned = normalize_audio_source(media, config, sample_rate=8000)
    executed = normalize_audio_source(media, config, sample_rate=8000, execute=True)

    assert planned.source_created is True
    assert planned.source_updated is False
    assert executed.source_created is False
    assert executed.source_updated is True
    assert executed.normalization_policy_id == planned.normalization_policy_id
    assert calls == [(executed.command, True)]

    sources = [payload for _, payload in read_jsonl(config.jsonl_paths()["sources"])]
    assert len(sources) == 1
    metadata = sources[0]["metadata"]
    assert metadata["normalization_status"] == "created"
    assert metadata["audio_normalizations"][0]["normalization_status"] == "created"


def test_audio_media_dry_run_does_not_downgrade_created_source(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    media = Path("meeting.mp3")
    media.write_bytes(b"fake audio bytes")
    config = PipelineConfig()

    monkeypatch.setattr(audio_ingest.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(audio_ingest.subprocess, "run", lambda command, check: None)

    executed = normalize_audio_source(media, config, sample_rate=8000, execute=True)
    planned = normalize_audio_source(media, config, sample_rate=8000)

    assert executed.source_created is True
    assert planned.source_created is False
    assert planned.source_updated is False
    sources = [payload for _, payload in read_jsonl(config.jsonl_paths()["sources"])]
    metadata = sources[0]["metadata"]
    assert metadata["normalization_status"] == "created"
    assert metadata["audio_normalizations"][0]["normalization_status"] == "created"


def test_audio_media_normalization_rejects_overwriting_input(tmp_path: Path):
    media = tmp_path / "meeting.mp3"
    media.write_bytes(b"fake audio bytes")

    with pytest.raises(ValueError, match="must differ from the input path"):
        normalize_audio_source(media, PipelineConfig(), normalized_file=media)


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

        sources = [payload for _, payload in read_jsonl(Path("data/jsonl/sources.jsonl"))]
        assert len(sources) == 1
        assert sources[0]["metadata"]["duration_seconds"] == 8.0
        assert len(list(read_jsonl(Path("data/jsonl/audio_utterances.jsonl")))) == 3
        evidence = [payload for _, payload in read_jsonl(Path("data/jsonl/evidence.jsonl"))]
        assert len(evidence) == 3
        assert {record["provenance"]["source_duration"] for record in evidence} == {8.0}
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
