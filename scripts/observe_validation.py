"""Replay validate_videos with read-only per-frame and per-cycle diagnostics."""
from collections import Counter
from dataclasses import asdict
import json
from unittest.mock import patch

import validate_videos
from exercise_analysis.form.twist import TwistEvidence
from exercise_analysis.pose.mediapipe_adapter import PoseEstimator

run_session = validate_videos.run


def observed_run(source, model, bundle, directory, *, on_frame, **options):
    mode = "manual" if options["manual"] else "automatic"
    raw, active, phase = Counter(), Counter(), Counter()
    pose_frames, twist_samples = Counter(), Counter()
    events = []
    estimate_pose = PoseEstimator.estimate
    sample_twist = TwistEvidence.samples

    def observe_frame(state):
        raw[state.prediction.exercise or "Missing"] += 1
        active[state.decision.active] += 1
        phase[state.count.phase] += 1
        if state.count.event:
            verdict = asdict(state.last)
            events.append({key: verdict[key] for key in
                           ("cycle", "status", "satisfied", "failed", "unavailable")})
        on_frame(state)

    def observe_pose(estimator, frame):
        pose = estimate_pose(estimator, frame)
        pose_frames["tracked"] += bool(pose.landmarks)
        pose_frames["world_shoulders_hips"] += all(
            side + "_" + joint in pose.world for side in ("LEFT", "RIGHT") for joint in ("SHOULDER", "HIP")
        )
        pose_frames["face: " + pose.face_status] += 1
        pose_frames["face_enabled"] += estimator.face is not None
        pose_frames["chin_available"] += pose.chin_y is not None
        return pose

    def observe_twist(evidence, *args, **kwargs):
        values = sample_twist(evidence, *args, **kwargs)
        twist_samples["calls"] += 1
        twist_samples["neutral_set"] += evidence.neutral is not None
        for key, value in values.items():
            twist_samples[key + "_available"] += value is not None
            twist_samples[key + "_missing"] += value is None
        return values

    with patch.object(PoseEstimator, "estimate", observe_pose), patch.object(TwistEvidence, "samples", observe_twist):
        session, path = run_session(source, model, bundle, directory, on_frame=observe_frame, **options)
    record = dict(video_path=str(source), mode=mode, raw=raw, active=active, phase=phase,
                  pose=pose_frames, twist=twist_samples, events=events)
    with (directory.parent / "observations.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, allow_nan=False) + "\n")
    return session, path


if __name__ == "__main__":
    with patch.object(validate_videos, "run", observed_run):
        raise SystemExit(validate_videos.main())
