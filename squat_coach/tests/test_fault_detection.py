"""Tests for fault detection."""
import numpy as np
from squat_coach.faults.evidence_engine import EvidenceEngine
from squat_coach.faults.fault_types import FaultDetection
from squat_coach.utils.enums import FaultType

def test_detects_insufficient_depth():
    engine = EvidenceEngine()
    features = {"primary_knee_angle": 120.0, "hip_depth_vs_knee": -0.1}
    faults = engine.evaluate(features, {})
    fault_types = [f.fault_type for f in faults]
    assert FaultType.INSUFFICIENT_DEPTH in fault_types

def test_no_depth_fault_when_deep():
    engine = EvidenceEngine()
    features = {"primary_knee_angle": 80.0, "hip_depth_vs_knee": 0.1}
    faults = engine.evaluate(features, {})
    fault_types = [f.fault_type for f in faults]
    assert FaultType.INSUFFICIENT_DEPTH not in fault_types

def test_detects_forward_lean():
    engine = EvidenceEngine()
    features = {"torso_inclination_deg": 45.0}
    config = {"baseline_torso_angle": 10.0}
    faults = engine.evaluate(features, config)
    fault_types = [f.fault_type for f in faults]
    assert FaultType.EXCESSIVE_FORWARD_LEAN in fault_types

def test_fault_has_required_fields():
    engine = EvidenceEngine()
    features = {"primary_knee_angle": 120.0, "hip_depth_vs_knee": -0.1}
    faults = engine.evaluate(features, {})
    for fault in faults:
        assert 0.0 <= fault.severity <= 1.0
        assert 0.0 <= fault.confidence <= 1.0
        assert len(fault.evidence) > 0
