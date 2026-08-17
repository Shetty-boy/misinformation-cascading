import os
import pytest
import pandas as pd
import numpy as np
from cascade2vec.phase13_14_adaptive_stopping.detection_loop import run_detection_loop, compute_detection_metrics
from cascade2vec.phase13_14_adaptive_stopping.adaptive_threshold import _find_optimal_static_thresholds

class DummyModel:
    def __init__(self, thresholds):
        self.thresholds = thresholds
    def predict(self, X):
        return self.thresholds

@pytest.fixture
def mock_confidence_df():
    # 2 cascades:
    # C1 (rumour): confidence crosses threshold at t=5
    # C2 (non-rumour): confidence never crosses threshold
    
    rows = []
    # C1
    c1_confs = {1: 0.1, 2: 0.3, 5: 0.6, 10: 0.8, 15: 0.8, 30: 0.9, 60: 0.95, 120: 0.99}
    for t, conf in c1_confs.items():
        rows.append({
            "cascade_id": "C1",
            "t_minutes": t,
            "label_binary": 1,
            "split": "test",
            "confidence_c2v": conf,
            "feature1": 10.0
        })
        
    # C2
    c2_confs = {1: 0.1, 2: 0.1, 5: 0.1, 10: 0.2, 15: 0.2, 30: 0.3, 60: 0.3, 120: 0.4}
    for t, conf in c2_confs.items():
        rows.append({
            "cascade_id": "C2",
            "t_minutes": t,
            "label_binary": 0,
            "split": "test",
            "confidence_c2v": conf,
            "feature1": 5.0
        })
        
    return pd.DataFrame(rows)


def test_loop_emits_at_first_crossing(mock_confidence_df):
    # If best_fixed = 0.5, adaptive_thresh = 0.55
    # C1 confidence_c2v crosses 0.5 at t=5 (0.6 >= 0.5).
    # crosses 0.55 at t=5 (0.6 >= 0.55).
    
    thresholds = np.array([0.55] * len(mock_confidence_df))
    model = DummyModel(thresholds)
    
    res = run_detection_loop(mock_confidence_df, "test", model, ["feature1"], 0.5, "c2v")
    
    assert len(res) == 2
    c1_res = res[res["cascade_id"] == "C1"].iloc[0]
    
    assert c1_res["adaptive_time"] == 5
    assert c1_res["adaptive_pred"] == 1
    
    assert c1_res["fixed_thresh_time"] == 5
    assert c1_res["fixed_thresh_pred"] == 1

def test_loop_falls_through_to_120(mock_confidence_df):
    # C2 never crosses 0.5
    thresholds = np.array([0.55] * len(mock_confidence_df))
    model = DummyModel(thresholds)
    
    res = run_detection_loop(mock_confidence_df, "test", model, ["feature1"], 0.5, "c2v")
    
    c2_res = res[res["cascade_id"] == "C2"].iloc[0]
    
    assert c2_res["adaptive_time"] == 120
    assert c2_res["adaptive_pred"] == 0
    assert c2_res["fixed_thresh_time"] == 120
    assert c2_res["fixed_thresh_pred"] == 0

def test_fixed_baseline_deterministic(mock_confidence_df):
    thresholds = np.array([0.55] * len(mock_confidence_df))
    model = DummyModel(thresholds)
    
    res1 = run_detection_loop(mock_confidence_df, "test", model, ["feature1"], 0.5, "c2v")
    res2 = run_detection_loop(mock_confidence_df, "test", model, ["feature1"], 0.5, "c2v")
    
    assert res1.equals(res2)

def test_mdt_less_than_full_window(mock_confidence_df):
    thresholds = np.array([0.55] * len(mock_confidence_df))
    model = DummyModel(thresholds)
    
    res = run_detection_loop(mock_confidence_df, "test", model, ["feature1"], 0.5, "c2v")
    metrics = compute_detection_metrics(res)
    
    assert metrics["adaptive"]["mdt"] < 120.0
    assert metrics["fixed_thresh"]["mdt"] < 120.0
    assert metrics["fixed_120"]["mdt"] == 120.0
    assert metrics["fixed_30"]["mdt"] == 30.0

def test_optimal_static_thresholds():
    # Test _find_optimal_static_thresholds
    rows = []
    for i in range(100):
        # Good separator at t=5: if conf > 0.8, all are label 1
        conf = i / 100.0
        label = 1 if conf > 0.8 else 0
        rows.append({"t_minutes": 5, "label_binary": label, "confidence_c2v": conf})
        
    df = pd.DataFrame(rows)
    t_opt = _find_optimal_static_thresholds(df, "confidence_c2v", 0.99)
    assert 5 in t_opt
    assert t_opt[5] >= 0.8
