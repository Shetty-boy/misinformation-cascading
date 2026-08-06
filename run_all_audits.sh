#!/bin/bash
set -e

echo "1. Running Traversal Correctness Audit..."
venv/bin/python tests/phase04_05_graph/audit_traversal_correctness.py

echo ""
echo "2. Running Graph Integrity Audit..."
venv/bin/python tests/phase04_05_graph/audit_integrity.py

echo ""
echo "3. Running Determinism Audit..."
venv/bin/python tests/phase04_05_graph/audit_determinism.py

echo ""
echo "4. Running Depth Correctness Validation..."
venv/bin/python tests/phase04_05_graph/audit_depth.py

echo ""
echo "5. Running Reachability Audit..."
venv/bin/python tests/phase04_05_graph/audit_reachability.py

echo ""
echo "6. Running Performance Benchmark (takes time due to 3 runs)..."
venv/bin/python tests/phase04_05_graph/audit_performance.py

echo ""
echo "7. Running Graph Statistics Regeneration..."
venv/bin/python src/cascade2vec/phase04_05_graph/stats.py

echo ""
echo "8. Running Feature Readiness matrix generator..."
venv/bin/python tests/phase04_05_graph/audit_feature_readiness.py

echo ""
echo "All automated scripts completed successfully!"
