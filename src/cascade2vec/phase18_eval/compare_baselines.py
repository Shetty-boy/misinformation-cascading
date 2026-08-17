import os
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)

OUT_DIR = "logs/phase18_eval"
COMP_TABLE = os.path.join(OUT_DIR, "master_comparison.md")

def build_master_comparison():
    logger.info("Building Master Comparison Table...")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    results = []
    
    # Load C2V results
    c2v_res_file = "logs/phase11_12_cascade2vec/c2v_results.json"
    if os.path.exists(c2v_res_file):
        with open(c2v_res_file, "r") as f:
            c2v_data = json.load(f)
            test_metrics = c2v_data["test_metrics"]
            results.append({
                "Model": "CASCADE2VEC",
                "Type": "Proposed",
                "Macro F1": test_metrics.get("macro_f1", 0),
                "Accuracy": test_metrics.get("accuracy", 0),
                "Weighted F1": test_metrics.get("weighted_f1", 0),
                "ROC-AUC": test_metrics.get("roc_auc", 0),
                "MDT (mins)": "—",
                "Source Phase": "11-12"
            })
            
    # Load SOTA results from sota_comparison.md parsing (simplest approach for now)
    sota_file = "logs/phase08_10_sota_baselines/sota_comparison.md"
    if os.path.exists(sota_file):
        with open(sota_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("|") and not line.startswith("| Model") and not line.startswith("|---"):
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 6:
                        model_name = parts[1].replace("**", "")
                        # avoid duplicating c2v if it's there
                        if "CASCADE2VEC" in model_name:
                            continue
                        macro_f1 = parts[3].replace("**", "")
                        accuracy = parts[2]
                        weighted_f1 = parts[4]
                        roc_auc = parts[5]
                        results.append({
                            "Model": model_name,
                            "Type": "SOTA Baseline",
                            "Macro F1": float(macro_f1) if macro_f1 else 0,
                            "Accuracy": float(accuracy) if accuracy else 0,
                            "Weighted F1": float(weighted_f1) if weighted_f1 else 0,
                            "ROC-AUC": float(roc_auc) if roc_auc else 0,
                            "MDT (mins)": "—",
                            "Source Phase": "8-10"
                        })
                        
    # Load Adaptive Threshold (H2) results
    h2_file = "logs/phase13_14_adaptive_stopping/detection_results.json"
    if os.path.exists(h2_file):
        with open(h2_file, "r") as f:
            h2_data = json.load(f)
            
            for model_type, data in h2_data.items():
                name = "CASCADE2VEC + Adaptive" if model_type == "c2v" else "KPG-Simplified + Adaptive"
                results.append({
                    "Model": name,
                    "Type": "Proposed + H2" if model_type == "c2v" else "SOTA + H2",
                    "Macro F1": data["adaptive"]["macro_f1"],
                    "Accuracy": "N/A", # Optional
                    "Weighted F1": "N/A",
                    "ROC-AUC": "N/A",
                    "MDT (mins)": f"{data['adaptive']['mdt']:.2f}",
                    "Source Phase": "13-14"
                })
                
    df = pd.DataFrame(results)
    
    # Sort by Macro F1 descending
    if not df.empty:
        df["Macro F1_Sort"] = pd.to_numeric(df["Macro F1"], errors='coerce')
        df = df.sort_values(by="Macro F1_Sort", ascending=False).drop(columns=["Macro F1_Sort"])
    
    # Write Markdown Table
    content = f"""# Master Comparison Table
    
{df.to_markdown(index=False)}
"""
    with open(COMP_TABLE, "w") as f:
        f.write(content)
        
    # Write LaTeX Table
    latex_table = df.to_latex(index=False)
    with open(os.path.join(OUT_DIR, "master_comparison.tex"), "w") as f:
        f.write(latex_table)
        
    logger.info("Master Comparison Table generated.")
