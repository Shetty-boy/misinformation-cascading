import os
import re
import subprocess
from pathlib import Path

# 1. Collect all MD files
md_files = []
for root, dirs, files in os.walk('docs'):
    for f in files:
        if f.endswith('.md'):
            md_files.append(os.path.join(root, f))
for root, dirs, files in os.walk('logs'):
    for f in files:
        if f.endswith('.md'):
            md_files.append(os.path.join(root, f))

renames = {} # old_path -> new_path

for path in md_files:
    p = Path(path)
    parent = p.parent.name
    if parent.startswith('phase'):
        parts = parent.split('_')
        prefix_parts = []
        for part in parts:
            if part.startswith('phase') or part.isdigit():
                prefix_parts.append(part)
            else:
                break
        prefix = "_".join(prefix_parts)
        
        if not p.name.startswith(prefix):
            new_name = f"{prefix}_{p.name}"
            new_path = p.with_name(new_name)
            renames[path] = str(new_path)

# Collect all files that could contain references (src, docs, logs)
all_text_files = []
for d in ['src', 'docs', 'logs']:
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.endswith('.md') or f.endswith('.py') or f.endswith('.txt') or f.endswith('.csv'):
                all_text_files.append(os.path.join(root, f))

# Update references in all files
for fpath in all_text_files:
    try:
        with open(fpath, 'r') as f:
            content = f.read()
    except Exception:
        continue
        
    new_content = content
    # Replace full paths first
    for old_p, new_p in renames.items():
        new_content = new_content.replace(old_p, new_p)
        # Replace ../../logs/... style paths
        old_rel = old_p.replace('docs/', '../../docs/').replace('logs/', '../../logs/')
        new_rel = new_p.replace('docs/', '../../docs/').replace('logs/', '../../logs/')
        new_content = new_content.replace(old_rel, new_rel)

    # For files in the same directory, replace bare basenames
    # e.g., if we are inside docs/phase01_data_acquisition/implementation_notes.md, replace "design_doc.md" with "phase01_design_doc.md"
    p = Path(fpath)
    parent = p.parent.name
    if parent.startswith('phase'):
        for old_p, new_p in renames.items():
            if Path(old_p).parent == p.parent:
                old_base = os.path.basename(old_p)
                new_base = os.path.basename(new_p)
                # Safe regex replacement for basenames
                new_content = re.sub(r'(?<![A-Za-z0-9_/-])' + re.escape(old_base) + r'(?![A-Za-z0-9_.-])', new_base, new_content)

    if new_content != content:
        with open(fpath, 'w') as f:
            f.write(new_content)

# Rename files using git mv
for old_p, new_p in renames.items():
    print(f"git mv {old_p} {new_p}")
    subprocess.run(["git", "mv", old_p, new_p], check=True)

