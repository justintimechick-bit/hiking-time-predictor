"""Check tracked/indexed publication contents, without printing secret values.

Run after staging and before pushing. This is a focused guard, not a complete
secret scanner. Only empty templates and synthetic fixtures belong in Git.
"""
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\bsb_secret_[A-Za-z0-9_-]{16,}'),
    re.compile(r'\bGOCSPX-[A-Za-z0-9_-]{12,}'),
    re.compile(r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}'),
    re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    re.compile(r'\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b'),
]


def check(path, contents):
    issues = []
    if ((path.startswith('.env') and path != '.env.example')
            or path in ('.streamlit/secrets.toml', 'MODEL_EVALUATION.md', 'BREAK_ANALYSIS.md', 'PLANNED_ROUTES.md')
            or path.startswith(('.auth/', 'private/', 'reports/', 'supabase/.temp/'))
            or (path.startswith('data/') and not (path.endswith('_template.csv') or path.endswith('/.gitkeep')))
            or (path.startswith('models/') and not path.endswith('/.gitkeep'))
            or path.endswith(('.gpx', '.joblib', '.sqlite3'))):
        issues.append('private/generated path')
    text = contents.decode('utf-8', errors='replace')
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        issues.append('possible credential; inspect privately')
    if path.endswith('.ipynb'):
        notebook = json.loads(text)
        if any(c.get('outputs') or c.get('execution_count') is not None
               for c in notebook.get('cells', []) if c.get('cell_type') == 'code'):
            issues.append('saved notebook output/execution counts')
        if any(c.get('attachments') for c in notebook.get('cells', [])):
            issues.append('notebook attachments need review')
    return issues


def main():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    count, failures = 0, []
    for path in filter(None, files):
        # Inspect the index, exactly what the next commit would publish.
        contents = subprocess.check_output(['git', 'show', ':' + path], cwd=ROOT)
        count += 1
        for issue in check(path, contents):
            failures.append(f'{path}: {issue}')
    if not count:
        print('No staged/tracked files. Stage the intended publication first.', file=sys.stderr)
        return 1
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print(f'Publication checks passed for {count} files.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
