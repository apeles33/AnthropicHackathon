"""Flask backend for the Anthropic Hackathon gallery."""

from datetime import datetime
import json
from pathlib import Path

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS


OUTPUT_DIR = Path(__file__).parent.parent / 'output'
SHOWCASE_DIR = Path(__file__).parent.parent / 'showcase'
FRONTEND_BUILD_DIR = Path(__file__).parent / 'frontend' / 'dist'

app = Flask(__name__)
CORS(app)


def get_projects():
    """Scan output and showcase directories for generated projects."""
    projects = []

    for base_dir in [OUTPUT_DIR, SHOWCASE_DIR]:
        if not base_dir.exists():
            continue

        for project_dir in base_dir.iterdir():
            if not project_dir.is_dir() or project_dir.name.startswith('.'):
                continue

            metadata = {}
            metadata_file = project_dir / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as file_handle:
                        metadata = json.load(file_handle)
                except json.JSONDecodeError:
                    metadata = {}

            project_name = project_dir.name
            category = metadata.get('category', 'unknown')
            if category == 'unknown':
                if project_name.startswith('agentic-'):
                    category = 'agentic'
                elif project_name.startswith('rag-'):
                    category = 'rag'
                elif project_name.startswith('trading-'):
                    category = 'trading'
                elif project_name.startswith('research-'):
                    category = 'research'

            created_at = datetime.fromtimestamp(project_dir.stat().st_ctime)

            project_info = {
                'id': project_name,
                'name': metadata.get('name', project_name.replace('-', ' ').title()),
                'category': category,
                'description': metadata.get('description', 'AI-generated hackathon project'),
                'innovation': metadata.get('innovation', 'Novel approach to AI problem-solving'),
                'tech_stack': metadata.get('tech_stack', ['Python', 'Claude API']),
                'wow_factor': metadata.get('wow_factor', 7),
                'complexity': metadata.get('complexity', 'medium'),
                'created_at': created_at.isoformat(),
                'has_demo': (project_dir / 'index.html').exists(),
                'has_readme': (project_dir / 'README.md').exists(),
                'source': 'showcase' if base_dir == SHOWCASE_DIR else 'output',
            }

            projects.append(project_info)

    return projects


@app.route('/api/projects')
def api_projects():
    """Return every generated project."""
    return jsonify(get_projects())


@app.route('/api/stats')
def api_stats():
    """Return summary statistics about the gallery."""
    projects = get_projects()

    stats = {
        'total': len(projects),
        'by_category': {
            'agentic': len([p for p in projects if p['category'] == 'agentic']),
            'rag': len([p for p in projects if p['category'] == 'rag']),
            'trading': len([p for p in projects if p['category'] == 'trading']),
            'research': len([p for p in projects if p['category'] == 'research']),
        },
        'avg_wow_factor': (sum(p['wow_factor'] for p in projects) / len(projects)) if projects else 0,
        'with_demo': len([p for p in projects if p['has_demo']]),
    }

    return jsonify(stats)


@app.route('/project/<project_id>')
def view_project(project_id):
    """Serve a specific project demo."""
    for base_dir in [OUTPUT_DIR, SHOWCASE_DIR]:
        project_dir = base_dir / project_id
        if project_dir.exists():
            index_file = project_dir / 'index.html'
            if index_file.exists():
                return send_from_directory(project_dir, 'index.html')
            return (
                f"<h1>Project: {project_id}</h1><p>No demo available."
                " Check the project directory for code and documentation.</p>"
            )

    return "Project not found", 404


@app.route('/project/<project_id>/<path:filename>')
def project_file(project_id, filename):
    """Serve static assets for a project demo."""
    for base_dir in [OUTPUT_DIR, SHOWCASE_DIR]:
        project_dir = base_dir / project_id
        if project_dir.exists():
            return send_from_directory(project_dir, filename)

    return "File not found", 404


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve the built React frontend (SPA)."""
    if not FRONTEND_BUILD_DIR.exists():
        return (
            "<h1>Frontend build not found</h1>"
            "<p>Run <code>npm install && npm run build</code> inside web/frontend.</p>"
        ), 501

    file_path = FRONTEND_BUILD_DIR / path if path else FRONTEND_BUILD_DIR / 'index.html'
    if path and file_path.exists():
        return send_from_directory(FRONTEND_BUILD_DIR, path)

    return send_from_directory(FRONTEND_BUILD_DIR, 'index.html')


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)

    print('=' * 60)
    print('= Anthropic Hackathon Meta-Builder Gallery')
    print('=' * 60)
    print(f'= Output directory: {OUTPUT_DIR}')
    print(f'= Showcase directory: {SHOWCASE_DIR}')
    if FRONTEND_BUILD_DIR.exists():
        print(f'= Serving frontend build from: {FRONTEND_BUILD_DIR}')
    else:
        print('= Frontend build missing. Run npm run build inside web/frontend.')
    print('= Server running at: http://localhost:8000')
    print('=' * 60)

    app.run(debug=True, host='0.0.0.0', port=8000)
