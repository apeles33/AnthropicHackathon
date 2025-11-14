"""
Flask Web Gallery for Anthropic Hackathon Meta-Builder System
Displays generated projects with filtering, sorting, and interactive demos
"""

from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
SHOWCASE_DIR = Path(__file__).parent.parent / 'showcase'

def get_projects():
    """Scan output and showcase directories for generated projects"""
    projects = []

    # Scan both output and showcase directories
    for base_dir in [OUTPUT_DIR, SHOWCASE_DIR]:
        if not base_dir.exists():
            continue

        for project_dir in base_dir.iterdir():
            if not project_dir.is_dir() or project_dir.name.startswith('.'):
                continue

            # Read metadata if available
            metadata_file = project_dir / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except:
                    metadata = {}
            else:
                metadata = {}

            # Extract category from directory name prefix
            project_name = project_dir.name
            category = 'unknown'
            if project_name.startswith('agentic-'):
                category = 'agentic'
            elif project_name.startswith('rag-'):
                category = 'rag'
            elif project_name.startswith('trading-'):
                category = 'trading'
            elif project_name.startswith('research-'):
                category = 'research'

            # Get creation time
            created_at = datetime.fromtimestamp(project_dir.stat().st_ctime)

            # Build project info
            project_info = {
                'id': project_name,
                'name': metadata.get('name', project_name.replace('-', ' ').title()),
                'category': metadata.get('category', category),
                'description': metadata.get('description', 'AI-generated hackathon project'),
                'innovation': metadata.get('innovation', 'Novel approach to AI problem-solving'),
                'tech_stack': metadata.get('tech_stack', ['Python', 'Claude API']),
                'wow_factor': metadata.get('wow_factor', 7),
                'complexity': metadata.get('complexity', 'medium'),
                'created_at': created_at.isoformat(),
                'has_demo': (project_dir / 'index.html').exists(),
                'has_readme': (project_dir / 'README.md').exists(),
                'source': 'showcase' if base_dir == SHOWCASE_DIR else 'output'
            }

            projects.append(project_info)

    return projects

@app.route('/')
def index():
    """Main gallery page"""
    return render_template('gallery.html')

@app.route('/api/projects')
def api_projects():
    """API endpoint to list all projects"""
    projects = get_projects()
    return jsonify(projects)

@app.route('/api/stats')
def api_stats():
    """API endpoint for gallery statistics"""
    projects = get_projects()

    stats = {
        'total': len(projects),
        'by_category': {
            'agentic': len([p for p in projects if p['category'] == 'agentic']),
            'rag': len([p for p in projects if p['category'] == 'rag']),
            'trading': len([p for p in projects if p['category'] == 'trading']),
            'research': len([p for p in projects if p['category'] == 'research']),
        },
        'avg_wow_factor': sum(p['wow_factor'] for p in projects) / len(projects) if projects else 0,
        'with_demo': len([p for p in projects if p['has_demo']]),
    }

    return jsonify(stats)

@app.route('/project/<project_id>')
def view_project(project_id):
    """View a specific project"""
    # Check both directories
    for base_dir in [OUTPUT_DIR, SHOWCASE_DIR]:
        project_dir = base_dir / project_id
        if project_dir.exists():
            index_file = project_dir / 'index.html'
            if index_file.exists():
                return send_from_directory(project_dir, 'index.html')
            else:
                return f"<h1>Project: {project_id}</h1><p>No demo available. Check the project directory for code and documentation.</p>"

    return "Project not found", 404

@app.route('/project/<project_id>/<path:filename>')
def project_file(project_id, filename):
    """Serve project files (for demos that reference other files)"""
    for base_dir in [OUTPUT_DIR, SHOWCASE_DIR]:
        project_dir = base_dir / project_id
        if project_dir.exists():
            return send_from_directory(project_dir, filename)

    return "File not found", 404

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("=€ Anthropic Hackathon Meta-Builder Gallery")
    print("=" * 60)
    print(f"=Á Output directory: {OUTPUT_DIR}")
    print(f"<ª Showcase directory: {SHOWCASE_DIR}")
    print(f"< Server running at: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
