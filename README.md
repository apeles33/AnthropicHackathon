# 🤖 Anthropic Hackathon: Meta-Builder System

> **Meta-Innovation:** An AI agent system that autonomously generates hackathon-quality projects

[![Demo](https://img.shields.io/badge/Demo-Live-green)](http://localhost:5000)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Built with Claude](https://img.shields.io/badge/Built%20with-Claude-orange)](https://www.anthropic.com)

## 🎯 What Is This?

A sophisticated multi-agent system built with **Claude Code** that:
- 🤖 Reads specialized **Skills** (packaged best practices)
- 🎨 **Orchestrates** project generation across 4 domains
- 🏗️ **Spawns** specialized builder sub-agents
- ✨ **Generates** 20-40 hackathon-quality projects autonomously
- 🎪 **Displays** them in a live, interactive gallery

**The Innovation:** Not just building one project—building a system that builds many projects.

## 🏆 Project Categories

### 1. Agentic AI & MCP
Multi-agent systems, agent orchestration, novel MCP integrations
- Multi-Agent Debate Systems
- Meta-Agents (agents that build agents)
- Workflow Orchestrators
- Task Automation Agents

### 2. AI RAG & Machine Learning
Production-ready RAG systems, ML applications, intelligent assistants
- Document Q&A Systems
- Semantic Search Engines
- Few-Shot Classifiers
- Knowledge Graph Builders

### 3. Statistics, Trading & Analytics
AI-powered quantitative tools, trading algorithms, risk analytics
- Portfolio Optimizers
- Sentiment Trading Systems
- Risk Dashboards
- Algorithmic Backtesting

### 4. Research & Innovation
Cutting-edge ML implementations, paper recreations, novel architectures
- Interactive Transformer Visualizations
- GAN Art Generators
- Model Interpretability Tools
- Architecture Explorers

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Anthropic API key
- Claude Code (in Cursor)
- Flask

### Installation
```bash
# Clone repository
git clone https://github.com/benkassan/AnthropicHackathon.git
cd AnthropicHackathon

# Install dependencies
cd web
pip install -r requirements.txt

# Set up environment
export ANTHROPIC_API_KEY='your-key-here'

# Run gallery server
python app.py
```

### Using the System

1. **Upload Skills** to Claude Code (`skills/` directory)
2. **Create Agents** (Orchestrator + 4 Builders) in Claude Code `/agents` tab
3. **Generate Projects** via Claude Code chat:
   ```
   @Orchestrator Generate 8 diverse, impressive projects
   ```
4. **View Gallery** at `http://localhost:5000`

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Claude Code (Cursor IDE)                               │
│                                                          │
│  /agents Tab:                                           │
│  ├─ Orchestrator Agent                                  │
│  │  └─ Strategic project planning & coordination        │
│  │                                                       │
│  └─ 4 Builder Sub-Agents:                              │
│     ├─ builder-agentic    (Agentic AI & MCP)          │
│     ├─ builder-rag        (RAG & ML)                   │
│     ├─ builder-trading    (Trading & Analytics)        │
│     └─ builder-research   (Research & Innovation)      │
│                                                          │
│  /skills Tab:                                           │
│  ├─ agentic-ai-mcp-builder.zip                         │
│  ├─ ai-rag-ml-builder.zip                              │
│  ├─ stats-trading-analytics-builder.zip                │
│  └─ research-paper-builder.zip                         │
└─────────────────────────────────────────────────────────┘
                           ↓
                  Generates Projects
                           ↓
┌─────────────────────────────────────────────────────────┐
│  output/                                                 │
│  ├─ agentic-{project-name}/                             │
│  ├─ rag-{project-name}/                                 │
│  ├─ trading-{project-name}/                             │
│  └─ research-{project-name}/                            │
└─────────────────────────────────────────────────────────┘
                           ↓
                    Displayed In
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Web Gallery (Flask App)                                 │
│  - Interactive project cards                             │
│  - Filter by category                                    │
│  - Sort by wow factor                                    │
│  - Click to demo                                         │
└─────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
AnthropicHackathon/
├── skills/                          # Domain-specific skills (ZIPs)
│   ├── agentic-ai-mcp-builder.zip
│   ├── ai-rag-ml-builder.zip
│   ├── stats-trading-analytics-builder.zip
│   └── research-paper-builder.zip
│
├── output/                          # Generated projects (gitignored)
│   ├── agentic-*/
│   ├── rag-*/
│   ├── trading-*/
│   └── research-*/
│
├── web/                            # Gallery web application
│   ├── app.py                      # Flask server
│   ├── requirements.txt
│   └── templates/
│       └── gallery.html            # Interactive gallery UI
│
├── agent-prompts/                  # Agent system prompts (for reference)
│   ├── orchestrator.md
│   ├── builder-agentic.md
│   ├── builder-rag.md
│   ├── builder-trading.md
│   └── builder-research.md
│
├── docs/                           # Documentation
│   ├── DEMO_SCRIPT.md             # Presentation guide
│   ├── ARCHITECTURE.md            # System design
│   └── SKILLS_GUIDE.md            # How to create skills
│
├── showcase/                       # Example projects for demo
├── logs/                           # Generation logs (gitignored)
├── .gitignore
├── LICENSE
└── README.md
```

## 🎬 Live Demo

**Watch the system generate a project in real-time:**

1. Open Claude Code in Cursor
2. Chat with `@Orchestrator`: "Generate an impressive multi-agent debate system"
3. Orchestrator reads skills, creates spec, spawns `@builder-agentic`
4. Builder creates complete project in `output/`
5. Refresh gallery at `localhost:5000` to see new project
6. Click project card to view interactive demo

## 🏆 Why This Project Wins

### 1. Meta-Level Innovation
Not just *one* hackathon project—a *system that generates* hackathon projects

### 2. Technical Depth
- Multi-agent coordination
- Skill-based knowledge transfer
- Domain-specific builders
- Production-quality code generation

### 3. Scale & Variety
- 4 distinct categories
- 55 total project types
- Can generate 100s overnight
- Each project is unique

### 4. Practical Value
- Accelerates research & prototyping
- Democratizes AI development
- Educational tool for learning AI patterns
- Framework for rapid experimentation

### 5. Impressive Demo
- Live generation (watch it work)
- Interactive gallery (judges can explore)
- Professional polish (production-ready)
- Clear wow moments (agents coordinating)

## 🎓 Technical Approach

### Skills Framework
Each skill is a packaged knowledge base:
- SKILL.md: Project types, workflows, patterns
- scripts/: Utility functions and templates
- references/: Best practices and examples
- assets/: UI templates and boilerplate

### Agent Architecture
- **Orchestrator:** Strategic planning, spec creation, coordination
- **Builders:** Domain specialists that read skills and generate projects
- **Communication:** Structured specifications, progress monitoring
- **Quality:** Each builder optimized for hackathon-grade output

### Quality Assurance
Every generated project includes:
- ✅ Working code (200-800 lines)
- ✅ Interactive UI (professional design)
- ✅ Documentation (README, demo script)
- ✅ Metadata (innovation, tech stack, demo hooks)
- ✅ Demo readiness (can present immediately)

## 🚧 Future Enhancements

- [ ] **Phase II:** Reviewer agent that ranks projects
- [ ] **Self-improvement:** Agents learn from successful projects
- [ ] **Multi-modal:** Generate video demos automatically
- [ ] **Deployment:** Auto-deploy projects to cloud
- [ ] **Collaboration:** Multiple users generating together
- [ ] **Templates:** Export as project templates

## 👥 Team

- **Ben** - Agent architecture, prompt engineering, domain expertise
- **Adam** - Systems engineering, web development, infrastructure

## 📜 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- Built with [Claude](https://www.anthropic.com/claude) by Anthropic
- Uses [Claude Code](https://docs.anthropic.com/claude-code) for agent orchestration
- Skills framework inspired by best practices in AI engineering
- Thanks to Anthropic for the hackathon opportunity!

## 📞 Contact

Questions? Reach out:
- GitHub Issues: [Project Issues](https://github.com/benkassan/AnthropicHackathon/issues)
- Demo Video: [Coming Soon]

---

**Built for Anthropic Hackathon 2024** 🚀