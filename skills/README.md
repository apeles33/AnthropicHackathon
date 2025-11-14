# Skills Directory

This directory contains the 5 domain-specific skills used by builder agents to generate hackathon-quality projects.

## 📦 Available Skills

### 1. agentic-ai-mcp-builder.zip
**Category:** Agentic AI & MCP
**Project Types:** 15
**Focus:** Agent architectures, multi-agent systems, MCP integrations

**Projects Include:**
- Task Automation Agent
- Code Review Agent
- Documentation Generator
- Research Assistant
- Content Creator Agent
- Data Analysis Agent
- Multi-Agent Debate System
- Agent Builder (Meta-Agent)
- File Organizer Agent
- Meeting Summarizer
- Code Refactoring Agent
- Personal Wiki Agent
- SQL Query Generator
- API Documentation Agent
- Workflow Orchestrator

**Key Features:**
- Complete agent implementations with system prompts
- Interactive HTML demos
- Observable reasoning and decision-making
- Tool/function calling capabilities
- Example interactions for each agent type

---

### 2. ai-rag-ml-builder.zip
**Category:** RAG & Machine Learning
**Project Types:** 15
**Focus:** RAG systems, semantic search, ML applications

**Projects Include:**
- Document Q&A System
- Semantic Search Engine
- Chatbot with Memory
- Knowledge Graph Builder
- Text Classification
- Named Entity Recognition (NER)
- Sentiment Analysis
- Text Summarization
- Question Generation
- Fact Checking System
- Citation Finding
- Document Clustering
- Topic Modeling
- Intent Recognition
- Multi-Document RAG

**Key Features:**
- Vector embeddings and similarity search
- In-memory vector databases
- Dummy data generation (no external APIs)
- Interactive dashboards
- Complete ML/NLP pipelines

---

### 3. trading-analytics-builder.zip
**Category:** Trading & Analytics
**Project Types:** 15
**Focus:** Trading algorithms, risk analytics, portfolio optimization

**Projects Include:**
- Momentum Bot
- Mean Reversion Strategy
- Sentiment Analyzer
- Options Pricer
- Portfolio Optimizer
- Market Making Simulator
- Arbitrage Scanner
- Risk Dashboard
- Technical Indicator Suite
- Backtesting Engine
- Market Microstructure Analysis
- Pairs Trading System
- Volatility Forecaster
- High-Frequency Simulator
- Multi-Asset Allocator

**Key Features:**
- Working trading algorithms
- Backtest results and visualizations
- Performance metrics (Sharpe, drawdown, etc.)
- Risk analysis tools
- Professional HTML presentations

---

### 4. quant-research-builder.zip
**Category:** Research & Innovation (Quantitative)
**Project Types:** 10
**Focus:** Quantitative finance research, factor models, risk analytics

**Projects Include:**
- Factor Models (Fama-French, momentum, value)
- Portfolio Optimization (Markowitz, Black-Litterman, risk parity)
- Risk Analytics (VaR, CVaR, stress testing)
- Market Microstructure (order book dynamics, liquidity)
- Options Pricing (Greeks, vol surfaces)
- Algorithmic Trading Strategies
- Time Series Forecasting (ARIMA, GARCH, LSTM)
- Volatility Modeling
- High-Frequency Analysis
- Quantamental Research

**Key Features:**
- Academic literature references
- PyTorch/NumPy implementations
- Synthetic and historical data
- Research-quality visualizations
- Complete backtest frameworks

---

### 5. ml-research-builder.zip
**Category:** Research & Innovation (Machine Learning)
**Project Types:** 15
**Focus:** ML/DL research, paper recreations, novel architectures

**Projects Include:**
- Paper Recreation (arXiv experiments)
- Architecture from Scratch (ResNet, Transformer, U-Net)
- Fine-Tuning Experiments (BERT, GPT, ViT)
- Training Innovations (Mixup, curriculum learning, SAM)
- Interpretability Studies (attention visualization, Grad-CAM, SHAP)
- Synthetic Data Generation (GANs, VAEs, diffusion)
- RL Environments
- Loss Function Design
- Model Compression (quantization, pruning)
- Few-Shot Learning
- Transfer Learning
- Neural Architecture Search
- Adversarial ML (FGSM, PGD)
- Multi-Modal Fusion
- Continual Learning

**Key Features:**
- PyTorch implementations
- Training visualizations
- Model checkpoints
- Experiment tracking
- Research-quality presentations

---

## 📊 Total Project Coverage

**55 Distinct Project Types** across 5 skills:
- **Agentic AI & MCP:** 15 types
- **RAG & ML:** 15 types
- **Trading & Analytics:** 15 types
- **Quant Research:** 10 types
- **ML Research:** 15 types

## 🏗️ Skill Structure

Each skill ZIP contains:

```
skill-name/
├── SKILL.md              # Main documentation with project types
├── scripts/              # Utility functions
│   ├── *_utils.py       # Helper functions for the domain
│   └── *_patterns.py    # Implementation patterns
├── references/           # Best practices and patterns
│   ├── *_patterns.md    # Design patterns
│   └── project_ideas.md # Project variations
└── assets/               # Templates and boilerplate
    └── *_template.html  # UI templates
```

## 🔧 Using Skills in Claude Code

### Step 1: Upload Skills
1. Open Claude Code in Cursor
2. Click **Skills** tab in sidebar
3. Click **"Add Skill"**
4. Upload each ZIP file:
   - `agentic-ai-mcp-builder.zip`
   - `ai-rag-ml-builder.zip`
   - `trading-analytics-builder.zip`
   - `quant-research-builder.zip`
   - `ml-research-builder.zip`
5. Verify all 5 skills appear in the Skills list

### Step 2: Assign Skills to Agents

When creating agents in the `/agents` tab:

**Orchestrator:**
- Assign: ALL 5 skills
- Purpose: Reads all skills to plan diverse project portfolios

**builder-agentic:**
- Assign: `agentic-ai-mcp-builder`
- Generates: Agent and MCP projects

**builder-rag:**
- Assign: `ai-rag-ml-builder`
- Generates: RAG and ML projects

**builder-trading:**
- Assign: `trading-analytics-builder`
- Generates: Trading and analytics projects

**builder-research:**
- Assign: `quant-research-builder` AND `ml-research-builder`
- Generates: Both quantitative finance and ML research projects

## 🎯 Quality Standards

Every generated project includes:
- ✅ Working code (200-800 lines)
- ✅ Interactive UI (HTML demo)
- ✅ Documentation (README with setup)
- ✅ Metadata (JSON with project info)
- ✅ Demo readiness (can present immediately)

## 📝 Skill Updates

If you update a skill:
1. Update the ZIP file in this directory
2. Re-upload to Claude Code Skills tab
3. Update version in metadata
4. Commit changes with clear description

## 🚀 Next Steps

After uploading skills:
1. Create 5 agents using prompts from `agent-prompts/`
2. Test with: `@Orchestrator Generate a test project`
3. Generate portfolio: `@Orchestrator Generate 8 diverse projects across all categories`
4. View results in web gallery at `localhost:5000`
