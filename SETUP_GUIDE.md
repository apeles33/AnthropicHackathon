# 🚀 Setup Guide - Anthropic Hackathon Meta-Builder

This guide walks you through setting up the complete meta-builder system in Claude Code.

## ✅ What's Already Done

- ✅ Project structure created
- ✅ Flask web gallery built
- ✅ 5 domain skills prepared (55 project types total)
- ✅ 5 agent system prompts written
- ✅ Documentation completed
- ✅ Everything committed to git

## 📋 Next Steps

### Step 1: Push to GitHub

First, push the code so your partner can access it:

```bash
cd /Users/benkassan/AnthropicHackathon/AnthropicHackathon
git push origin main
```

You'll need to authenticate with GitHub. Once pushed, your partner can clone:
```bash
git clone https://github.com/apeles33/AnthropicHackathon.git
```

---

### Step 2: Upload Skills to Claude Code

1. Open Claude Code (in Cursor IDE)
2. Click the **Skills** tab in the left sidebar
3. Click **"+ Add Skill"** button
4. Upload each ZIP file from `skills/` directory:
   - `agentic-ai-mcp-builder.zip`
   - `ai-rag-ml-builder.zip`
   - `trading-analytics-builder.zip`
   - `quant-research-builder.zip`
   - `ml-research-builder.zip`

5. Verify all 5 skills appear in your Skills list

**Tip:** You can view skill contents by clicking on them in the Skills tab.

---

### Step 3: Create Agents in Claude Code

Now create 5 agents using the prompts in `agent-prompts/`:

#### Agent 1: Orchestrator

1. Click the **Agents** tab in left sidebar
2. Click **"+ New Agent"**
3. **Name:** `Orchestrator`
4. **System Prompt:** Copy ALL content from `agent-prompts/orchestrator.md`
5. **Skills:** Select ALL 5 skills:
   - ✅ agentic-ai-mcp-builder
   - ✅ ai-rag-ml-builder
   - ✅ trading-analytics-builder
   - ✅ quant-research-builder
   - ✅ ml-research-builder
6. Click **"Create Agent"**

#### Agent 2: builder-agentic

1. Click **"+ New Agent"**
2. **Name:** `builder-agentic`
3. **System Prompt:** Copy ALL content from `agent-prompts/builder-agentic.md`
4. **Skills:** Select ONLY:
   - ✅ agentic-ai-mcp-builder
5. Click **"Create Agent"**

#### Agent 3: builder-rag

1. Click **"+ New Agent"**
2. **Name:** `builder-rag`
3. **System Prompt:** Copy ALL content from `agent-prompts/builder-rag.md`
4. **Skills:** Select ONLY:
   - ✅ ai-rag-ml-builder
5. Click **"Create Agent"**

#### Agent 4: builder-trading

1. Click **"+ New Agent"**
2. **Name:** `builder-trading`
3. **System Prompt:** Copy ALL content from `agent-prompts/builder-trading.md`
4. **Skills:** Select ONLY:
   - ✅ trading-analytics-builder
5. Click **"Create Agent"**

#### Agent 5: builder-research

1. Click **"+ New Agent"**
2. **Name:** `builder-research`
3. **System Prompt:** Copy ALL content from `agent-prompts/builder-research.md`
4. **Skills:** Select BOTH:
   - ✅ quant-research-builder
   - ✅ ml-research-builder
5. Click **"Create Agent"**

---

### Step 4: Start the Web Gallery

In a separate terminal, start the Flask gallery server:

```bash
cd /Users/benkassan/AnthropicHackathon/AnthropicHackathon/web
pip install -r requirements.txt
python app.py
```

The gallery will be available at: **http://localhost:5000**

Keep this running in the background. It will automatically detect new projects as they're generated.

---

### Step 5: Test the System

Now test the complete system:

#### Test 1: Simple Generation

In Claude Code chat, type:

```
@Orchestrator Generate a test project to verify the system is working
```

The Orchestrator should:
1. Read relevant skills
2. Create a project specification
3. Spawn the appropriate builder agent
4. The builder creates the project in `output/`

Check the `output/` directory and refresh the gallery to see the project.

#### Test 2: Full Portfolio Generation

Once the test works, generate a full portfolio:

```
@Orchestrator Generate 8 diverse, impressive hackathon projects across all 4 categories. Make them creative and maximize wow factor!
```

The Orchestrator should:
1. Plan a balanced portfolio (2 per category)
2. Spawn 8 builder agents with detailed specs
3. Projects appear in `output/` directory
4. Gallery updates automatically

---

### Step 6: Monitor Results

**Check Progress:**

1. **File System:** Watch `output/` directory for new project folders
2. **Gallery:** Refresh http://localhost:5000 to see projects appear
3. **Claude Code Chat:** Monitor builder agents' progress messages

**Each completed project should have:**
```
output/{category}-{project-name}/
├── index.html          # Interactive demo
├── {main}.py           # Core implementation
├── README.md           # Documentation
├── metadata.json       # Project info
├── requirements.txt    # Dependencies
└── [other files]       # Domain-specific files
```

**View projects:**
- Click project cards in gallery
- Open `index.html` files directly in browser
- Read README.md files for usage instructions

---

### Step 7: Generate More Projects

Once satisfied with initial results, scale up:

#### Generate 16 Projects (4 per category):
```
@Orchestrator Generate 16 hackathon-quality projects - 4 per category. Be creative and ensure diversity!
```

#### Generate 24 Projects (6 per category):
```
@Orchestrator Generate 24 impressive projects across all categories. Maximize innovation and variety!
```

#### Target Specific Categories:
```
@Orchestrator Generate 8 agentic AI projects with high wow factor

@Orchestrator Generate 6 advanced trading and quant research projects

@Orchestrator Generate 10 cutting-edge ML/RAG projects
```

---

## 🎯 Success Criteria

Your system is working correctly when:

- ✅ Orchestrator reads all 5 skills before planning
- ✅ Orchestrator creates detailed project specifications
- ✅ Builder agents generate complete project directories
- ✅ Projects appear in web gallery automatically
- ✅ Demo UIs are interactive and polished
- ✅ README files have clear instructions
- ✅ metadata.json files are complete
- ✅ Projects could win a hackathon on their own

---

## 🐛 Troubleshooting

### Issue: Agent not found
**Solution:** Make sure agent name matches exactly (case-sensitive):
- `@Orchestrator` (capital O)
- `@builder-agentic` (lowercase b, with hyphen)
- etc.

### Issue: Skill not loading
**Solution:**
1. Check skill was uploaded (visible in Skills tab)
2. Verify skill is assigned to agent in agent settings
3. Try re-uploading the ZIP file

### Issue: Projects not appearing in gallery
**Solution:**
1. Check `output/` directory exists
2. Verify Flask app is running
3. Refresh browser (F5)
4. Check Flask console for errors

### Issue: Builder generates incomplete projects
**Solution:**
1. Check builder's system prompt was copied correctly
2. Ensure builder has access to correct skill(s)
3. Verify specification from Orchestrator was detailed
4. Re-generate the specific project

---

## 📊 Expected Results

After generating 24 projects, you should have:

- **6 Agentic AI Projects** (agents, MCP, multi-agent systems)
- **6 RAG/ML Projects** (semantic search, classification, Q&A)
- **6 Trading Projects** (algorithms, backtesting, risk analysis)
- **6 Research Projects** (quant models, ML architectures, papers)

**Portfolio Characteristics:**
- Diverse project types (no duplicates)
- High wow factors (8-10/10)
- Professional presentation
- Complete documentation
- Interactive demos

---

## 🎬 Demo Preparation

Before presenting to judges:

1. **Select Best Projects:** Choose 3-4 most impressive for live demo
2. **Test Demos:** Open each `index.html` and verify it works
3. **Prepare Talking Points:** Use innovation angles from metadata.json
4. **Practice Live Generation:** Show generating a project in real-time
5. **Gallery Ready:** Ensure gallery shows all projects beautifully
6. **Backup Screenshots:** Take screenshots in case of connectivity issues

---

## 🚀 Final Deployment

Once you have 20+ quality projects:

### Commit Generated Projects (Optional - Best Ones Only)
```bash
# Create showcase directory with 4 best projects
mkdir -p showcase
cp -r output/agentic-multi-agent-debate-system showcase/
cp -r output/rag-intelligent-legal-assistant showcase/
cp -r output/trading-sentiment-analyzer showcase/
cp -r output/research-transformer-visualization showcase/

git add showcase/
git commit -m "feat: Add 4 showcase projects for demo"
git push origin main
```

**Note:** Don't commit all projects to git (output/ is gitignored). Only commit 3-4 showcase examples.

### Tag the Release
```bash
git tag -a v1.0.0 -m "Anthropic Hackathon Submission v1.0.0

Complete meta-builder system:
- 5 skills with 55 project types
- 5 agents (Orchestrator + 4 builders)
- 24+ generated hackathon projects
- Interactive web gallery
- Professional documentation

Ready for demo!"

git push origin v1.0.0
```

---

## 🏆 You're Ready!

The system is now complete and ready to:
- Generate unlimited hackathon projects
- Impress judges with live demonstrations
- Showcase the meta-innovation concept
- Win the Anthropic Hackathon!

**Good luck!** 🚀
