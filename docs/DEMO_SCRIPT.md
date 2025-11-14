# Demo Script for Judges

## Setup (Before Demo)
- [ ] Gallery running at `localhost:5000`
- [ ] Claude Code open with agents loaded
- [ ] Browser with gallery open
- [ ] Backup screenshots ready
- [ ] Test internet connection
- [ ] 15-20 projects pre-generated

## 3-Minute Pitch Structure

### Slide 1: The Problem (30 seconds)
"Building hackathon projects takes 6-12 hours. Testing multiple approaches takes days. What if we could generate dozens of high-quality prototypes automatically?"

**Key points:**
- Research/prototyping is slow
- Hard to explore many ideas quickly
- Each approach requires full implementation

### Slide 2: Our Solution (30 seconds)
"We built a meta-system: AI agents that autonomously generate hackathon-quality projects."

**Show architecture diagram:**
- Orchestrator plans portfolio
- 4 specialized builders (agents, RAG, trading, research)
- Each generates production-ready projects
- Gallery displays all results

### Slide 3: Live Demo (90 seconds)

**Part 1: Show Gallery (30s)**
1. Open `localhost:5000`
2. "Here are 20+ projects, all generated autonomously"
3. Quickly show variety:
   - Filter by category
   - Show wow factors
   - Mention 55 total project types

**Part 2: Live Generation (60s)**
1. Switch to Claude Code
2. "@Orchestrator generate an impressive multi-agent debate system"
3. Watch orchestrator:
   - Read skill
   - Create detailed spec
   - Spawn builder
4. Builder creates project
5. "In 2-3 minutes, we have a complete hackathon project"
6. Refresh gallery ’ new project appears
7. Click project ’ show interactive demo

**Wow moment:**
"Each of these projects could win a hackathon on its own. The innovation is that AGENTS generated them."

### Slide 4: Impact & Future (30 seconds)

**Impact:**
- Scales to 100s of projects overnight
- Accelerates research and prototyping
- Democratizes AI development
- Educational tool for learning patterns

**Future:**
- Reviewer agent for automatic ranking
- Self-improvement from successful projects
- Multi-modal (generate videos, docs)
- Collaborative generation

## Q&A Preparation

**Q: "How is this different from GitHub Copilot or code generation?"**
A: "Copilot helps you write code. We generate complete hackathon projectscode, UI, documentation, demo-ready. It's full-stack autonomous creation, not code completion."

**Q: "Are the generated projects actually good?"**
A: "Each is optimized for hackathon judges: innovation, technical merit, polish, completeness. We can demo any project live. [Click on a project to show]"

**Q: "What's the technical innovation?"**
A: "Three levels: (1) Skills framework for knowledge transfer, (2) Multi-agent coordination with specialized builders, (3) Quality optimizationeach agent is trained to generate hackathon-winning projects."

**Q: "How long did this take to build?"**
A: "5 hours for the meta-system. It then generated 20 projects in 2 hours. That's 22 hackathon projects total."

**Q: "Can it generate any type of project?"**
A: "Currently 4 domains, 55 types. But the framework is extensibleadd new skills, spawn new builders. Could expand to web dev, mobile, embedded, etc."

**Q: "What if judges want to see the code?"**
A: "Every project includes clean, commented code. [Open a project folder, show structure]. We can walk through any implementation."

## Technical Details (If Asked)

**Architecture:**
- Claude Code for agent orchestration
- Skills framework (ZIPs with best practices)
- 1 Orchestrator + 4 domain builders
- Flask gallery for visualization
- All generated projects in output/

**Quality Control:**
- Each builder optimized for hackathon criteria
- Detailed specifications from orchestrator
- Standard project structure enforced
- Demo scripts included
- Metadata tracking

**Performance:**
- Project generation: 2-5 minutes each
- Can generate in parallel (future)
- Gallery updates in real-time
- All projects <50MB

## Backup Plan

If live demo fails:
1. **Show video** of live generation (record beforehand)
2. **Gallery screenshots** showing projects
3. **Walk through pre-generated project** in detail
4. **Show code** directly in IDE

## Post-Demo

After presenting:
- Share GitHub repo link
- Offer to demo specific projects judges want to see
- Be available for technical deep-dives
- Gather feedback

## Timing Reminders

- **2:30** - Start wrapping up
- **3:00** - End with impact statement
- **Keep energy high** - This is exciting!
- **Smile and be enthusiastic** - Show passion for the project
