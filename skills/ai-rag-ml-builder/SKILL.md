---
name: ai-rag-ml-builder
description: Builds diverse RAG systems and ML projects across 15 types including Document Q&A, Semantic Search, Chatbots with Memory, Knowledge Graphs, Text Classification, NER, Sentiment Analysis, Text Summarization, Question Generation, Fact Checking, Citation Finding, Document Clustering, Topic Modeling, Intent Recognition, and Multi-Document RAG. Each project produces a working system with dummy data, vector embeddings, and an interactive HTML demo. Use when building RAG/ML hackathon projects, research demonstrations, or educational examples.
license: MIT
---

# AI RAG ML Builder

This skill builds complete RAG (Retrieval-Augmented Generation) systems and ML projects with working code, dummy data, and interactive demos.

## When to Use This Skill

Use this skill when the user requests:
- RAG system demonstrations or prototypes
- ML/NLP project examples (classification, NER, sentiment, etc.)
- Chatbot systems with memory
- Knowledge graph construction
- Document analysis and retrieval systems
- Text processing pipelines
- Educational ML/RAG demonstrations
- Hackathon-style ML/RAG projects

## Available Resources

### Scripts

**scripts/rag_utils.py**
- `SimpleEmbedder`: TF-IDF-based text embedding (no external APIs)
- `VectorDatabase`: In-memory vector storage and similarity search
- `ConversationMemory`: Conversation history with semantic search
- `chunk_text()`: Split text into overlapping chunks
- `generate_dummy_documents()`: Create test documents
- `cosine_similarity()`: Calculate vector similarity

**scripts/ml_utils.py**
- `TextClassifier`: Naive Bayes text classifier
- `SentimentAnalyzer`: Rule-based sentiment analysis
- `SimpleNER`: Pattern-based named entity recognition
- `IntentClassifier`: Intent classification for chatbots
- `SimpleSummarizer`: Extractive text summarization
- `TopicModeler`: Topic modeling via word co-occurrence
- `generate_training_data()`: Create dummy training data
- `calculate_metrics()`: Classification metrics

### References

**references/rag_patterns.md**
- RAG architecture patterns
- Document processing strategies
- Retrieval optimization techniques
- Context window management
- Memory patterns for chatbots

**references/ml_patterns.md**
- Text classification best practices
- NER implementation patterns
- Sentiment analysis approaches
- Summarization strategies
- Model evaluation techniques

### Assets

**assets/rag_dashboard_template.html**
- Interactive HTML dashboard template
- Pre-styled components
- Responsive design
- Chart and visualization support

## Project Types

This skill supports 15 distinct project types. Each project includes:
1. Complete working system (Python + HTML)
2. Dummy/generated data (no external APIs)
3. Vector embeddings where applicable
4. Interactive demo interface
5. Clear documentation

### 1. Document Q&A System

**Purpose**: Answer questions about document content using RAG

**Implementation Steps**:
1. Generate dummy documents using `rag_utils.generate_dummy_documents()`
2. Create `VectorDatabase` and index documents
3. Build query interface that:
   - Embeds user question
   - Retrieves relevant chunks
   - Formats answer with sources
4. Create HTML demo with question input and answer display

**Key Code Pattern**:
```python
from scripts.rag_utils import VectorDatabase, generate_dummy_documents

# Initialize
vector_db = VectorDatabase()
docs = generate_dummy_documents(num_docs=50, topic="general")

# Index documents
texts = [f"{doc['title']}\n{doc['content']}" for doc in docs]
vector_db.add_documents(texts, docs)

# Query
def answer_question(question):
    results = vector_db.search(question, top_k=3)
    return results
```

### 2. Semantic Search Engine

**Purpose**: Search documents using semantic similarity

**Implementation Steps**:
1. Create document collection with metadata
2. Build vector index
3. Implement search with filtering options (category, date, etc.)
4. Display results with relevance scores
5. Add sorting and ranking features

**Key Features**:
- Semantic similarity scoring
- Metadata filtering
- Result ranking
- Snippet highlighting

### 3. Chatbot with Memory

**Purpose**: Conversational agent with context retention

**Implementation Steps**:
1. Use `ConversationMemory` from rag_utils
2. Store user-bot exchanges
3. Retrieve relevant conversation history
4. Combine with document knowledge base
5. Generate contextual responses

**Key Code Pattern**:
```python
from scripts.rag_utils import ConversationMemory, VectorDatabase

memory = ConversationMemory(max_history=50)
knowledge_base = VectorDatabase()

def chat(user_message):
    # Search conversation history
    past_context = memory.search_history(user_message, top_k=3)
    
    # Search knowledge base
    relevant_docs = knowledge_base.search(user_message, top_k=3)
    
    # Generate response (simulated)
    response = generate_response(user_message, past_context, relevant_docs)
    
    # Store exchange
    memory.add_exchange(user_message, response)
    
    return response
```

### 4. Knowledge Graph Builder

**Purpose**: Extract entities and relationships from text

**Implementation Steps**:
1. Use `SimpleNER` to extract entities
2. Identify relationships between co-occurring entities
3. Build graph structure (nodes = entities, edges = relationships)
4. Visualize as interactive graph
5. Enable querying (e.g., "Show all connections to X")

**Entity Types**:
- PERSON, ORG, EMAIL, PHONE, DATE, MONEY, URL

**Graph Structure**:
```python
graph = {
    "nodes": [{"id": "entity_1", "type": "PERSON", "name": "..."}],
    "edges": [{"source": "entity_1", "target": "entity_2", "type": "WORKS_AT"}]
}
```

### 5. Text Classification

**Purpose**: Categorize text into predefined classes

**Implementation Steps**:
1. Generate training data using `ml_utils.generate_training_data()`
2. Train `TextClassifier`
3. Evaluate on test set
4. Create demo with custom text input
5. Display predictions with confidence scores

**Key Code Pattern**:
```python
from scripts.ml_utils import TextClassifier, generate_training_data

# Generate data
texts, labels = generate_training_data("sentiment", num_samples=200)

# Train
classifier = TextClassifier()
classifier.train(texts[:150], labels[:150])

# Predict
prediction = classifier.predict("This is a test")
probabilities = classifier.predict_proba("This is a test")
```

### 6. Named Entity Recognition

**Purpose**: Identify and extract named entities from text

**Implementation Steps**:
1. Use `SimpleNER` with built-in patterns
2. Process sample texts
3. Highlight entities in display
4. Show entity types and positions
5. Add entity frequency statistics

**Demo Features**:
- Entity highlighting with color coding
- Entity type filtering
- Export extracted entities
- Statistics dashboard

### 7. Sentiment Analysis

**Purpose**: Analyze sentiment of text (positive/negative/neutral)

**Implementation Steps**:
1. Use `SentimentAnalyzer` from ml_utils
2. Process text to extract sentiment
3. Display sentiment label and score
4. Show contributing words
5. Add visualization (gauge or bar chart)

**Key Code Pattern**:
```python
from scripts.ml_utils import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze("This product is amazing!")
# Returns: {"sentiment": "positive", "score": 0.15, ...}
```

### 8. Text Summarization

**Purpose**: Generate concise summaries of documents

**Implementation Steps**:
1. Use `SimpleSummarizer` for extractive summarization
2. Score sentences by importance
3. Select top sentences for summary
4. Maintain readability and coherence
5. Add configurable summary length

**Key Features**:
- Extractive summarization
- Adjustable length
- Sentence scoring display
- Original vs summary comparison

### 9. Question Generation

**Purpose**: Automatically generate questions from text

**Implementation Steps**:
1. Parse input text for key information
2. Extract entities using NER
3. Apply question templates:
   - "Who is {PERSON}?"
   - "What is {ORG}?"
   - "When did {EVENT} occur?"
4. Validate question quality
5. Rank by usefulness

**Template Types**:
- Who/What/When/Where/Why questions
- Yes/No questions
- How questions
- List questions

### 10. Fact Checker

**Purpose**: Verify claims against knowledge base

**Implementation Steps**:
1. Create knowledge base with verified facts
2. Extract claim from input
3. Search for supporting/contradicting evidence
4. Score claim veracity
5. Display evidence and verdict

**Verdict Categories**:
- SUPPORTED: Evidence confirms claim
- REFUTED: Evidence contradicts claim
- NOT ENOUGH INFO: Insufficient evidence

### 11. Citation Finder

**Purpose**: Find and attribute sources for claims

**Implementation Steps**:
1. Index documents with full metadata
2. Extract claims from query
3. Find source documents
4. Generate proper citations
5. Link claims to sources

**Citation Formats**:
- In-text citations
- Reference list
- Relevance scoring
- Source metadata

### 12. Document Clustering

**Purpose**: Group similar documents together

**Implementation Steps**:
1. Generate diverse document set
2. Embed all documents
3. Apply clustering algorithm (use simple k-means)
4. Assign cluster labels
5. Visualize clusters and extract themes

**Key Code Pattern**:
```python
from scripts.rag_utils import VectorDatabase
import numpy as np

# Embed documents
vector_db = VectorDatabase()
vector_db.add_documents(documents)

# Simple clustering
embeddings = np.array(vector_db.vectors)
# Apply k-means logic or group by similarity
```

### 13. Topic Modeling

**Purpose**: Discover topics in document collection

**Implementation Steps**:
1. Use `TopicModeler` from ml_utils
2. Fit on document corpus
3. Extract topic keywords
4. Assign documents to topics
5. Visualize topic distributions

**Key Code Pattern**:
```python
from scripts.ml_utils import TopicModeler

modeler = TopicModeler(num_topics=5, top_words=10)
topics = modeler.fit(documents)

# Display topics
for i, topic in enumerate(topics):
    print(f"Topic {i+1}: {', '.join(topic)}")
```

### 14. Intent Recognition

**Purpose**: Classify user intent in conversations

**Implementation Steps**:
1. Define intent categories (greeting, question, complaint, etc.)
2. Create training examples per intent
3. Train `IntentClassifier`
4. Classify user inputs
5. Show confidence scores

**Common Intents**:
- greeting, farewell
- question, request
- complaint, feedback
- confirmation, cancellation

### 15. Multi-Document RAG

**Purpose**: Answer questions across multiple documents

**Implementation Steps**:
1. Index multiple document sources
2. Implement cross-document search
3. Synthesize information from multiple sources
4. Handle contradictions
5. Attribute to specific sources

**Key Features**:
- Cross-document retrieval
- Source attribution
- Contradiction detection
- Synthesis of multiple perspectives

## Building a Project: Step-by-Step Process

When a user requests a project, follow this process:

### Step 1: Understand Requirements

Ask clarifying questions if needed:
- Which project type?
- Any specific features or customizations?
- Domain or topic (medical, business, general)?
- Preferred output format?

### Step 2: Read Relevant References

Always start by reading the appropriate reference materials:
- For RAG projects: Read `references/rag_patterns.md`
- For ML projects: Read `references/ml_patterns.md`
- For all projects: Understand patterns before coding

### Step 3: Create Python Implementation

Structure the Python code as follows:

```python
"""
{Project Name}
{Brief description}
"""

import sys
sys.path.append('/home/claude')

from scripts.rag_utils import *  # Import as needed
from scripts.ml_utils import *   # Import as needed

# Configuration
CONFIG = {
    'num_documents': 50,
    'num_topics': 5,
    # ... other config
}

# Main system class
class {ProjectName}System:
    def __init__(self):
        # Initialize components
        pass
    
    def process(self, input_data):
        # Main processing logic
        pass
    
    def get_results(self):
        # Return results
        pass

# Demo data generation
def generate_demo_data():
    # Create dummy data
    pass

# Main execution
if __name__ == "__main__":
    system = {ProjectName}System()
    # Demo usage
```

### Step 4: Create Interactive HTML Demo

Use the dashboard template:

1. Copy `assets/rag_dashboard_template.html`
2. Replace placeholders:
   - `{{PROJECT_TITLE}}`
   - `{{PROJECT_DESCRIPTION}}`
   - `{{CONTENT_SECTION}}` - Input forms
   - `{{BUTTON_TEXT}}`
   - `{{RESULTS_SECTION}}` - Results display
   - `{{JAVASCRIPT_CODE}}` - Interactive logic

3. Add project-specific JavaScript for demo interaction

### Step 5: Test and Validate

Ensure:
- Python code runs without errors
- Demo data is generated correctly
- HTML interface is functional
- Results are displayed properly
- No external API calls (all self-contained)

### Step 6: Package and Deliver

Create clear file structure:
```
{project_name}/
├── {project_name}.py        # Main implementation
├── demo.html                 # Interactive demo
├── README.md                 # Documentation
└── requirements.txt          # Dependencies (minimal)
```

## HTML Demo Template Usage

The dashboard template provides:
- Responsive layout
- Pre-styled components
- Loading states
- Result cards
- Charts and visualizations

### Template Placeholders

Replace these in the HTML template:

**{{PROJECT_TITLE}}**: "Document Q&A System"

**{{PROJECT_DESCRIPTION}}**: "Ask questions about documents and get AI-powered answers"

**{{CONTENT_SECTION}}**: Your input forms
```html
<div class="input-section">
    <label>Enter your question:</label>
    <input type="text" id="question" placeholder="What is machine learning?">
</div>
```

**{{BUTTON_TEXT}}**: "Search" or "Analyze" or "Process"

**{{RESULTS_SECTION}}**: Results container (initially empty, filled by JavaScript)

**{{JAVASCRIPT_CODE}}**: Your processing logic
```javascript
function processInput() {
    showLoading();
    const query = document.getElementById('question').value;
    
    // Simulate processing (in real demo, call Python backend)
    setTimeout(() => {
        const results = [
            {title: "Result 1", content: "...", score: 0.95},
            {title: "Result 2", content: "...", score: 0.87}
        ];
        
        displayResults(results);
    }, 1000);
}

function displayResults(results) {
    let html = '<h3>Results</h3>';
    results.forEach(r => {
        html += createResultItem(r.title, r.content, r.score);
    });
    document.getElementById('results').innerHTML = html;
    showResults();
}
```

## Best Practices

### Code Quality

- Write clear, documented code
- Use type hints where helpful
- Handle edge cases
- Include error handling

### Demo Quality

- Use realistic dummy data
- Make UI intuitive
- Show informative results
- Add helpful explanations

### Educational Value

- Comment key algorithms
- Explain design decisions
- Reference best practices
- Suggest improvements

### Self-Contained

- No external API requirements
- All dependencies in scripts/
- Dummy data generation included
- Works offline

## Common Patterns

### Loading Dummy Data

```python
from scripts.rag_utils import generate_dummy_documents

# Generate documents
docs = generate_dummy_documents(
    num_docs=50,
    topic="medical"  # or "business", "general"
)
```

### Vector Search

```python
from scripts.rag_utils import VectorDatabase

# Create and populate
vector_db = VectorDatabase()
vector_db.add_documents(texts, metadata)

# Search
results = vector_db.search(query, top_k=5)
for meta, score in results:
    print(f"{meta['title']}: {score}")
```

### Text Classification

```python
from scripts.ml_utils import TextClassifier

# Train
classifier = TextClassifier()
classifier.train(train_texts, train_labels)

# Predict
label = classifier.predict(new_text)
probs = classifier.predict_proba(new_text)
```

### HTML Result Display

```javascript
function displayResults(results) {
    let html = '<h3>Search Results</h3>';
    
    results.forEach((result, index) => {
        html += `
            <div class="result-item">
                <h4>${result.title}</h4>
                <p>${result.content}</p>
                <span class="score-badge">
                    Score: ${formatScore(result.score)}
                </span>
                <div class="metadata">
                    ${result.category} | ${result.date}
                </div>
            </div>
        `;
    });
    
    document.getElementById('results').innerHTML = html;
    showResults();
}
```

## Example Project Structures

### RAG Project Structure

```
document_qa/
├── document_qa.py
│   ├── DocumentQASystem class
│   ├── generate_demo_documents()
│   ├── process_query()
│   └── main demo
├── demo.html
│   ├── Question input
│   ├── Search button
│   └── Results display
└── README.md
```

### ML Project Structure

```
sentiment_analyzer/
├── sentiment_analyzer.py
│   ├── SentimentSystem class
│   ├── generate_test_data()
│   ├── analyze_text()
│   └── main demo
├── demo.html
│   ├── Text input
│   ├── Analyze button
│   └── Sentiment results
└── README.md
```

## Troubleshooting

### Common Issues

**Import errors**: Ensure `sys.path.append('/home/claude')` at top of Python files

**Empty results**: Check that data is generated and indexed properly

**JavaScript errors**: Verify all functions are defined before use

**Display issues**: Check HTML template placeholders are replaced

## Output Guidelines

When delivering a project:

1. **Create all files in `/home/claude/project_name/`**
2. **Move to outputs**: Copy final files to `/mnt/user-data/outputs/`
3. **Provide instructions**: Include README with usage
4. **Test before delivery**: Verify everything works
5. **Offer enhancements**: Suggest possible improvements

## Customization Options

Users may request:
- Different domains (medical, legal, technical)
- Additional features (export, filtering, etc.)
- Custom styling or branding
- Integration with other systems
- Performance optimizations

Adapt the base patterns to meet these needs while maintaining the core structure.

## Important Notes

- **No real APIs**: All projects use dummy data and local processing
- **Educational focus**: Code should be clear and well-commented
- **Self-contained**: Each project works independently
- **Scalable patterns**: Design allows for future enhancements
- **Best practices**: Follow patterns from reference documents

## Summary

This skill enables rapid development of RAG and ML demonstration projects. By combining the utility scripts, reference patterns, and HTML templates, complete functional systems with interactive demos can be built in minutes. Always read the relevant reference files before starting, use the provided utilities, and follow the step-by-step process for best results.
