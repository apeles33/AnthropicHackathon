# ML Patterns and Best Practices

This document provides patterns and best practices for building machine learning systems, with a focus on NLP and text-based tasks.

## Text Classification Patterns

### Baseline Approaches

**Naive Bayes**
- Fast and simple
- Works well with small datasets
- Assumes feature independence
- Good baseline for text classification

**Logistic Regression**
- Linear decision boundaries
- Interpretable coefficients
- Fast training and inference
- Robust baseline

**TF-IDF + Classifier**
```python
# Standard pipeline
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)
classifier = NaiveBayes()
classifier.fit(X, labels)
```

### Feature Engineering

**Text Features**
- Word counts and frequencies
- TF-IDF scores
- N-grams (bigrams, trigrams)
- Character n-grams
- Word embeddings

**Statistical Features**
- Document length
- Average word length
- Vocabulary richness
- Punctuation counts
- Capitalization patterns

**Domain Features**
- Named entities
- Keywords presence
- Sentiment scores
- Topic distributions

### Handling Imbalanced Data

**Sampling Techniques**
- Oversample minority class
- Undersample majority class
- Generate synthetic examples (SMOTE)

**Algorithmic Approaches**
- Class weights
- Focal loss
- Ensemble methods

**Evaluation Considerations**
- Use F1-score, not just accuracy
- Per-class metrics
- Confusion matrix analysis

## Named Entity Recognition (NER)

### Rule-Based NER

**Pattern Matching**
```python
patterns = {
    'EMAIL': r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}',
    'PHONE': r'\d{3}-\d{3}-\d{4}',
    'DATE': r'\d{1,2}/\d{1,2}/\d{4}'
}
```

**Gazetteers**
- Lists of known entities
- Company names, locations, products
- Fast and precise for known entities

**Hybrid Approaches**
- Rules for structured entities (email, phone)
- ML for context-dependent entities (person, org)

### Statistical NER

**Sequence Labeling**
- BIO tagging scheme
- IOB2 format
- BILOU tags

**Features for NER**
- Word shape (capitalization)
- POS tags
- Context windows
- Prefix/suffix
- Gazetteer membership

### NER Evaluation

```python
# Entity-level metrics
precision = correct_entities / predicted_entities
recall = correct_entities / gold_entities
f1 = 2 * precision * recall / (precision + recall)

# Token-level metrics
# May differ from entity-level
```

## Sentiment Analysis Patterns

### Lexicon-Based Approaches

**Word Lists**
- Positive/negative word dictionaries
- Intensity modifiers (very, quite, barely)
- Negation handling (not good = negative)

**Composite Scores**
```python
score = (positive_count - negative_count) / total_words
if score > 0.1: sentiment = "positive"
elif score < -0.1: sentiment = "negative"
else: sentiment = "neutral"
```

### ML-Based Sentiment

**Feature Engineering**
- Word embeddings
- Sentiment lexicon scores
- Emoticons and emoji
- Punctuation (!!!, ???)
- CAPS usage

**Multi-Class vs Binary**
- Binary: Positive/Negative
- Multi-class: Very Negative, Negative, Neutral, Positive, Very Positive
- Consider task requirements

### Aspect-Based Sentiment

Extract sentiment about specific aspects:
```
"The food was great but service was slow"
- Food: Positive
- Service: Negative
```

## Text Summarization

### Extractive Summarization

**Sentence Scoring Methods**
1. **Word Frequency**
   - Score sentences by word importance
   - TF-IDF weights

2. **Position**
   - First/last sentences often important
   - Section headers

3. **Centrality**
   - Similarity to other sentences
   - Graph-based methods (TextRank)

**Selection Strategies**
- Top-K highest scoring sentences
- MMR (Maximal Marginal Relevance) - diverse selection
- Maintain order for coherence

### Abstractive Summarization

For demo purposes, use templates:
```python
summary = f"This document discusses {main_topic}. "
summary += f"Key points include {key_point_1} and {key_point_2}."
```

### Evaluation

- ROUGE scores (for reference summaries)
- Content coverage
- Coherence and fluency
- Length constraints

## Intent Classification

### Intent Categories

Common categories for chatbots:
- Greeting/Farewell
- Question/Query
- Request/Command
- Complaint/Feedback
- Confirmation/Cancellation

### Multi-Intent Handling

```python
# Single query may have multiple intents
"Hello, can you help me book a flight?"
# Intents: [greeting, request_assistance, book_flight]
```

### Confidence Thresholding

```python
if confidence < 0.5:
    return "clarification_needed"
elif confidence < 0.7:
    return intent, "low_confidence"
else:
    return intent, "high_confidence"
```

## Topic Modeling

### Simple Topic Extraction

**Word Co-occurrence**
- Find frequently co-occurring words
- Cluster into topics
- Works for basic cases

**TF-IDF Top Terms**
```python
# Get most important words per document
# Group documents by similarity
# Extract common top terms as topics
```

### Topic Labeling

Strategies for naming topics:
- Top N words (e.g., "health, medical, disease, treatment")
- Most frequent phrases
- Manual labeling based on inspection

### Applications

- Document clustering
- Trend analysis
- Content recommendation
- Search query understanding

## Clustering Patterns

### K-Means Clustering

```python
# Simple document clustering
embeddings = embed_documents(documents)
clusters = kmeans(embeddings, n_clusters=5)

# Assign cluster labels
for doc, cluster in zip(documents, clusters):
    doc['cluster'] = cluster
```

### Hierarchical Clustering

Good for:
- Unknown number of clusters
- Dendrogram visualization
- Multi-level categorization

### Evaluation

- Silhouette score
- Within-cluster variance
- Between-cluster separation
- Manual inspection of clusters

## Knowledge Graph Construction

### Entity Extraction

1. Extract entities from text (NER)
2. Resolve entities (coreference)
3. Identify unique entities

### Relationship Extraction

**Pattern-Based**
```
"X founded Y" → (X, FOUNDED, Y)
"X is CEO of Y" → (X, CEO_OF, Y)
```

**Co-occurrence**
- Entities in same sentence likely related
- Use proximity as relationship strength

### Graph Representation

```python
graph = {
    "nodes": [
        {"id": "entity_1", "type": "Person", "name": "John"},
        {"id": "entity_2", "type": "Company", "name": "Acme"}
    ],
    "edges": [
        {"source": "entity_1", "target": "entity_2", "type": "WORKS_AT"}
    ]
}
```

## Question Generation

### Rule-Based Generation

**Template-Based**
```python
templates = [
    "What is {entity}?",
    "How does {process} work?",
    "When did {event} occur?",
    "Where is {location} located?",
    "Why is {concept} important?"
]
```

**Answer-Focused**
```python
# Given a sentence with an entity
"John founded Microsoft in 1975"

# Generate questions
questions = [
    "Who founded Microsoft?",  # Answer: John
    "When was Microsoft founded?",  # Answer: 1975
    "What did John found in 1975?"  # Answer: Microsoft
]
```

### Question Types

- **Factoid**: Who, What, When, Where
- **Explanation**: How, Why
- **Yes/No**: Is, Does, Can, Will
- **List**: Name all, What are examples of

## Fact Checking Patterns

### Evidence Retrieval

1. Query knowledge base for claims
2. Retrieve relevant documents
3. Extract supporting/contradicting evidence

### Verification Approaches

**Simple Agreement**
```python
if claim in retrieved_facts:
    return "SUPPORTED"
elif contradiction_found:
    return "REFUTED"
else:
    return "NOT ENOUGH INFO"
```

**Confidence Scoring**
```python
support_score = count_supporting_evidence() / total_evidence
if support_score > 0.8: return "TRUE"
elif support_score < 0.2: return "FALSE"
else: return "UNCERTAIN"
```

### Limitations

For demos, acknowledge:
- Limited knowledge base
- Simple matching logic
- No real-time fact checking
- Placeholder verification

## Model Evaluation Best Practices

### Train/Validation/Test Split

```python
# 70/15/15 split common
train, temp = train_test_split(data, test_size=0.3)
val, test = train_test_split(temp, test_size=0.5)
```

### Cross-Validation

```python
# K-fold cross-validation
for fold in range(k):
    train_data = get_train_fold(data, fold)
    val_data = get_val_fold(data, fold)
    
    model.train(train_data)
    evaluate(model, val_data)
```

### Metrics Selection

**Classification**
- Accuracy: Overall correctness
- Precision: Of predicted positives, how many correct
- Recall: Of actual positives, how many found
- F1: Harmonic mean of precision and recall

**Ranking**
- Precision@K
- Recall@K
- MAP (Mean Average Precision)
- NDCG (Normalized Discounted Cumulative Gain)

**Regression**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

## Data Generation for Demos

### Synthetic Text Generation

```python
templates = [
    "{subject} {verb} {object}",
    "The {adjective} {noun} is {verb}",
    "{topic} includes {item1} and {item2}"
]

# Fill with word lists
subjects = ["AI", "ML", "Data Science"]
verbs = ["transforms", "improves", "enables"]
```

### Realistic Data Properties

- Vary length (short and long examples)
- Include edge cases (empty, very long)
- Mix easy and hard examples
- Add noise (typos, informal language)

### Labeling Strategies

- Rule-based automatic labeling
- Programmatic label generation
- Consistency checks

## Common Pitfalls

1. **Overfitting**: Model memorizes training data
   - Solution: Regularization, more data, simpler model

2. **Data Leakage**: Test data influences training
   - Solution: Strict separation, proper validation

3. **Ignoring Class Imbalance**
   - Solution: Proper metrics, sampling, class weights

4. **Poor Feature Engineering**
   - Solution: Domain knowledge, experimentation

5. **Not Handling Unknown Tokens**
   - Solution: OOV handling, subword tokenization

6. **Ignoring Data Quality**
   - Solution: Cleaning, validation, error analysis

## Production Considerations

### Model Serving

- **Batch inference**: Process multiple items together
- **Online inference**: Real-time predictions
- **Caching**: Cache common predictions

### Monitoring

Track:
- Prediction latency
- Model accuracy over time
- Input distribution shifts
- Error rates

### Model Updates

- Retrain periodically
- A/B test new models
- Graceful degradation
- Rollback capability

## Explainability

### Feature Importance

```python
# For linear models
important_features = sorted(
    zip(feature_names, model.coefficients),
    key=lambda x: abs(x[1]),
    reverse=True
)[:10]
```

### Prediction Explanations

- Highlight important words
- Show similar training examples
- Provide confidence scores
- List key features

### User Trust

- Transparent about limitations
- Provide sources when possible
- Allow feedback
- Explain uncertainty
