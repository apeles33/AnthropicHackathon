"""
ML Utilities for Classification, NER, and Other ML Tasks
Simple, self-contained implementations using sklearn
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict


class TextClassifier:
    """
    Simple Naive Bayes text classifier
    """
    
    def __init__(self):
        self.class_word_counts = defaultdict(Counter)
        self.class_doc_counts = defaultdict(int)
        self.vocabulary = set()
        self.classes = set()
        self.trained = False
    
    def train(self, texts: List[str], labels: List[str]):
        """Train the classifier"""
        for text, label in zip(texts, labels):
            self.classes.add(label)
            self.class_doc_counts[label] += 1
            
            words = text.lower().split()
            for word in words:
                self.vocabulary.add(word)
                self.class_word_counts[label][word] += 1
        
        self.trained = True
    
    def predict(self, text: str) -> str:
        """Predict class for text"""
        if not self.trained:
            raise ValueError("Classifier must be trained first")
        
        words = text.lower().split()
        scores = {}
        
        total_docs = sum(self.class_doc_counts.values())
        
        for cls in self.classes:
            # Prior probability
            prior = self.class_doc_counts[cls] / total_docs
            
            # Likelihood
            likelihood = 1.0
            total_words = sum(self.class_word_counts[cls].values())
            
            for word in words:
                if word in self.vocabulary:
                    word_count = self.class_word_counts[cls].get(word, 0)
                    # Laplace smoothing
                    prob = (word_count + 1) / (total_words + len(self.vocabulary))
                    likelihood *= prob
            
            scores[cls] = prior * likelihood
        
        return max(scores, key=scores.get)
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """Predict probabilities for all classes"""
        if not self.trained:
            raise ValueError("Classifier must be trained first")
        
        words = text.lower().split()
        scores = {}
        
        total_docs = sum(self.class_doc_counts.values())
        
        for cls in self.classes:
            prior = self.class_doc_counts[cls] / total_docs
            likelihood = 1.0
            total_words = sum(self.class_word_counts[cls].values())
            
            for word in words:
                if word in self.vocabulary:
                    word_count = self.class_word_counts[cls].get(word, 0)
                    prob = (word_count + 1) / (total_words + len(self.vocabulary))
                    likelihood *= prob
            
            scores[cls] = prior * likelihood
        
        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores


class SentimentAnalyzer:
    """
    Simple rule-based sentiment analyzer
    """
    
    def __init__(self):
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'best', 'perfect', 'awesome', 'happy', 'joy', 'brilliant',
            'outstanding', 'superb', 'delightful', 'pleased', 'satisfied'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'hate',
            'disappointing', 'disappointing', 'sad', 'angry', 'frustrated',
            'annoyed', 'upset', 'disgusting', 'useless', 'pathetic'
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        words = text.lower().split()
        
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        
        score = (pos_count - neg_count) / max(len(words), 1)
        
        if score > 0.1:
            sentiment = "positive"
        elif score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": score,
            "positive_words": pos_count,
            "negative_words": neg_count,
            "confidence": abs(score)
        }


class SimpleNER:
    """
    Simple rule-based Named Entity Recognition
    """
    
    def __init__(self):
        self.patterns = {
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'DATE': r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b',
            'MONEY': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            'URL': r'https?://[^\s]+',
            'ORG': r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b'
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'type': entity_type,
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Sort by position
        entities.sort(key=lambda x: x['start'])
        
        return entities


class IntentClassifier:
    """
    Simple intent classifier for chatbots
    """
    
    def __init__(self):
        self.intent_patterns = {}
        self.trained = False
    
    def train(self, intents: Dict[str, List[str]]):
        """
        Train with intent examples
        intents: dict mapping intent names to example phrases
        """
        self.intent_patterns = {}
        
        for intent, examples in intents.items():
            # Extract key words from examples
            all_words = []
            for example in examples:
                words = example.lower().split()
                all_words.extend(words)
            
            # Count word frequencies
            word_freq = Counter(all_words)
            # Keep top words as patterns
            top_words = [w for w, _ in word_freq.most_common(10)]
            self.intent_patterns[intent] = set(top_words)
        
        self.trained = True
    
    def classify(self, text: str) -> Tuple[str, float]:
        """Classify intent of text"""
        if not self.trained:
            raise ValueError("Classifier must be trained first")
        
        words = set(text.lower().split())
        scores = {}
        
        for intent, keywords in self.intent_patterns.items():
            # Calculate overlap
            overlap = len(words & keywords)
            scores[intent] = overlap
        
        if not scores:
            return "unknown", 0.0
        
        best_intent = max(scores, key=scores.get)
        max_score = scores[best_intent]
        confidence = max_score / len(words) if words else 0.0
        
        return best_intent, confidence


class SimpleSummarizer:
    """
    Simple extractive text summarizer
    """
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Create extractive summary"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Score sentences by word frequency
        words = text.lower().split()
        word_freq = Counter(words)
        
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            score = 0
            words_in_sentence = sentence.lower().split()
            for word in words_in_sentence:
                score += word_freq.get(word, 0)
            
            sentence_scores[i] = score / len(words_in_sentence) if words_in_sentence else 0
        
        # Get top sentences
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        top_indices.sort()  # Maintain order
        
        summary_sentences = [sentences[i] for i in top_indices]
        return '. '.join(summary_sentences) + '.'


class TopicModeler:
    """
    Simple topic modeling using word co-occurrence
    """
    
    def __init__(self, num_topics: int = 5, top_words: int = 10):
        self.num_topics = num_topics
        self.top_words = top_words
        self.topics = []
    
    def fit(self, documents: List[str]):
        """Extract topics from documents"""
        # Build word co-occurrence matrix
        all_words = []
        doc_words = []
        
        for doc in documents:
            words = doc.lower().split()
            doc_words.append(words)
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        # Get top words
        vocab = [w for w, _ in word_freq.most_common(100)]
        
        # Simple clustering by co-occurrence
        topics = []
        used_words = set()
        
        for _ in range(self.num_topics):
            # Find unused high-frequency word
            seed_word = None
            for word in vocab:
                if word not in used_words:
                    seed_word = word
                    break
            
            if not seed_word:
                break
            
            # Find related words
            related = Counter()
            for words in doc_words:
                if seed_word in words:
                    for word in words:
                        if word != seed_word and word in vocab:
                            related[word] += 1
            
            # Create topic
            topic_words = [seed_word] + [w for w, _ in related.most_common(self.top_words - 1)]
            topics.append(topic_words)
            used_words.update(topic_words)
        
        self.topics = topics
        return topics
    
    def get_topics(self) -> List[List[str]]:
        """Get discovered topics"""
        return self.topics


def generate_training_data(task: str, num_samples: int = 100) -> Tuple[List[str], List[str]]:
    """
    Generate dummy training data for various tasks
    """
    if task == "sentiment":
        positive_templates = [
            "This is amazing and wonderful",
            "I love this product so much",
            "Excellent quality and great service",
            "Best purchase I've ever made",
            "Highly recommend this to everyone"
        ]
        
        negative_templates = [
            "This is terrible and disappointing",
            "Worst experience I've ever had",
            "Poor quality and bad service",
            "Complete waste of money",
            "Would not recommend to anyone"
        ]
        
        texts = []
        labels = []
        
        for i in range(num_samples):
            if i % 2 == 0:
                texts.append(positive_templates[i % len(positive_templates)])
                labels.append("positive")
            else:
                texts.append(negative_templates[i % len(negative_templates)])
                labels.append("negative")
        
        return texts, labels
    
    elif task == "intent":
        intent_templates = {
            "greeting": ["hello", "hi there", "good morning", "hey"],
            "goodbye": ["bye", "goodbye", "see you later", "farewell"],
            "question": ["what is", "how do", "can you tell me", "why does"],
            "complaint": ["I'm unhappy", "this doesn't work", "I have a problem", "issue with"]
        }
        
        texts = []
        labels = []
        
        for intent, templates in intent_templates.items():
            for _ in range(num_samples // len(intent_templates)):
                texts.append(templates[np.random.randint(len(templates))])
                labels.append(intent)
        
        return texts, labels
    
    else:  # Generic classification
        texts = [f"Sample text {i} with content about topic {i % 3}" for i in range(num_samples)]
        labels = [f"category_{i % 3}" for i in range(num_samples)]
        return texts, labels


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Calculate classification metrics"""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0
    
    # Per-class metrics
    classes = set(y_true)
    class_metrics = {}
    
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return {
        "accuracy": accuracy,
        "per_class": class_metrics
    }
