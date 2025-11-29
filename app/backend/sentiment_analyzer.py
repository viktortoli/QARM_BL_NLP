"""FinBERT sentiment analyzer"""
from typing import Dict, List, Tuple
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
FINBERT_AVAILABLE = True


class FinBERTSentimentAnalyzer:
    """FinBERT sentiment analyzer"""

    def __init__(self):
        """Initialize FinBERT model"""
        model_name = "ProsusAI/finbert"

        print("[INFO] Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ['negative', 'neutral', 'positive']
        self.model.eval()
        print("[INFO] FinBERT model loaded successfully")

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        if not text or not text.strip():
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'probabilities': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
            }

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

        predicted_idx = torch.argmax(probabilities).item()
        predicted_label = self.labels[predicted_idx]
        confidence = probabilities[predicted_idx].item()

        probs_dict = {
            label: probabilities[i].item()
            for i, label in enumerate(self.labels)
        }

        sentiment_score = (
            probs_dict['positive'] * 1.0 +
            probs_dict['neutral'] * 0.0 +
            probs_dict['negative'] * (-1.0)
        )

        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': predicted_label,
            'confidence': confidence,
            'probabilities': probs_dict
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment of multiple texts"""
        return [self.analyze_text(text) for text in texts]

    def analyze_ticker_news(
        self,
        news_articles: List[Dict],
        use_weighted: bool = True,
        lambda_decay: float = 0.25
    ) -> Dict:
        """Analyze sentiment for news articles"""
        if not news_articles:
            return {
                'avg_sentiment_score': 0.0,
                'simple_avg_sentiment_score': 0.0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'articles': []
            }

        article_analyses = []
        total_score = 0.0
        label_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

        now = datetime.now()

        for article in news_articles:
            title = article.get('title', '')
            content = article.get('content', '')

            text_to_analyze = title
            if content and len(content) > 10:
                text_to_analyze += " " + content[:200]

            analysis = self.analyze_text(text_to_analyze)

            analysis['article_title'] = title
            analysis['article_url'] = article.get('url', '')
            analysis['published_date'] = article.get('published_date', '')

            article_analyses.append(analysis)

            total_score += analysis['sentiment_score']
            label_counts[analysis['sentiment_label']] += 1

        article_count = len(article_analyses)
        simple_avg_score = total_score / article_count

        if use_weighted:
            weighted_avg_score = self._calculate_weighted_sentiment(
                article_analyses,
                now,
                lambda_decay
            )
        else:
            weighted_avg_score = simple_avg_score

        distribution = {
            label: (count / article_count * 100)
            for label, count in label_counts.items()
        }

        return {
            'avg_sentiment_score': weighted_avg_score,
            'simple_avg_sentiment_score': simple_avg_score,
            'article_count': article_count,
            'positive_count': label_counts['positive'],
            'negative_count': label_counts['negative'],
            'neutral_count': label_counts['neutral'],
            'distribution': distribution,
            'articles': article_analyses
        }

    def _calculate_weighted_sentiment(
        self,
        article_analyses: List[Dict],
        current_time: datetime,
        lambda_decay: float
    ) -> float:
        """Calculate weighted sentiment with temporal decay"""
        weighted_sum = 0.0
        weight_sum = 0.0

        for analysis in article_analyses:
            sentiment_score = analysis['sentiment_score']
            confidence = analysis['confidence']

            try:
                pub_date_str = analysis['published_date']
                if 'T' in pub_date_str:
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                else:
                    pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d')

                time_diff = (current_time - pub_date).total_seconds() / 86400
                temporal_decay = math.exp(-lambda_decay * time_diff)
                weight = confidence * temporal_decay

                weighted_sum += weight * sentiment_score
                weight_sum += weight

            except Exception as e:
                weight = confidence
                weighted_sum += weight * sentiment_score
                weight_sum += weight

        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return sum(a['sentiment_score'] for a in article_analyses) / len(article_analyses)

    def get_sentiment_summary(self, score: float) -> Tuple[str, str]:
        """Get human-readable sentiment summary"""
        if score >= 0.3:
            return "🟢", "Positive"
        elif score >= 0.1:
            return "🟢", "Slightly Positive"
        elif score >= -0.1:
            return "🟡", "Neutral"
        elif score >= -0.3:
            return "🔴", "Slightly Negative"
        else:
            return "🔴", "Negative"
