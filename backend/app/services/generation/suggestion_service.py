"""
Smart suggestion service for follow-up questions.
Context-aware and intent-driven suggestion generation.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import re


class SuggestionService:
    """
    Generates smart follow-up question suggestions based on conversation context.
    Uses intent, entities, and document context to create relevant suggestions.
    """
    
    def generate_suggestions(
        self,
        last_query: str,
        last_response: str,
        context_summary: Dict[str, Any],
        sources: List[Dict[str, Any]],
        processed_query: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate smart follow-up suggestions.
        
        Args:
            last_query: User's last query
            last_response: AI's last response
            context_summary: Current conversation context
            sources: Sources used in last response
            processed_query: Processed query information (optional)
            
        Returns:
            List of suggested follow-up questions (max 4)
        """
        suggestions = []
        
        # Get context information
        primary_doc = context_summary.get('primary_document')
        intent = context_summary.get('last_intent')
        time_period = context_summary.get('recent_time_period')
        entities = context_summary.get('entities', {})
        
        logger.debug(f"Generating suggestions: intent={intent}, doc={primary_doc}, time={time_period}")
        
        # 1. Intent-based suggestions (highest priority)
        intent_suggestions = self._get_intent_based_suggestions(
            intent, last_query, last_response
        )
        suggestions.extend(intent_suggestions)
        
        # 2. Document-specific suggestions
        if primary_doc and sources:
            doc_suggestions = self._get_document_suggestions(primary_doc, sources)
            suggestions.extend(doc_suggestions)
        
        # 3. Entity-based suggestions
        if entities:
            entity_suggestions = self._get_entity_suggestions(entities, time_period)
            suggestions.extend(entity_suggestions)
        
        # 4. Response-based suggestions (analyze AI's answer)
        response_suggestions = self._get_response_based_suggestions(last_response)
        suggestions.extend(response_suggestions)
        
        # 5. General helpful suggestions (fallback)
        if len(suggestions) < 2:
            general_suggestions = self._get_general_suggestions()
            suggestions.extend(general_suggestions)
        
        # Deduplicate and return top 4
        unique_suggestions = self._deduplicate_suggestions(suggestions)
        
        final_suggestions = unique_suggestions[:4]
        logger.info(f"Generated {len(final_suggestions)} suggestions: {final_suggestions}")
        
        return final_suggestions
    
    def _get_intent_based_suggestions(
        self,
        intent: Optional[str],
        query: str,
        response: str
    ) -> List[str]:
        """Generate suggestions based on query intent."""
        suggestions = []
        
        if intent == 'financial':
            # Financial queries often need breakdowns and comparisons
            suggestions.extend([
                "Show me a detailed breakdown",
                "Compare with previous period",
            ])
            
            # If specific numbers mentioned, suggest analysis
            if any(symbol in response for symbol in ['$', '%', 'revenue', 'expense', 'profit']):
                suggestions.append("What drove these numbers?")
            
            # If trends or patterns mentioned
            if any(word in response.lower() for word in ['increase', 'decrease', 'growth', 'decline']):
                suggestions.append("What are the trends over time?")
        
        elif intent == 'procedural':
            # How-to queries need follow-ups
            suggestions.extend([
                "What are the exceptions?",
                "Who do I contact for help?",
            ])
            
            # If steps mentioned
            if any(word in response.lower() for word in ['step', 'first', 'then', 'next']):
                suggestions.append("What documents do I need?")
        
        elif intent == 'compliance':
            # Policy questions need clarification
            suggestions.extend([
                "Are there recent updates?",
                "What happens if I don't comply?",
            ])
        
        elif intent == 'factual':
            # Information queries can go deeper
            suggestions.extend([
                "Tell me more about this",
                "Can you give an example?",
            ])
        
        elif intent == 'analytical':
            # Comparison queries need more analysis
            suggestions.extend([
                "What are the key differences?",
                "Which option is better?",
            ])
        
        return suggestions
    
    def _get_document_suggestions(
        self,
        primary_doc: str,
        sources: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate document-specific suggestions."""
        suggestions = []
        
        # Clean document name for display
        doc_name = primary_doc.replace('.pdf', '').replace('_', ' ')
        
        # Suggest exploring the same document
        suggestions.append(f"What else is in this document?")
        
        # If multiple pages referenced
        pages = set()
        for source in sources:
            if source.get('page'):
                pages.add(source['page'])
        
        if len(pages) > 2:
            suggestions.append("Summarize the key points")
        
        # If multiple documents in sources
        unique_docs = set(s.get('document') for s in sources if s.get('document'))
        if len(unique_docs) > 1:
            suggestions.append("Compare across these documents")
        
        return suggestions
    
    def _get_entity_suggestions(
        self,
        entities: Dict[str, List[str]],
        time_period: Optional[str]
    ) -> List[str]:
        """Generate suggestions based on extracted entities."""
        suggestions = []
        
        # Financial amounts
        if 'amount' in entities and entities['amount']:
            suggestions.append("How does this compare to budget?")
        
        # Percentages
        if 'percentage' in entities and entities['percentage']:
            suggestions.append("What affects this percentage?")
        
        # Time periods
        if time_period:
            # Extract period type (Q1, FY2024, etc.)
            if 'Q' in time_period:
                # Quarterly data
                quarter_match = re.search(r'Q(\d)', time_period)
                if quarter_match:
                    q_num = int(quarter_match.group(1))
                    if q_num > 1:
                        suggestions.append(f"Compare with Q{q_num-1}")
                    if q_num < 4:
                        suggestions.append(f"Show Q{q_num+1} projections")
            
            elif 'FY' in time_period or re.search(r'\b20\d{2}\b', time_period):
                # Yearly data
                suggestions.append("Show year-over-year trends")
        
        return suggestions
    
    def _get_response_based_suggestions(self, response: str) -> List[str]:
        """Analyze AI response to generate relevant follow-ups."""
        suggestions = []
        response_lower = response.lower()
        
        # If response mentions multiple options/items
        if any(word in response_lower for word in ['option', 'options', 'types', 'categories']):
            suggestions.append("Which one should I choose?")
        
        # If response has lists (bullet points)
        bullet_count = response.count('\n-') + response.count('\n•') + response.count('\n*')
        if bullet_count > 2:
            suggestions.append("Explain the most important one")
        
        # If response mentions requirements
        if any(word in response_lower for word in ['must', 'required', 'mandatory', 'need to']):
            suggestions.append("What if I can't meet these?")
        
        # If response mentions deadlines
        if any(word in response_lower for word in ['deadline', 'due date', 'by', 'before']):
            suggestions.append("Can deadlines be extended?")
        
        # If response mentions contact/department
        if any(word in response_lower for word in ['contact', 'department', 'hr', 'finance']):
            suggestions.append("How do I reach them?")
        
        # If response mentions approval/process
        if any(word in response_lower for word in ['approval', 'approve', 'process', 'application']):
            suggestions.append("How long does this take?")
        
        # If response has specific amounts
        if '$' in response or re.search(r'\d+%', response):
            suggestions.append("Show me the calculation")
        
        return suggestions
    
    def _get_general_suggestions(self) -> List[str]:
        """Get general helpful suggestions (fallback)."""
        return [
            "Explain in simpler terms",
            "Give me an example",
            "What should I do next?",
            "Where is the official document?",
        ]
    
    def _deduplicate_suggestions(self, suggestions: List[str]) -> List[str]:
        """Remove duplicates while preserving order and priority."""
        seen = set()
        unique = []
        
        for suggestion in suggestions:
            # Normalize for comparison
            normalized = suggestion.lower().strip().rstrip('?')
            
            if normalized not in seen:
                seen.add(normalized)
                unique.append(suggestion)
        
        return unique
    
    def filter_suggestions_by_confidence(
        self,
        suggestions: List[str],
        confidence: str
    ) -> List[str]:
        """
        Filter suggestions based on response confidence.
        Low confidence → offer simpler follow-ups
        """
        if confidence == 'low':
            # For low confidence, suggest clarification
            return [
                "Can you rephrase that?",
                "Show me related information",
                "Search in a different document"
            ][:2] + suggestions[:2]
        
        return suggestions


# Global instance
suggestion_service = SuggestionService()

__all__ = ['SuggestionService', 'suggestion_service']