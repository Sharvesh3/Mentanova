"""
LLM generation service using Google Gemini.
Handles answer generation with strict sourcing and formatting.
Supports both document-based and conversational responses.
"""
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from app.core.config import settings


class LLMService:
    """
    Service for generating answers using Google Gemini.
    Implements strict prompting for accurate, source-based responses.
    Also handles natural conversational queries.
    """
    
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)
        
        self.model_name = settings.gemini_chat_model
        self.temperature = settings.temperature
        self.max_tokens = settings.gemini_max_tokens
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": settings.max_response_tokens,
                "top_p": 0.95,
            }
        )
        
        # System prompt for document-based RAG
        self.system_instruction = """You are Mentanova, a helpful AI assistant specializing in Finance and HRMS documentation.

Your role is to provide accurate, helpful answers based on the provided document context.

Guidelines:
1. Answer questions using ONLY the information from the provided context when documents are available
2. Be conversational and friendly, but professional
3. For financial figures: Always cite the source document and page
4. If information is not in the context: Politely say you don't have that information in the available documents
5. For policies: Reference the exact document and section
6. Use format [Document: X, Page: Y] for citations
7. If unsure, acknowledge it and suggest alternatives

Response style:
- Use natural, conversational language
- Structure responses with:
  - Clear paragraphs
  - Bullet points for lists
  - Bold for emphasis (use **text**)
  - Proper spacing
- Be concise but thorough
- For greetings and general conversation, respond naturally without forcing document references"""
        
        logger.info(f"LLM service initialized: {self.model_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_conversational_response(
        self,
        query: str,
        context_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        is_error: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a natural conversational response (no document context).
        Used for greetings, small talk, and general conversation.
        
        Args:
            query: User's query
            context_message: Context or guidance message
            history: Conversation history
            is_error: Whether this is an error response
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Generating conversational response for: '{query[:50]}...'")
        
        # Build conversational prompt
        if is_error:
            prompt = f"""The user asked: "{query}"

However, there's an issue: {context_message}

Please respond in a helpful, friendly way that explains the limitation while guiding the user on how they can get help with their Finance and HRMS documents."""
        else:
            prompt = f"""The user said: "{query}"

Please respond naturally and helpfully. You are Mentanova, an AI assistant that helps with Finance and HRMS documents. 

Be conversational, friendly, and guide the user on how you can help them find information from documents.

Use natural language, be warm and helpful."""
        
        # Build message history
        messages = []
        
        # Add conversational system prompt
        messages.append({
            "role": "user",
            "parts": ["You are Mentanova, a friendly AI assistant. Respond naturally to user queries. Be helpful, conversational, and professional. Use markdown formatting (bullet points, bold, etc.) when appropriate."]
        })
        messages.append({
            "role": "model",
            "parts": ["I understand. I'll be helpful, conversational, and use proper formatting."]
        })
        
        # Add history if exists
        if history:
            for msg in history[-4:]:
                role = "model" if msg["role"] == "assistant" else "user"
                messages.append({"role": role, "parts": [msg["content"]]})
        
        # Add current prompt
        messages.append({"role": "user", "parts": [prompt]})
        
        try:
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    messages,
                    generation_config={
                        "temperature": 0.7,  # More creative for conversation
                        "max_output_tokens": 500,
                    }
                )
            )
            
            answer = response.text
            
            return {
                'answer': answer,
                'confidence': 'high',
                'citations': [],
                'usage': {
                    'prompt_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    'completion_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    'total_tokens': response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Conversational generation failed: {str(e)}")
            # Fallback to context message
            return {
                'answer': context_message,
                'confidence': 'medium',
                'citations': [],
                'usage': {'total_tokens': 0}
            }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_answer(
        self,
        query: str,
        context: str,
        sources: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        reformulated_query: Optional[str] = None,
        conversation_context: Optional[Dict[str, Any]] = None,
        is_conversational: bool = False
    ) -> Dict[str, Any]:
        """
        Generate answer using Gemini with context awareness.
        Enhanced to handle conversational flow and document scoping.
        
        Args:
            query: User's original question
            context: Retrieved context (can be empty for conversational)
            sources: Source documents
            conversation_history: Previous messages
            reformulated_query: Query enhanced with context (optional)
            conversation_context: Summary of conversation state
            is_conversational: If True, focus on natural conversation
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        logger.info(f"Generating answer: conversational={is_conversational}, sources={len(sources)}")
        
        # Build context-aware prompt
        if is_conversational or not context or len(context.strip()) < 50:
            # Conversational mode - no strong document context
            user_prompt = self._build_conversational_prompt(
                query, conversation_history, conversation_context
            )
        else:
            # Document-based mode with context awareness
            user_prompt = self._build_contextual_prompt(
                query, context, sources, reformulated_query, conversation_context
            )
        
        # Build messages
        messages = []
        
        # Enhanced system instruction
        system_instruction = self._get_context_aware_system_instruction(conversation_context)
        messages.append({"role": "user", "parts": [system_instruction]})
        messages.append({"role": "model", "parts": ["Understood. I'll provide accurate, context-aware responses with proper citations."]})
        
        # Add conversation history (last 6 messages for better context)
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = "model" if msg["role"] == "assistant" else "user"
                messages.append({"role": role, "parts": [msg["content"]]})
        
        # Add current query
        messages.append({"role": "user", "parts": [user_prompt]})
        
        try:
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    messages,
                    generation_config={
                        "temperature": 0.7 if is_conversational else self.temperature,
                        "max_output_tokens": settings.max_response_tokens,
                    }
                )
            )
            
            answer = response.text
            citations = self._extract_citations(answer, sources)
            confidence = self._assess_confidence(answer, context, conversation_context)
            
            usage = {
                'prompt_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                'completion_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                'total_tokens': response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
            }
            
            logger.info(f"✅ Generated: {len(answer)} chars, {len(citations)} citations")
            
            return {
                'answer': answer,
                'citations': citations,
                'confidence': confidence,
                'finish_reason': 'stop',
                'usage': usage
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def _get_context_aware_system_instruction(
        self,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get system instruction based on conversation context.
        
        Args:
            conversation_context: Current conversation state
            
        Returns:
            Context-aware system instruction
        """
        base_instruction = """You are Mentanova, a helpful AI assistant specializing in Finance and HRMS documentation.

Your role is to provide accurate, conversational answers based on the provided context and conversation history.

Core Guidelines:
1. **Context Awareness**: Pay attention to the conversation history and maintain context across messages
2. **Document Scoping**: When discussing a specific document, stay focused on that document unless explicitly asked to look elsewhere
3. **Natural Conversation**: Respond naturally like ChatGPT - be conversational while staying accurate
4. **Source Attribution**: Always cite sources for factual information using [Document: X, Page: Y] format
5. **Clarification**: If information is not in the context, say so clearly and offer alternatives
6. **Follow-ups**: Handle follow-up questions by referring back to previous context

Response Guidelines:
- Use natural, conversational language
- Structure responses clearly with paragraphs, bullet points, and bold text
- For financial data: Always include exact figures and sources
- For follow-up questions: Reference previous answers when relevant
- Be concise but thorough
"""
        
        # Add context-specific instructions
        if conversation_context:
            primary_doc = conversation_context.get('primary_document')
            time_period = conversation_context.get('recent_time_period')
            
            if primary_doc:
                base_instruction += f"""
**Current Context**:
- Primary Document: {primary_doc}
- When answering follow-up questions, prioritize information from this document unless asked otherwise
"""
            
            if time_period:
                base_instruction += f"- Time Period in Focus: {time_period}\n"
        
        return base_instruction
    
    def _extract_citations(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract citation references from the answer text.
        
        Args:
            answer: Generated answer text
            sources: List of source documents
            
        Returns:
            List of citation dictionaries with document and page info
        """
        import re
        
        citations = []
        
        # Pattern to match [Document: X, Page: Y] format
        citation_pattern = r'\[Document:\s*([^,]+),\s*Page:\s*(\d+|N/A)\]'
        
        matches = re.findall(citation_pattern, answer, re.IGNORECASE)
        
        for doc_name, page in matches:
            doc_name = doc_name.strip()
            
            # Find matching source
            matching_source = None
            for source in sources:
                source_doc = source.get('document', '')
                if doc_name.lower() in source_doc.lower() or source_doc.lower() in doc_name.lower():
                    matching_source = source
                    break
            
            citation = {
                'document': doc_name,
                'page': int(page) if page.isdigit() else None,
                'text_reference': f"[Document: {doc_name}, Page: {page}]"
            }
            
            if matching_source:
                citation['chunk_id'] = matching_source.get('chunk_id')
                citation['section'] = matching_source.get('section')
            
            citations.append(citation)
        
        # Deduplicate citations
        seen = set()
        unique_citations = []
        for citation in citations:
            key = (citation['document'], citation['page'])
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        logger.debug(f"Extracted {len(unique_citations)} unique citations from answer")
    
        return unique_citations

    def _build_contextual_prompt(
        self,
        query: str,
        context: str,
        sources: List[Dict[str, Any]],
        reformulated_query: Optional[str] = None,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build context-aware prompt for document-based queries.
        
        Args:
            query: Original user query
            context: Retrieved document context
            sources: Source documents
            reformulated_query: Enhanced query with context
            conversation_context: Conversation state
            
        Returns:
            Formatted prompt
        """
        # Format sources
        sources_text = "\n".join([
            f"- {src['document']}, Page {src.get('page', 'N/A')}" + 
            (f", Section: {src['section']}" if src.get('section') else "")
            for src in sources
        ])
        
        prompt_parts = []
        
        # Add conversation context if available
        if conversation_context:
            primary_doc = conversation_context.get('primary_document')
            if primary_doc:
                prompt_parts.append(f"**Context**: We are currently discussing '{primary_doc}'")
        
        # Show both original and reformulated query if different
        if reformulated_query and reformulated_query != query:
            prompt_parts.append(f"**User's Question**: {query}")
            prompt_parts.append(f"**Interpreted as**: {reformulated_query}")
        else:
            prompt_parts.append(f"**Question**: {query}")
        
        prompt_parts.append(f"""
**Available Context from Documents**:
{context}

**Source Documents**:
{sources_text}

**Instructions**:
1. Answer based ONLY on the context above
2. If this is a follow-up question, connect it to previous context
3. Cite sources using format: [Document: X, Page: Y]
4. If context is insufficient, clearly state what's missing
5. For financial data, include exact figures and sources
6. Be conversational and natural in your response

**Formatting**:
- Use **bold** for important points
- Use bullet points for lists
- Use clear paragraphs
- Use markdown formatting

**Answer**:""")
        
        return "\n\n".join(prompt_parts)
    
    def _build_conversational_prompt(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for conversational queries (no document context).
        
        Args:
            query: User query
            conversation_history: Recent messages
            conversation_context: Conversation state
            
        Returns:
            Conversational prompt
        """
        prompt_parts = [f"**User**: {query}"]
        
        # Add context hints
        if conversation_context:
            message_count = conversation_context.get('message_count', 0)
            if message_count > 0:
                prompt_parts.append("\n**Context**: This is part of an ongoing conversation.")
        
        prompt_parts.append("""
**Instructions**:
- Respond naturally and conversationally
- If this is a greeting or small talk, respond warmly
- If the user is asking about documents but no context is available, guide them helpfully
- Use proper markdown formatting
- Be friendly and professional

**Response**:""")
        
        return "\n".join(prompt_parts)
    
    def _assess_confidence(
        self,
        answer: str,
        context: str,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Assess confidence level with context awareness.
        
        Args:
            answer: Generated answer
            context: Document context used
            conversation_context: Conversation state
            
        Returns:
            Confidence level: 'high', 'medium', 'low'
        """
        # High confidence indicators
        high_indicators = [
            'according to', 'states that', 'specified in',
            'the document shows', 'as per', 'clearly mentions'
        ]
        
        # Low confidence indicators
        low_indicators = [
            'not available', 'unclear', 'doesn\'t specify',
            'may', 'might', 'possibly', 'appears to', 'i don\'t have'
        ]
        
        answer_lower = answer.lower()
        
        # Check for uncertainty
        if any(indicator in answer_lower for indicator in low_indicators):
            return 'low'
        
        # Check for strong sourcing
        has_citations = '[document:' in answer_lower
        has_strong_language = any(indicator in answer_lower for indicator in high_indicators)
        
        if has_citations and has_strong_language:
            return 'high'
        
        # Context-based confidence
        if conversation_context:
            # If we're in a scoped document conversation with sources
            if conversation_context.get('primary_document') and has_citations:
                return 'high'
        
        return 'medium'
    
    async def summarize_document(
        self,
        document_content: str,
        document_title: str,
        max_length: int = 500
    ) -> str:
        """
        Generate a concise summary of a document.
        
        Args:
            document_content: Full or partial document content
            document_title: Title of the document
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        prompt = f"""Summarize the following document concisely in {max_length} words or less.
Focus on key points, main topics, and important information.

Use proper markdown formatting:
- **Bold** for key terms
- Bullet points for main topics
- Clear paragraphs

Document: {document_title}

Content:
{document_content[:4000]}

Summary:"""
        
        try:
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": max_length * 2,
                    }
                )
            )
            
            summary = response.text
            logger.info(f"Generated summary for '{document_title}'")
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return f"Summary unavailable for {document_title}"


# Global instance
llm_service = LLMService()

__all__ = ['LLMService', 'llm_service']