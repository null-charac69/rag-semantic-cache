"""
LLM Service
===========
Handles interaction with Groq API for text generation.

Groq advantages:
- Free tier: 14,400 requests/day
- Ultra-fast inference (LPU technology)
- Supports Llama 3.1, Mixtral, Gemma models
- Simple API (OpenAI-compatible)

This service:
1. Takes context (retrieved chunks) + query
2. Constructs prompt
3. Calls Groq API
4. Returns generated answer
"""

from groq import Groq
from typing import List, Optional, Dict, Any
from loguru import logger

from app.core.config import settings
from app.models.schemas import SourceChunk


class LLMService:
    """
    Language Model service using Groq API.
    
    Handles prompt construction and text generation.
    """
    
    def __init__(self):
        """Initialize Groq client."""
        logger.info("Initializing LLM service...")
        
        try:
            # Initialize Groq client
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            
            # Model configuration
            self.model = settings.LLM_MODEL
            self.max_tokens = 2048  # Maximum response length
            self.temperature = 0.7  # Creativity (0 = deterministic, 1 = creative)
            
            logger.success(f"✅ LLM service initialized (model: {self.model})")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM service: {e}")
            raise
    
    
    def construct_prompt(
        self,
        query: str,
        context_chunks: List[SourceChunk]
    ) -> str:
        """
        Construct RAG prompt with context.
        
        Prompt structure:
        1. System instructions (how to behave)
        2. Context (retrieved chunks)
        3. User question
        
        This is CRITICAL for good RAG responses!
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Formatted prompt string
        """
        # Build context from chunks
        context_parts = []
        
        for idx, chunk in enumerate(context_chunks, 1):
            # Include source metadata for citations
            source_info = ""
            if chunk.metadata:
                filename = chunk.metadata.get('filename', 'Unknown')
                doc_type = chunk.metadata.get('document_type', 'document')
                source_info = f"[Source: {filename} ({doc_type})]"
            
            context_parts.append(
                f"Context {idx} {source_info}:\n{chunk.content}\n"
            )
        
        context_text = "\n".join(context_parts)
        
        # Construct full prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.

INSTRUCTIONS:
- Answer the question using ONLY the information from the context below
- If the context doesn't contain enough information, say "I don't have enough information to answer that question based on the provided context"
- Be concise but comprehensive
- If you reference specific information, mention which context it came from
- Do not make up information not present in the context

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[SourceChunk],
        stream: bool = False
    ) -> str:
        """
        Generate answer using LLM.
        
        Process:
        1. Construct prompt with context
        2. Call Groq API
        3. Return generated text
        
        Args:
            query: User's question
            context_chunks: Retrieved context
            stream: Whether to stream response (future feature)
            
        Returns:
            Generated answer
        """
        logger.info(f"Generating answer for: '{query[:50]}...'")
        
        try:
            # Construct prompt
            prompt = self.construct_prompt(query, context_chunks)
            
            # Log prompt length (for debugging token limits)
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                stream=False  # Non-streaming for now
            )
            
            # Extract generated text
            answer = response.choices[0].message.content
            
            # Log token usage (useful for monitoring costs/limits)
            usage = response.usage
            logger.info(
                f"✅ Answer generated "
                f"(tokens: {usage.total_tokens} = "
                f"{usage.prompt_tokens} prompt + "
                f"{usage.completion_tokens} completion)"
            )
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            
            # Return user-friendly error message
            if "rate_limit" in str(e).lower():
                return "I'm currently experiencing high demand. Please try again in a moment."
            elif "context_length" in str(e).lower():
                return "The context is too long for me to process. Please try a more specific question."
            else:
                return f"I encountered an error while generating an answer: {str(e)}"
    
    
    def generate_simple_answer(
        self,
        query: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate answer without RAG context.
        
        Used for:
        - General questions that don't need document context
        - Fallback when no relevant chunks found
        
        Args:
            query: User's question
            temperature: Override default temperature
            
        Returns:
            Generated answer
        """
        temp = temperature if temperature is not None else self.temperature
        
        try:
            logger.info("Generating simple answer (no RAG context)")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Provide accurate and concise answers."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=temp,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            logger.success("✅ Simple answer generated")
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating simple answer: {e}")
            return f"I encountered an error: {str(e)}"
    
    
    def generate_summary(
        self,
        text: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Generate summary of text.
        
        Useful for:
        - Summarizing long documents
        - Creating document previews
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            
        Returns:
            Summary
        """
        max_length = max_length or 200
        
        try:
            logger.info(f"Generating summary (max {max_length} words)")
            
            prompt = f"""Summarize the following text in no more than {max_length} words. 
Be concise and capture the main points.

TEXT:
{text}

SUMMARY:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for factual summaries
                max_tokens=max_length * 2  # Rough token estimation
            )
            
            summary = response.choices[0].message.content
            logger.success("✅ Summary generated")
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error creating summary: {str(e)}"
    
    
    def check_relevance(
        self,
        query: str,
        chunks: List[SourceChunk],
        threshold: float = 0.7
    ) -> List[SourceChunk]:
        """
        Filter chunks by relevance using LLM.
        
        Sometimes vector search returns chunks with high similarity
        but low actual relevance. LLM can judge better!
        
        Args:
            query: User's question
            chunks: Retrieved chunks
            threshold: Relevance threshold (0-1)
            
        Returns:
            Filtered chunks (only relevant ones)
        """
        try:
            logger.info(f"Checking relevance of {len(chunks)} chunks")
            
            relevant_chunks = []
            
            for chunk in chunks:
                # Ask LLM: "Is this chunk relevant?"
                prompt = f"""On a scale of 0.0 to 1.0, how relevant is this text to answering the question?
Only respond with a number.

QUESTION: {query}

TEXT: {chunk.content[:500]}...

RELEVANCE SCORE:"""
                
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,  # Deterministic
                        max_tokens=10
                    )
                    
                    # Parse score
                    score_text = response.choices[0].message.content.strip()
                    score = float(score_text)
                    
                    if score >= threshold:
                        relevant_chunks.append(chunk)
                        logger.debug(f"Chunk relevant (score: {score:.2f})")
                    else:
                        logger.debug(f"Chunk filtered out (score: {score:.2f})")
                        
                except ValueError:
                    # Couldn't parse score, include chunk to be safe
                    relevant_chunks.append(chunk)
            
            logger.info(f"✅ Filtered to {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return chunks  # Return all chunks if error
    
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict with model details
        """
        return {
            "model": self.model,
            "provider": "Groq",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_llm_service_instance = None

def get_llm_service() -> LLMService:
    """
    Get or create singleton LLM service.
    
    Usage:
        from app.services.llm import get_llm_service
        
        llm = get_llm_service()
        answer = llm.generate_answer(query, context_chunks)
    """
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance


# ============================================================
# PROMPT TEMPLATES (Advanced)
# ============================================================

class PromptTemplates:
    """
    Collection of prompt templates for different scenarios.
    
    Different questions need different prompts!
    """
    
    @staticmethod
    def rag_qa_prompt(query: str, context: str) -> str:
        """Standard RAG Q&A prompt."""
        return f"""Answer the question based on the context below. Be concise and accurate.

Context: {context}

Question: {query}

Answer:"""
    
    @staticmethod
    def code_explanation_prompt(code: str, query: str) -> str:
        """Prompt for explaining code."""
        return f"""Explain the following code in simple terms.

Code:
{code}

Question: {query}

Explanation:"""
    
    @staticmethod
    def comparison_prompt(items: List[str], query: str) -> str:
        """Prompt for comparing multiple items."""
        items_text = "\n\n".join([f"Item {i+1}:\n{item}" for i, item in enumerate(items)])
        
        return f"""Compare the following items based on the question.

{items_text}

Question: {query}

Comparison:"""
    
    @staticmethod
    def chain_of_thought_prompt(query: str, context: str) -> str:
        """Prompt that encourages step-by-step reasoning."""
        return f"""Answer the question by thinking step-by-step. Show your reasoning.

Context: {context}

Question: {query}

Let's think step by step:
1. <your first step>
2. <your second step>
...
Answer:"""