from typing import Dict, List
from logger_config import get_logger

class MedicalAgent:
    """Simple agent with tool calling and self-reflection with logging"""
    def __init__(self, retriever_tool, llm, critic, max_iterations=2):
        self.retriever = retriever_tool
        self.llm = llm
        self.critic = critic
        self.max_iterations = max_iterations
        self.conversation_history = []
        self.logger = get_logger('agent')
    
    def run(self, question: str) -> Dict:
        """Execute agent workflow"""
        self.logger.info(f"=" * 60)
        self.logger.info(f"Processing new query: '{question}'")
        self.logger.info(f"=" * 60)
        
        try:
            # Step 1: Retrieve context
            self.logger.info("Step 1: Retrieving relevant context")
            docs = self.retriever.run(question)
            context = "\n\n".join([doc["content"] for doc in docs])
            self.logger.debug(f"Context length: {len(context)} characters")
            
            # Step 2: Generate initial answer
            self.logger.info("Step 2: Generating initial answer")
            answer = self._generate_answer(question, context)
            self.logger.debug(f"Initial answer: {answer[:100]}...")
            
            # Step 3: Self-reflection loop
            self.logger.info("Step 3: Starting self-reflection loop")
            for i in range(self.max_iterations):
                self.logger.info(f"Reflection iteration {i+1}/{self.max_iterations}")
                critique = self.critic.critique(question, context, answer)
                
                if critique["status"] == "approved":
                    self.logger.info("Answer approved by critic")
                    break
                
                self.logger.info(f"Improving answer: {critique['feedback']}")
                answer = self._improve_answer(question, context, answer, critique["feedback"])
            
            # Store in conversation history
            self.conversation_history.append({
                "question": question,
                "answer": answer
            })
            self.logger.info(f"Conversation history size: {len(self.conversation_history)}")
            
            # Keep only last 5 exchanges
            if len(self.conversation_history) > 5:
                self.conversation_history.pop(0)
                self.logger.debug("Trimmed conversation history to last 5 exchanges")
            
            self.logger.info("Query processing completed successfully")
            
            return {
                "question": question,
                "answer": answer,
                "context": context,
                "sources": len(docs)
            }
        
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise
    
    def _get_history_context(self) -> str:
        """Format conversation history"""
        if not self.conversation_history:
            return ""
        
        history_text = "\n\nPrevious Conversation:\n"
        for exchange in self.conversation_history[-3:]:
            history_text += f"Q: {exchange['question']}\nA: {exchange['answer']}\n\n"
        return history_text
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer from context"""
        history = self._get_history_context()
        
        prompt = f"""
        You are a medical assistant. Answer the question based on the context.
        {history}
        Context:
        {context}
        
        Question: {question}
        
        Provide a clear, concise answer (3-4 sentences max). If the question refers to previous conversation, use that context.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"Answer generation failed: {str(e)}", exc_info=True)
            raise
    
    def _improve_answer(self, question: str, context: str, previous_answer: str, feedback: str) -> str:
        """Improve answer based on feedback"""
        history = self._get_history_context()
        
        prompt = f"""
        {history}
        Context:
        {context}
        
        Question: {question}
        
        Previous Answer: {previous_answer}
        
        Feedback: {feedback}
        
        Provide an improved answer addressing the feedback.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"Answer improvement failed: {str(e)}", exc_info=True)
            return previous_answer  # Fallback to previous answer
    
    def clear_history(self):
        """Clear conversation history"""
        self.logger.info("Clearing conversation history")
        self.conversation_history = []
