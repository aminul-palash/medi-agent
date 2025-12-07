from typing import Dict
from logger_config import get_logger

class SelfReflectionCritic:
    """Simple critic for self-reflection with logging"""
    def __init__(self, llm):
        self.llm = llm
        self.logger = get_logger('critic')
    
    def critique(self, question: str, context: str, answer: str) -> Dict:
        """Evaluate if answer needs improvement"""
        self.logger.info(f"Evaluating answer for: '{question[:50]}...'")
        
        critique_prompt = f"""
        Question: {question}
        Context Retrieved: {context[:500]}...
        Generated Answer: {answer}
        
        Evaluate this answer:
        1. Is it accurate based on the context?
        2. Is it complete?
        3. Does it need improvement?
        
        Respond with:
        - "GOOD" if answer is satisfactory
        - "IMPROVE: <reason>" if it needs work
        """
        
        try:
            response = self.llm.invoke(critique_prompt)
            content = response.content.strip()
            
            if content.startswith("GOOD"):
                self.logger.info("Critique result: APPROVED")
                return {"status": "approved", "feedback": "Answer is satisfactory"}
            else:
                self.logger.info(f"Critique result: NEEDS IMPROVEMENT - {content}")
                return {"status": "needs_improvement", "feedback": content}
        except Exception as e:
            self.logger.error(f"Critique failed: {str(e)}", exc_info=True)
            # Return approval as fallback
            return {"status": "approved", "feedback": "Error in critique, accepting answer"}
