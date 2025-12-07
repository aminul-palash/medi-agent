"""
Centralized logging configuration for Medical RAG Agent
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

class AgentLogger:
    """Singleton logger for the agent system"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """Setup logging configuration"""
        
        # Create logs directory
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            fmt='%(levelname)s: %(message)s'
        )
        
        # Main application logger
        self.app_logger = logging.getLogger('medical_agent')
        self.app_logger.setLevel(logging.DEBUG)
        self.app_logger.handlers.clear()
        
        # File handler - All logs (rotating)
        all_logs_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, 'agent.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        all_logs_handler.setLevel(logging.DEBUG)
        all_logs_handler.setFormatter(detailed_formatter)
        
        # File handler - Errors only
        error_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, 'errors.log'),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Console handler - Info and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.app_logger.addHandler(all_logs_handler)
        self.app_logger.addHandler(error_handler)
        self.app_logger.addHandler(console_handler)
        
        # Component-specific loggers
        self.retriever_logger = self._create_component_logger('retriever')
        self.critic_logger = self._create_component_logger('critic')
        self.agent_logger = self._create_component_logger('agent')
        self.api_logger = self._create_component_logger('api')
    
    def _create_component_logger(self, component_name):
        """Create logger for specific component"""
        logger = logging.getLogger(f'medical_agent.{component_name}')
        logger.setLevel(logging.DEBUG)
        
        # Component-specific log file
        handler = RotatingFileHandler(
            filename=os.path.join('logs', f'{component_name}.log'),
            maxBytes=5*1024*1024,
            backupCount=3
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        logger.addHandler(handler)
        return logger
    
    def get_logger(self, component=None):
        """Get logger for specific component or main logger"""
        if component == 'retriever':
            return self.retriever_logger
        elif component == 'critic':
            return self.critic_logger
        elif component == 'agent':
            return self.agent_logger
        elif component == 'api':
            return self.api_logger
        else:
            return self.app_logger


# Helper function for easy access
def get_logger(component=None):
    """Get logger instance"""
    agent_logger = AgentLogger()
    return agent_logger.get_logger(component)