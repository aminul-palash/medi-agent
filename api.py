from flask import Flask, request, jsonify, render_template
from main import setup_agent
from logger_config import get_logger

app = Flask(__name__)

# Get API logger
api_logger = get_logger('api')

# Initialize agent on startup
api_logger.info("Starting Medical Agent API")
agent = None

try:
    from main import setup_agent
    api_logger.info("Initializing agent...")
    agent = setup_agent()
    api_logger.info("Agent initialized successfully")
except Exception as e:
    api_logger.error(f"Failed to initialize agent: {str(e)}", exc_info=True)
    raise

@app.route('/')
def index():
    api_logger.info("Index page accessed")
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question', '')
        
        api_logger.info(f"Received question: '{question}'")
        
        if not question:
            api_logger.warning("Empty question received")
            return jsonify({"error": "No question provided"}), 400
        
        result = agent.run(question)
        
        api_logger.info(f"Successfully processed question, returned answer with {result['sources']} sources")
        
        return jsonify({
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"]
        })
    
    except Exception as e:
        api_logger.error(f"Error in /ask endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    try:
        api_logger.info("Clear history requested")
        agent.clear_history()
        return jsonify({"status": "success", "message": "History cleared"})
    except Exception as e:
        api_logger.error(f"Error clearing history: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    api_logger.debug("Health check requested")
    return jsonify({"status": "healthy", "agent": "ready"})

if __name__ == '__main__':
    api_logger.info("Starting Flask server on port 5000")
    app.run(debug=True, port=5000)