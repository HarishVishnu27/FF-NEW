from threading import Thread
import database
import vision_processing
from routes import app

if __name__ == "__main__":
    # Initialize the database
    database.init_db()
    
    # Start the background processing thread
    processing_thread = Thread(target=vision_processing.cyclic_processing, daemon=True)
    processing_thread.start()
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=3000, debug=True, threaded=True)
