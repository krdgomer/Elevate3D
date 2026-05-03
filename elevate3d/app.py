from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
import sys
import logging
from werkzeug.utils import secure_filename
from pathlib import Path
from elevate3d.run_pipeline import Pipeline
import platform
from elevate3d.config import ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH, MODEL_DIR, DATA_DIR, UPLOAD_DIR
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def create_app():
    app = Flask(__name__)

    # Configuration
    app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

    # Get local data directory within project
    def get_data_dir():
        """Returns local data directory within the project root"""
        # Get the project root (parent of elevate3d package)
        project_root = Path(__file__).parent.parent
        return project_root / "data"

    # Set up file storage paths
    data_dir = get_data_dir()
    app.config['UPLOAD_FOLDER'] = str(data_dir / UPLOAD_DIR)
    app.config['MODEL_FOLDER'] = str(data_dir / MODEL_DIR)
    
    # Ensure directories exist
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    Path(app.config['MODEL_FOLDER']).mkdir(parents=True, exist_ok=True)

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    @app.route('/')
    def index():
        # Handle both development and installed package cases
        try:
            from templates import templates
            return send_from_directory(Path(templates.__file__).parent, 'index.html')
        except:
            return send_from_directory('templates', 'index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            try:
                # Validate image content
                try:
                    img = Image.open(file)
                    img.verify()  # Check if it's a valid image
                    file.seek(0)  # Reset file pointer
                except Exception as e:
                    logging.error(f"Invalid image file: {e}")
                    return jsonify({'error': 'Invalid image file'}), 400

                # Save uploaded file
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)
                
                # Generate unique model filename
                model_filename = f"{uuid.uuid4()}.glb"
                model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
                
                # Process the image using the Pipeline class
                pipeline = Pipeline(upload_path, model_path)
                pipeline.run()
                
                # Verify model was created
                if not os.path.exists(model_path):
                    return jsonify({'error': 'Model generation failed'}), 500
                
                return jsonify({
                    'model_url': f"/models/{model_filename}",
                    'debug_path': model_path  # For debugging
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'traceback': str(sys.exc_info())
                }), 500
        
        return jsonify({'error': 'Invalid file type'}), 400

    @app.route('/models/<filename>')
    def serve_model(filename):
        return send_from_directory(app.config['MODEL_FOLDER'], filename)

    @app.route('/debug')
    def debug_info():
        return jsonify({
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'model_folder': app.config['MODEL_FOLDER'],
            'exists': {
                'upload_folder': os.path.exists(app.config['UPLOAD_FOLDER']),
                'model_folder': os.path.exists(app.config['MODEL_FOLDER'])
            },
            'files_in_model_folder': os.listdir(app.config['MODEL_FOLDER'])
        })

    return app

def run_app():
    app = create_app()
    app.run()