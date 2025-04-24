from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
import sys
from werkzeug.utils import secure_filename
from pathlib import Path
from elevate3d.run_pipeline import run_pipeline
import platform

def create_app():
    app = Flask(__name__)

    # Configuration
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

    # Get platform-specific data directories
    def get_data_dir():
        """Returns platform-appropriate data directory"""
        home = Path.home()
        
        if platform.system() == "Windows":
            return home / "AppData" / "Local" / "Elevate3D"
        elif platform.system() == "Darwin":  # Mac
            return home / "Library" / "Application Support" / "Elevate3D"
        else:  # Linux/Unix
            return home / ".local" / "share" / "elevate3d"

    # Set up file storage paths
    data_dir = get_data_dir()
    app.config['UPLOAD_FOLDER'] = str(data_dir / 'uploads')
    app.config['MODEL_FOLDER'] = str(data_dir / 'models')
    
    # Ensure directories exist
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    Path(app.config['MODEL_FOLDER']).mkdir(parents=True, exist_ok=True)

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/')
    def index():
        # Handle both development and installed package cases
        try:
            from elevate3d import templates
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
                # Save uploaded file
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)
                
                # Generate unique model filename
                model_filename = f"{uuid.uuid4()}.glb"
                model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
                
                # Process the image
                run_pipeline(upload_path, model_path)
                
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