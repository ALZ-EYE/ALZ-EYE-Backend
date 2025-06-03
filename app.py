import os
import uuid
import numpy as np
import nibabel as nib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'nii', 'nii.gz'}
app.config['MODEL_PATH'] = 'model.h5'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model(app.config['MODEL_PATH'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_mri(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = np.expand_dims(data, axis=-1)
    data = np.expand_dims(data, axis=0)
    return data

@app.route('/')
def home():
    return "ALZ-EYE Server is Running!"

@app.route('/analyze', methods=['POST'])
def analyze_mri():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            try:
                processed_data = preprocess_mri(filepath)
                prediction = model.predict(processed_data)
                ad = prediction[0][0] * 100
                mci = prediction[0][1] * 100
                cn = prediction[0][2] * 100
                ad = round(ad, 3)
                mci = round(mci, 3)
                cn = round(cn, 3)
                
                return jsonify({
                    'status': 'success',
                    'scan_id': unique_filename.split('.')[0],
                    'AD': ad,
                    'MCI': mci,
                    'CN': cn,
                    'message': 'MRI analysis completed successfully'
                })
                
            except Exception as e:
                return jsonify({'error': f'File processing error: {str(e)}'}), 500
                
            finally:
                try:
                    os.remove(filepath)
                except:
                    pass
                
        else:
            return jsonify({'error': 'Invalid file type. Please upload .nii or .nii.gz file'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)