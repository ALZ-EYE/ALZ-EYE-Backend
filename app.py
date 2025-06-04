import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SAVED_ANALYZED_FOLDER'] = 'analyzed_images'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  
app.config['ALLOWED_EXTENSIONS'] = {'nii', 'nii.gz'}
app.config['MODEL_PATH'] = 'model.h5'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAVED_ANALYZED_FOLDER'], exist_ok=True)

try:
    import tensorflow as tf
    tf.keras.utils.disable_interactive_logging()
    
    model = load_model(
        app.config['MODEL_PATH'],
        compile=False,
        custom_objects=None
    )
    print("✅ تم تحميل النموذج بنجاح!")
    print(f"🔍 تفاصيل النموذج: مدخلات {model.input_shape} - مخرجات {model.output_shape}")
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {str(e)}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_nii(filepath):
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        
        if data.shape != (180, 180, 3):
            from skimage.transform import resize
            print(f"⚠️ تغيير حجم الصورة من {data.shape} إلى (180, 180, 3)")
            data = resize(data, (180, 180, 3), preserve_range=True, anti_aliasing=True)
        
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = data.astype(np.float32)
        data = np.expand_dims(data, axis=0)
        
        return data
    except Exception as e:
        print(f"❌ خطأ في معالجة الصورة: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'tensorflow_version': '2.18.0',
        'model_loaded': bool(model)
    })

@app.route('/analyze', methods=['POST'])
def analyze_mri():
    if not model:
        return jsonify({'error': 'النموذج غير متاح حاليًا'}), 503
        
    if 'file' not in request.files:
        return jsonify({'error': 'لم يتم تقديم ملف'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'نوع الملف غير مسموح به',
            'allowed_extensions': list(app.config['ALLOWED_EXTENSIONS'])
        }), 400
        
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"📁 تم استلام ملف: {filename}")
        
        processed_data = preprocess_nii(filepath)
        print("🔄 تمت معالجة الصورة بنجاح")
        
        predictions = model.predict(processed_data)
        print("🔮 تم إجراء التنبؤ بنجاح")
        
        ad = float(predictions[0][0]) * 100
        mci = float(predictions[0][1]) * 100
        cn = float(predictions[0][2]) * 100
        
        return jsonify({
            'status': 'success',
            'AD': round(ad, 3),
            'MCI': round(mci, 3),
            'CN': round(cn, 3),
            'scan_id': str(uuid.uuid4()),
            'shape': str(processed_data.shape)
        })
        
    except Exception as e:
        print(f"❌ خطأ أثناء التحليل: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        if os.path.exists(filepath):
            analyzed_path = os.path.join(app.config['SAVED_ANALYZED_FOLDER'], filename)
            os.rename(filepath, analyzed_path)
            print(f"✅ تم حفظ الصورة المفحوصة في: {analyzed_path}")

if __name__ == '__main__':
    print("🚀 بدء تشغيل الخادم...")
    print(f"🔗 عنوان الخادم: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)