from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from process_image import process_image

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            result_filename = 'result_' + file.filename
            result_filename = result_filename.rsplit('.', 1)[0] + '.png'
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            process_image(file_path, result_path)

            print(f"Processed file saved as: {result_path}")

            return redirect(url_for('result', filename=result_filename))
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
