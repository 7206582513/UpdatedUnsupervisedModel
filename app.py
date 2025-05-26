from flask import Flask, render_template, request
from utils import process_and_cluster
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                df_result, best_model, best_score, all_scores, explanation, plot_path = process_and_cluster(filepath)

                return render_template('result.html',
                                       best_model=best_model,
                                       best_score=best_score,
                                       all_scores=all_scores,
                                       explanation=explanation,
                                       image_path='/' + plot_path)
            except Exception as e:
                return f"<h4>Error processing file:</h4><pre>{e}</pre>"

        else:
            return "<h4>Please upload a valid .csv file.</h4>"

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
