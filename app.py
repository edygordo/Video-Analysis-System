from flask import Flask, render_template, redirect, url_for, session
import os
app = Flask(__name__)

imgFolder = os.path.join('static','img')
app.config['UPLOAD_FOLDER'] = imgFolder

@app.route('/', methods=["GET"]) # Only get method is allowed for input page
def input_page():
    video_photo = os.path.join(app.config['UPLOAD_FOLDER'],'video_analysis.jpeg')
    return render_template('input_page.html', img = video_photo)

@app.route('/action_page', methods=["POST"])
def action():
    return f"Backend Not created yet :-()"

if __name__ == "__main__":
    app.run(debug=True)
