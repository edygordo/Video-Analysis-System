from flask import Flask, render_template, redirect, url_for, session, request
import object_detection_video as obj_det
import os
app = Flask(__name__)


vidFolder = os.path.join('static','vid')
app.config['UPLOAD_FOLDER'] = vidFolder

@app.route('/', methods=["GET"]) # Only get method is allowed for input page
def input_page():
    video_photo = os.path.join(app.config['UPLOAD_FOLDER'],'video_analysis.jpeg')
    return render_template('input_page.html', img = video_photo)

@app.route('/action_page', methods=["POST"])
def action():
    #user_video = request.form.get('video')
    #user_video.save(os.path.join(app.config['UPLOAD_FOLDER'], user_video.filename))
    model, classLabels = obj_det.load_pretrained_model() # Load a pre-trained model
    obj_det.setInputParams(model=model) # Set input parameters to the model
    p1 = obj_det.Process(target=obj_det.real_time_detection(model, classLabels))
    p2 = obj_det.Process(target=obj_det.run_animation(model, classLabels))

    # Start Multiprocess
    p2.start()
    p1.start()
    # Wait till each thread finishes it's task
    p1.join()
    p2.join()
    obj_det.plt.show()
    return render_template('output_page.html')

if __name__ == "__main__":
    app.run(debug=True)
