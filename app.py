import threading
from bokeh.io.doc import curdoc
from bokeh.models.sources import ColumnDataSource
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
from bokeh.embed import server_document
from bokeh.resources import INLINE
from bokeh.themes.theme import Theme
from flask import Flask, render_template, Response
import object_detection_video as obj_det
import os
import global_holder
import pandas as pd
import cProfile
from bokeh.plotting import figure
import bokeh.server.views.ws

app = Flask(__name__)

# Setting up video path for user uploaded video
vidFolder = os.path.join('static','vid')
app.config['UPLOAD_FOLDER'] = vidFolder

def bkapp(doc):
    DataSource = ColumnDataSource(dict(Frame_Number=[],Person_Present=[],Frame_Active=[]))
    p = figure(width=900, height=350) # A figure on which plots will be generated
    p.circle(x="Frame_Number",y="Person_Present",source=DataSource)
    p.line(x="Frame_Number",y="Person_Present", source=DataSource, legend_label="Person Counter", line_width=2, line_color="blue")
    p.step(x="Frame_Number",y="Frame_Active", source=DataSource, legend_label="Frame Active", line_width=1, line_color="red")
    p.legend.title = "Video Statistics"
    p.xaxis.axis_label = "Frame Number"
    p.yaxis.axis_label = "Person Count"

    def update():
        # Need to update source stream of the figure
        # Only extract the last row of the dynamic CSV file
        last_row = pd.read_csv('Data Files/spatial.csv',sep=',').iloc[-1]
        Curr_frame, PersonCount, ActivityIndicator = last_row[0], last_row[1], last_row[2]
        print(f'Current Frame:-{Curr_frame}, Person Present:-{PersonCount}.')
        new_data = dict(Frame_Number=[Curr_frame],Person_Present=[PersonCount],Frame_Active=[ActivityIndicator])
        DataSource.stream(new_data,rollover=200)
        
    doc.add_root(p)
    doc.add_periodic_callback(update,1000)
    #doc.theme = Theme(filename="theme.yaml")

@app.route('/', methods=["GET"]) # Only get method is allowed for input page
def input_page():
    video_photo = os.path.join(app.config['UPLOAD_FOLDER'],'video_analysis.jpeg')
    return render_template('input_page.html',img = video_photo,js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css()).encode(encoding='UTF-8')

        
@app.route('/action_page', methods=["POST"])
def action(): # currently independent of user uploaded video
    model, classLabels = obj_det.load_pretrained_model() # Load a pre-trained model
    obj_det.setInputParams(model=model) # Set input parameters to the model
    video_src = 'videos/street_video_1.mp4'
    # THIS THREAD CREATES A LIVE CSV FROM WHICH DATA HAS TO BE READ
    thread1 = threading.Thread(target=obj_det.real_time_detection, kwargs={'model':model,'classLabels':classLabels,
    'video_src':video_src})
    thread1.start() 
    script = server_document('http://localhost:5006/bkapp') # url to bokeh application
    return render_template('output_page.html',script = script, template="Flask")

def bk_worker():
    server = Server({'/bkapp':bkapp}, io_loop=IOLoop(), allow_websocket_origin=["127.0.0.1:8000"])
    server.start()
    server.io_loop.start()

threading.Thread(target=bk_worker).start() # this thread starts bokeh app on localhost:8000

if __name__ == "__main__":

    app.run(debug=True, threaded = True, use_reloader=False, port=8000)
