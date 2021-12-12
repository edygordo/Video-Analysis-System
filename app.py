import time
import threading
from bokeh.models.layouts import Panel, Tabs
from bokeh.models.sources import ColumnDataSource
from bokeh.server.server import Server
import cv2
import numpy as np
from tornado.ioloop import IOLoop
from bokeh.embed import server_document
from bokeh.resources import INLINE
from bokeh.themes.theme import Theme
from bokeh.models import ImageURL,LinearAxis, Range1d, Plot
from flask import Flask, render_template, request, flash, redirect, url_for, Response, jsonify, make_response
from werkzeug.utils import secure_filename
import object_detection_video as obj_det
import os
import pandas as pd
from bokeh.plotting import figure
from PIL import Image
import matplotlib.pyplot as plt



app = Flask(__name__)
app.secret_key = "secret key"
# Setting up video path for user uploaded video
vidFolder = os.path.join('static')
app.config['UPLOAD_FOLDER'] = vidFolder
app.config['MAX _CONTENT_LENGTH'] = 100*1024*1024

def extract_Frame_Number(row,Person):
    if row['Person Count'] == Person:
        return int(row['Frame Number'])

def active_frames(row):
    if row['Activity Indicator'] == True:
        return row
    else:
        return None
    
def inactive_frames(row):
    if row['Activity Indicator'] == False:
        return row
    else:
        return None


def gen_frames():
    camera = cv2.VideoCapture(0)
    model, classLabels = obj_det.load_pretrained_model() # Load a pre-trained model
    while True:
        success, frame = camera.read()
        height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width  = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        obj_det.setInputParams(model=model,width=width,height=height) # Set input parameters to the model
        if not success:
            break
        else:
            frame = obj_det.generate_processed_frame(model=model,frame=frame,classLabels=classLabels) # Frame processed
            ret, buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

 # Sending Locally generated frames to the server
# def gen_proc_frames():
#     while True:
#         success, frame = cv2.imread('./videos/processed/my_video_feed.jpg')
#         if not success:
#             continue
#         else:
#             ret, buffer = cv2.imencode('.jpg',frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

# @app.route('/processed_feed')
# def processed_feed():
#     return Response(gen_proc_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# def generate_people_count():
#     while True:
#         last_row = pd.read_csv('Data Files/spatial.csv').iloc[-1]
#         Curr_frame, PersonCount, ActivityIndicator, Seconds = last_row[0], last_row[1], last_row[2], last_row[3]
#         if PersonCount is not None:
#             yield str(PersonCount)

# @app.route('/person_info')
# def person_info():
#     return Response(generate_people_count(),mimetype='text/csv')

def bkapp(doc):
    DataSource = ColumnDataSource(dict(Frame_Number=[],Person_Present=[],Frame_Active=[],Seconds=[]))

                    ### Live Chart Creation ###
    p = figure(title='Live Analysis of video',width=900, height=350) # A figure on which plots will be generated
    p.circle(x="Seconds",y="Person_Present",source=DataSource)
    p.line(x="Seconds",y="Person_Present", source=DataSource, legend_label="Person Counter", line_width=2, line_color="blue")
    p.step(x="Seconds",y="Frame_Active", source=DataSource, legend_label="Frame Active", line_width=1, line_color="red")
    p.legend.title = "Video Statistics"
    p.legend.location = "top_left" # Causing NSWindow drag region error!
    p.legend.click_policy="hide"
    p.xaxis.axis_label = "Seconds"
    p.yaxis.axis_label = "Person Count"
                    ### Live Chart Creation ###

                    ### Proccessed Video Frames ###
    # url = "http://localhost:5006/bkapp/videos/processed/my_video_feed.jpg"
    # url = "videos"
    # url = "https://static.bokeh.org/logos/logo.png"
    #url = url_for('video')
    # N = 5
    # source = ColumnDataSource(dict(
    # url = [url]*N,
    # x1  = np.linspace(  0, 150, N),
    # y1  = np.linspace(  0, 150, N),
    # w1  = np.linspace( 10,  50, N),
    # h1  = np.linspace( 10,  50, N),
    # x2  = np.linspace(-50, 150, N),
    # y2  = np.linspace(  0, 200, N),
    # ))
    # xdr = Range1d(start=-100, end=200)
    # ydr = Range1d(start=-100, end=200)
    # #plot = Plot(
    #title="Mobile Net SSD Model", x_range=xdr, y_range=ydr, width=1020, height=720,
    #min_border=0, toolbar_location=None)
    #image1 = ImageURL(url="url", x="x1", y="y1", w="w1", h="h1", anchor="center")
    #plot.add_glyph(source, image1)
    #- plot = figure(x_range=(0, 1), y_range=(0, 1))
    #- plot.image_url(url=['videos/processed/my_video_feed.jpg'], x=0, y=1, w=0.8, h=0.6)
    #xaxis = LinearAxis()
    #plot.add_layout(xaxis, 'below')
    #yaxis = LinearAxis()
    #plot.add_layout(yaxis,'left')

            # Forming tabs of figure
    Live_chart = Panel(child = p, title="Live Chart Analysis")
    #Proccessed_video = Panel(child = plot, title="Processed Frames")
    tabs = Tabs(tabs=[Live_chart])

    def update():
        # Need to update source stream of the figure
        # Only extract the last row of the dynamic CSV file
        last_row = pd.read_csv('Data Files/spatial.csv',sep=',').iloc[-1]
        Curr_frame, PersonCount, ActivityIndicator, Seconds = last_row[0], last_row[1], last_row[2], last_row[3]
        #print(f'Current Frame:-{Curr_frame}, Person Present:-{PersonCount}.')
        new_data = dict(Frame_Number=[Curr_frame],Person_Present=[PersonCount]
        ,Frame_Active=[ActivityIndicator], Seconds=[Seconds])
        DataSource.stream(new_data,rollover=200)
        
        #image1 = ImageURL(url="url",retry_attempts=10, x="x1", y="y1", w="w1", h="h1", anchor="center")
        #plot.add_glyph(source, image1)
        #plot.image_url(url=['videos/processed/my_video_feed.jpg'], x=0, y=1, w=0.8, h=0.6)

    doc.add_root(tabs)
    doc.add_periodic_callback(update,1000)
    #doc.theme = Theme(filename="theme.yaml")

@app.route('/', methods=["GET"]) # Only get method is allowed for input page
def input_page():
    video_photo = os.path.join(app.config['UPLOAD_FOLDER'],'img/video_analysis.jpeg')
    return render_template('input_page.html',img = video_photo,js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css()).encode(encoding='UTF-8')

@app.route('/display/<filename>')
def display_video(filename):
	#print('display_video filename: ' + filename)
	return redirect(url_for('static', filename='vid/' + filename), code=301)

def online_processing(request):
    model, classLabels = obj_det.load_pretrained_model() # Load a pre-trained model    
    # THIS THREAD CREATES A LIVE CSV FROM WHICH DATA HAS TO BE READ
    my_video = cv2.VideoCapture(0)
    height = my_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width  = my_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    #print(f'Video Height is:-{height} and width is:- {width}')
    obj_det.setInputParams(model=model,width=width,height=height) # Set input parameters to the model
    thread1 = threading.Thread(target=obj_det.online_processing, kwargs={'model':model,'classLabels':classLabels})
    thread1.start() 
    script = server_document('http://165.22.223.34:5006/bkapp') # url to bokeh application , localhost->To server localhost
    return script

def offline_processing(request):
    model, classLabels = obj_det.load_pretrained_model() # Load a pre-trained model
    uploaded_file_path = ""
    if 'file' not in request.files:
        flash('Video not uploaded')
        return redirect(request.url)
    else:
        file = request.files['file'] # get the video file, gives "on" if this button chosen
        if file.filename == '':
            flash('Please upload video in .mp4 format')
            return redirect(request.url)
        else: # Valid file obtained
            filename = secure_filename(file.filename)
            uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'],'vid/'+filename)
            file.save(uploaded_file_path) # save the video locally in static/vid
            flash('Video has been uploaded starting Processing now')

        video_src = uploaded_file_path # read this uploaded video
        # THIS THREAD CREATES A LIVE CSV FROM WHICH DATA HAS TO BE READ
        my_video = cv2.VideoCapture(uploaded_file_path)
        height = my_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width  = my_video.get(cv2.CAP_PROP_FRAME_WIDTH)
        #print(f'Video Height is:-{height} and width is:- {width}')
        obj_det.setInputParams(model=model,width=width,height=height) # Set input parameters to the model
        thread1 = threading.Thread(target=obj_det.offline_processing, kwargs={'model':model,'classLabels':classLabels,
        'video_src':video_src})
        thread1.start() 
        script = server_document('http://165.22.223.34:5006/bkapp') # url to bokeh application , localhost->To server localhost 
        return script, filename

@app.route('/action_page', methods=["POST"])
def action(): # currently independent of user uploaded video
    if 'processing_type_live' in request.form.keys():
            script = online_processing(request=request)
            return render_template('output_page_live.html',script = script, template="Flask")
    else:
        #camera.release()
        script, filename = offline_processing(request=request)
        return render_template('output_page.html',script = script, template="Flask",filename = filename)


def bk_worker():
    server = Server({'/bkapp':bkapp}, io_loop=IOLoop(), allow_websocket_origin=["165.22.223.34:8003"])
    server.start()
    server.io_loop.start()


@app.route('/gen_data', methods=["POST"])
def gen_data():
    my_file = pd.read_csv('./Data Files/spatial.csv')
    Person_col = my_file['Person Count']
    Max_people = Person_col.max(axis=0)
    # Min_people = Person_col.min(axis=0)
    # Active_Frames = my_file['Activity Indicator']
    # Total_Frames = len(my_file)
    Busiest_Frames = my_file.apply(extract_Frame_Number,axis=1, args=[Max_people])
    Busiest_Frames.dropna(inplace=True)
    Busiest_Frames = Busiest_Frames.to_numpy(dtype=int)
    # Get Busiest Frames in the processed videos
    reader = cv2.VideoCapture('./videos/processed/my_video_feed.avi') # The Processed Video frames written by Thread-1
    Images = []
    for frame in Busiest_Frames:
        reader.set(cv2.CAP_PROP_POS_FRAMES,frame)
        Images.append(reader.read())
    Total_Images = len(Images)
    for x in range(Total_Images):
        im = Image.fromarray(Images[x][1])
        im.save(f'./Data Files/Image_{x}.jpg')
    
    if Total_Images == 1:
        Image1 = Image.open('./Data Files/Image_0.jpg')
        fig, ax = plt.subplots(1,1)
        ax.imshow(Image1, interpolation='nearest', aspect='auto')
        ax.set_title('Frame 1')
        #plt.show()

    if Total_Images == 2:
        Image1 = Image.open('./Data Files/Image_0.jpg')
        Image2 = Image.open('./Data Files/Image_1.jpg')
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(Image1, interpolation='nearest', aspect='auto')
        ax[0].set_title('Frame 1')
        ax[1].imshow(Image2, interpolation='nearest', aspect='auto')
        ax[1].set_title('Frame 2')
        #print(Image1.size)
        #plt.show()

    if Total_Images == 3:
        Image1 = Image.open('./Data Files/Image_0.jpg')
        Image2 = Image.open('./Data Files/Image_1.jpg')
        Image3 = Image.open('./Data Files/Image_2.jpg')
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(Image1, interpolation='nearest', aspect='auto')
        ax[0].set_title('Frame 1')
        ax[1].imshow(Image2, interpolation='nearest', aspect='auto')
        ax[1].set_title('Frame 2')
        ax[2].imshow(Image3, interpolation='nearest', aspect='auto')
        ax[2].set_title('Frame 3')
        #print(Image1.size)
        #plt.show()
        
        
    if Total_Images >= 4:
        Image1 = Image.open('./Data Files/Image_0.jpg')
        Image2 = Image.open('./Data Files/Image_1.jpg')
        Image3 = Image.open('./Data Files/Image_2.jpg')
        Image4 = Image.open('./Data Files/Image_3.jpg')
        fig, ax = plt.subplots(2,2)
        ax[0,0].imshow(Image1, interpolation='nearest', aspect='auto')
        ax[0,0].set_title('Frame 1')
        ax[0,1].imshow(Image2, interpolation='nearest', aspect='auto')
        ax[0,1].set_title('Frame 2')
        ax[1,0].imshow(Image3, interpolation='nearest', aspect='auto')
        ax[1,0].set_title('Frame 3')
        ax[1,1].imshow(Image4, interpolation='nearest', aspect='auto')
        ax[1,1].set_title('Frame 4')

    # Calculating Ideal time and Active Time

    Active_frames = my_file.apply(active_frames,axis=1)
    Active_frames.dropna(inplace=True)

    InActive_frames = my_file.apply(inactive_frames,axis=1)
    InActive_frames.dropna(inplace=True)

    Interesting_ratio = len(Active_frames)/(len(Active_frames)+len(InActive_frames))
    Interesting_ratio = float(int(Interesting_ratio*10000)/10000)
    #print(Interesting_ratio)
    fig.savefig('./static/img/Results.jpeg')
    gen_res = os.path.join(app.config['UPLOAD_FOLDER'],'img/Results.jpeg')
    return render_template('Generated_Result.html', Interesting_ratio=Interesting_ratio, gen_res=gen_res)

threading.Thread(target=bk_worker).start() # this thread starts bokeh app on localhost:8000

if __name__ == "__main__":

    app.run(debug=True, threaded = True, use_reloader=False, port=8003)
