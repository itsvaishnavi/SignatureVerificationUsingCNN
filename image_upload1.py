from flask import *  
from flask_bootstrap import Bootstrap
import os
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity
from keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)  
bootstrap = Bootstrap(app)
img_name = ''

predict_datagen = ImageDataGenerator(rescale=1./255)

@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success1', methods = ['POST'])  
def success1():  
    if request.method == 'POST':  
        f = request.files['file']  
        fname = os.getcwd()+'/original.png'
        f.save(fname)  
        return render_template("success1.html", name = f.filename, path= fname)  

@app.route('/success2', methods = ['POST'])  
def success2():  
    if request.method == 'POST':  
        f = request.files['file']  
        sfname = os.getcwd()+'/test_d/test.png'
        f.save(sfname)
        return render_template("success2.html", name = f.filename, p=sfname) 

@app.route('/get_result', methods = ['POST'])
def get_result():
    print("img_name",img_name)
    print(os.getcwd()+"/"+img_name)

    model = load_model('model1.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    img=Image.open(os.getcwd()+"/test_d/"+'test.png')
    img = img.convert('L')
    img = img.resize((500, 500), Image.ANTIALIAS)
    img.show("test image")
    img_org=Image.open(os.getcwd()+"/"+'original.png')
    img_org=img_org.convert('L')
    img_org=img_org.resize((500,500), Image.ANTIALIAS)
    img_org.show("original image")
    #predict_data=[]
    #predict_data.append([np.array(img)])
    #predictImages = np.array([i for i in predict_data]).reshape(-1, 500)
    #result = model.predict(predictImages)
    #print("Result==>",result)
    """
    msg=''
    if result[0][0]>result[0][1]:
        msg='TEST SIGNATURE IS GENUINE'
        pass
    else:
        msg='TEST SIGNATURE IS FORGE'
    """
    img=np.array(img)
    img_org=np.array(img_org)
    val=structural_similarity(img, img_org, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
    val=val*100

    img = image.load_img(os.getcwd()+'/test_d/test.png', target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    img_class = model.predict_classes(x)
    prediction = img_class[0]
    classname = img_class[0]
    print("Class: ",classname)
    classn=""
    if classname==[1]:
        classn="TEST SIGNATURE SAMPLE IS GENUINE"
    else:
        classn="TEST SIGNATURE SAMPLE IS FORGE"    

    
    return render_template("get_result.html", m=classn, match=val)
  
if __name__ == '__main__':  
    app.run(debug = True)  