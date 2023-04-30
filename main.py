from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__,template_folder='template')
model = load_model('mnist.h5')


def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=True, target_size=(28, 28))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = x/255.0

    preds = model.predict(x)
    preds=np.argmax(preds)
    return preds


@app.route('/')
def upload_file():
   return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_image_file():
  if request.method == 'POST':
    f = request.files['image']
    basepath = globals()['_dh'][0]
    file_path = os.path.join(
        basepath, secure_filename(f.filename))
    f.save(file_path)

    # Make prediction
    preds = model_predict(file_path, model)
    
    return render_template('index.html', output='Predicted Number is :{}'.format(preds))
  return None


if __name__ == '__main__':
   print(("* Loading Keras model and Flask starting server..."
      "please wait until server has fully started"))
   app.run()