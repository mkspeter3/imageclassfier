from flask import Flask, redirect, url_for, request, jsonify, render_template
import os
import json
import io
import glob
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import string


app = Flask(__name__)


model = models.densenet121(pretrained=True)

model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

imagenet_class_mapping = json.load(open('imagenet_class_index.json'))

def get_category(image_path):
  # read the image in binary form
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
    # transform the image    
    transformed_image = transform_image(image_bytes=image_bytes)
    # use the model to predict the class
    outputs = model.forward(transformed_image)
    _, category = outputs.max(1)
    # return the value
    predicted_idx = str(category.item())
    return imagenet_class_mapping[predicted_idx]


@app.route('/')
def index ():
    return render_template('index.html')


@app.route('/', methods=['POST','GET'])
def upload_files():
    if request.method == 'POST':
        if 'file' not in request.files: # no filepart in the uploaded form
            return redirect(request.url)
        
        uploaded_file = request.files['file']

        if uploaded_file.filename == '': # No filename of file/ no file
            return redirect(request.url)
        
        if uploaded_file: #there's an uploaded file
            filename = uploaded_file.filename
            file_name_ht= os.path.dirname(filename)
            

            img_bytes = uploaded_file.read()
            transformed_image = transform_image(image_bytes=img_bytes)
            outputs = model.forward(transformed_image)
            _, category = outputs.max(1)
            predicted_idx = str(category.item())
            namee = imagenet_class_mapping[predicted_idx]
            class_name =  str(namee[1])
            return render_template("Result.html", class_name=class_name )

    else: #incase of GET method
        return redirect(request.url)
 
    # return class_name


if __name__ == '__main__' :
    app.run(debug=False)
