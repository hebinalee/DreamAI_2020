# coding=utf-8
import sys
sys.path.insert(0, '/home/u00u664m0mjzfAyslY357/task1_3/sw_code')
import os
import io
import json
import glob
import re
import numpy as np

# Keras
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, flash
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# +
import sw_model,model, util
# -
import time
import torchaudio.transforms as AT
import librosa
import torchaudio
import torch
import torch.nn as nn
import random
# Model saved with Keras model.save()
# basepath = 'Z:/hblee/dreamai/'
basepath = './'
UPLOAD_FOLDER = basepath + 'uploads'
ALLOWED_audio = {'wav', 'mp3'}
ALLOWED_image = {'png', 'jpg', 'jpeg'}

# Define a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# dataset_name = 'image'
num_classes = 2



# -
total_tic = time.time()

def load_image(img_path, dim, augmentation='test'):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)


        #image_tensor = transform(image)

        return image
    
def transform_image(infile):
    input_transforms = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    my_transforms = transforms.Compose(input_transforms)
    image = load_image(img_path=infile, dim=(224,224))                            # 이미지 파일 열기
    timg = my_transforms(image)                           
    timg.unsqueeze_(0)                                 
    return timg 

def load_audio(path):
    if not os.path.exists(path):
        print("AUDIO DOES NOT EXIST {}".format(path))
    audio, sr = librosa.load(path, sr=16000)
    return audio
def sw_transform_audio(infile):
    segment_length = 16000*4
    
    audio = load_audio(infile)
    audio = torch.from_numpy(audio)
    audio = mel_spectrogram(audio)
        # Take segment
    if audio.size(0) >= segment_length:
        max_audio_start = audio.size(0) - segment_length
        audio_start = random.randint(0, max_audio_start)
        audio = audio[audio_start:audio_start+segment_length]
    else:
        audio = torch.nn.functional.pad(audio, (0, segment_length - audio.size(0)), 'constant').data
    
    
    print(audio.shape)
    audio = audio.unsqueeze(0)
    print(audio.shape)
    return audio.unsqueeze(0)
def transform_audio(infile):
    segment_length = 16000*4
    
    audio = load_audio(infile)
    audio = torch.from_numpy(audio)
    
        # Take segment
    if audio.size(0) >= segment_length:
        max_audio_start = audio.size(0) - segment_length
        audio_start = random.randint(0, max_audio_start)
        audio = audio[audio_start:audio_start+segment_length]
    else:
        audio = torch.nn.functional.pad(audio, (0, segment_length - audio.size(0)), 'constant').data
    
    
    print(audio.shape)
    audio = audio.unsqueeze(0)
    print(audio.shape)
    return audio.unsqueeze(0)

mel_spectrogram = nn.Sequential(
                                AT.MelSpectrogram(sample_rate=16000, 
                                                  n_fft=512, 
                                                  win_length=400,
                                                  hop_length=160,
                                                  n_mels=80,
                                                  f_max=8000
                                                  ),
                                AT.AmplitudeToDB())
# Get a prediction
def get_prediction(image_tensor, audio_tensor, mode = ''):
    '''
    Help yt-oh!
    '''
    out=''
    
    if mode == 'image_audio':
        mymodel = model.Image_Audio_Ensemble(classes_num=2, extractor=True)
        mymodel.load_state_dict(torch.load('/home/u00u664m0mjzfAyslY357/task1_3/sw_code/check_point/image_audio_7_checkpoint.pth.tar'))
        mymodel.cuda()
        mymodel.eval()
        tic = time.time()
        outputs = mymodel(image_tensor.cuda(),audio_tensor.cuda())
        toc = time.time()
        _, preds = torch.max(outputs, 1)
        prediction = preds.item()
        out = str(prediction)
    
    else : 
        mymodel = sw_model.MMNet(num_classes=2)

        state_dict = torch.load('/home/u00u664m0mjzfAyslY357/Web2/mmnet1')

        del state_dict['audio_m.mel_spectrogram.0.spectrogram.window']
        del state_dict['audio_m.mel_spectrogram.0.mel_scale.fb']
        del state_dict['audio_m.mfcc.dct_mat']
        del state_dict['audio_m.mfcc.MelSpectrogram.spectrogram.window']
        del state_dict['audio_m.mfcc.MelSpectrogram.mel_scale.fb']
        mymodel.load_state_dict(state_dict)
        mymodel.cuda()
        mymodel.eval()
        
        if mode == 'image':
            audio_dummy = torch.randn(1, 1, 862, 80).cuda()
            tic = time.time()
            outputs,_ = mymodel((audio_dummy, image_tensor.cuda() ))
            toc = time.time()
            _, preds = torch.max(outputs, 1)
            prediction = preds.item()
            out = str(prediction)
        elif mode == 'audio':
            image_dummy = torch.randn(1, 3, 224, 224).cuda()
            tic = time.time()
            _,outputs = mymodel((audio_tensor.cuda(), image_dummy ))
            toc = time.time()
            _, preds = torch.max(outputs, 1)
            prediction = preds.item()
            out = str(prediction)
        
    toc = time.time()
    dur = toc - tic
    
    
    return out, dur


# Make the prediction human-readable
def render_prediction(prediction_idx):
    '''
    Help yt-oh!
    '''
    #stridx = str(prediction_idx)
    stridx = prediction_idx
    class_name = ''
    if stridx == '0' : class_name = 'negative'
    elif stridx == '1' : class_name = 'positive'

    return prediction_idx, class_name


def allowed_aud(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_audio


def allowed_img(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_image


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('demo.html')

'''
@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})
'''

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Get the file from post request
        f_audio = request.files['audio_file[]']
        f_image = request.files['image_file[]']
        
        print(f_audio, f_image)
        
        mode=''
        if f_audio.filename == '':
            mode = 'image'
            print('mode : ', mode)
        elif f_image.filename == '':
            mode = 'audio'
            print('mode : ', mode)
        else : 
            mode = 'image_audio'
            print('mode : ', mode)
        #input_data=[]
        
        if mode == 'image':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f_image.filename))
            f_image.save(file_path)
            #input_data.append(file_path)
            image_tensor = transform_image(file_path)
            preds, dur = get_prediction(image_tensor,None, mode)
            class_id, class_name = render_prediction(preds)
            total_toc = time.time()
            total_dur = total_toc - total_tic
            result1 = 'This subject is tested ' + str(class_name) + ' for COVID-19!' 
            result2 =  'It took {}sec||'.format(format(dur,".3f")) # Convert to string
            result3 =  'Total time {}sec'.format(format(total_dur,".3f"))
            result = result1 + result2 + result3
            return result
#             input_data.append(image_tensor)
        
        elif mode == 'audio':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f_audio.filename))
            f_audio.save(file_path)
            #input_data.append(file_path)
            audio_tensor = sw_transform_audio(file_path)
            preds, dur = get_prediction(None,audio_tensor, mode)
            class_id, class_name = render_prediction(preds)
            total_toc = time.time()
            total_dur = total_toc - total_tic
            result1 = 'This subject is tested ' + str(class_name) + ' for COVID-19!' 
            result2 =  ' It took {}sec||'.format(format(dur,".3f")) # Convert to string
            result3 =  ' Total time {}sec '.format(format(total_dur,".3f"))
            result = result1 + result2 + result3
            return result
            
        elif mode == 'image_audio':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f_image.filename))
            f_image.save(file_path)
            image_tensor = transform_image(file_path)
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f_audio.filename))
            f_audio.save(file_path)
            audio_tensor = transform_audio(file_path)
            preds, dur = get_prediction(image_tensor,audio_tensor, mode)
            class_id, class_name = render_prediction(preds)
            total_toc = time.time()
            total_dur = total_toc - total_tic
            result1 = 'This subject is tested ' + str(class_name) + ' for COVID-19!' 
            result2 =  'It took {}sec||'.format(format(dur,".3f")) # Convert to string
            result3 =  'Total time {}sec'.format(format(total_dur,".3f"))
            result = result1 + result2 + result3
            return result
            
            
#         if f_audio and allowed_aud(f_audio.filename):
#             flash(f_audio.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f_audio.filename))
#             f_audio.save(file_path)
#             audio_tensor = transform_audio(file_path)
#             input_data.append(audio_tensor)

#         if f_image and allowed_img(f_image.filename):
#             flash(f_image.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f_image.filename))
#             f_image.save(file_path)
#             image_tensor = transform_image(file_path)
#             input_data.append(image_tensor)

#         if allowed_aud(f_audio.filename) and allowed_img(f_image.filename):
#             '''
#             Model prediction!!
#             '''
            
#             #preds = get_prediction(image_tensor,audio_tensor)
#             preds, dur = get_prediction(input_data[1],input_data[0])
#             class_id, class_name = render_prediction(preds)
#             total_toc = time.time()
#             total_dur = total_toc - total_tic
#             result1 = 'This subject is tested ' + str(class_name) + ' for COVID-19!' 
#             result2 =  'It took {}sec'.format(dur) # Convert to string
#             result3 =  'Total time {}sec'.format(total_dur)
#             result = result1 + result2 + result3
#             return result
        else:
            return 'ok'

    else:
        return 'ok'


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host="0.0.0.0", port=4545, debug=True)

