from django.shortcuts import render
from django.http import HttpResponse

#add scheduler path
import sys,os
sys.path.append('../')
from scheduler import publisher as pb

# Create your views here.

def index(request):
    return HttpResponse("Hello World!")

WEB_ROOT = os.getcwd() + '/'

head_html = '\
<!DOCTYPE html>\n\
<html>\n\
<head>\n\
   <title>Image Retrieve Demo</title>\n\
   <link href="/static/bootstrap/css/bootstrap.min.css" rel="stylesheet">\n\
   <script src="/static/scripts/jquery.min.js"></script>\n\
   <script src="/static/bootstrap/js/bootstrap.min.js"></script>\n\
</head>\n\
<body>\n\
'

html = ""

def target_image():
    pass

def gen_body(img_list):
    body = ""
    for img in img_list:
        print img
        img = img.split('/')
        img[1] = 'images_pool'
        img = '/'.join(img[1:])
        #print os.getcwd()+'/' + img
            
        one = '<img src="'+ img +'" class = "img_thumbnail" style="width:200px;height:200px">\n'
        body += one
    return body

tail_html = '\
</body>\n\
</html>\n\
'

def index(request):
    #print BASE_DIR
    res = pb.retrieve('ILSVRC2013_val_00000016.jpg', 'color_histogram')
    html = head_html + gen_body(res) + tail_html
    return HttpResponse(html)

def image(request, path, document_root):
    #print request, path
    imgpath = WEB_ROOT+'images_pool/' + path
    one = '<img src="'+ imgpath +'" class = "img_thumbnail">\n'
    return HttpResponse(one)