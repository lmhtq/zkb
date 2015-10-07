from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
from django import forms
from django.db import models 
from django.conf import settings
import ImageFile

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
<div class="container">\n\
        <h1 style="text-align:center">Image Retrieve Demo</h1>\n\
'

select_weight = '\n\
<div style="padding: 10px 100px 10px;">\n\
   <form class="bs-example bs-example-form" role="form" action="/demo/" method="post">\n\
       <table class="table table-hover">\n\
            <div class="input-group">\n\
          <tr>\n\
              <td>  ColorHistogram</td><td> <input name="ch" type="text" value="0" ></td>\n\
              <td>   ColorCorrelogram</td><td>  <input name="cc" type="text" value="0" ></td>\n\
            </tr>\n\
            <tr>\n\
              <td>  ColorMoment</td><td> <input name="cm" type="text" value="0" ></td>\n\
              <td>   ColorCoherenceVector</td><td>  <input name="ccv" type="text" value="0" ></td>\n\
            </tr>\n\
            <tr>\n\
                <td>  TextureLBP</td><td> <input name="tl" type="text" value="0" ></td>\n\
              <td>   TextureGabor</td><td>  <input name="tg" type="text" value="0" ></td>\n\
            </tr>\n\
           <tr>\n\
              <td>  ShapeHOG</td><td> <input name="sh" type="text" value="0" ></td>\n\
              <td>   ShapeEOH</td><td>  <input name="se" type="text" value="0" ></td>\n\
            </tr>\n\
            <tr>\n\
            <td></td><td></td>\n\
              <td><input name="img_target" type="text" value="300.jpg"></td>\n\
              <td><input class="btn btn-success" type="submit" value="Retrive!"></td>\n\
            </tr>\n\
              </div>\n\
       </table>\n\
    </form>\n\
        <form action="/upload_img/" method="post" enctype="multipart/form-data">\n\
        <input id="id_image" type="file", class="", name="image" value=""/>\n\
        Rename it!<input name="newname" type="text" value="new001.jpg">\n\
        <input class="btn btn-success" type="submit" value="Upload and Retrive!">\n\
        </form>\n\
</div>\n\
'

tail_html = '\
</div>\n\
</body>\n\
</html>\n\
'

def get_target_html(target_image_name):
    target_image_html = '<div class="container">\n\
    <p><img src="'+ 'images_pool/'+ target_image_name +'" class = "img_thumbnail" style="width:200px;height:200px">\n\
    Retrive Target</p></div>\n\
'
    return target_image_html

html = ""

def target_image():
    pass

def gen_body(img_list):
    body = '<div class="container">'
    for img in img_list:
        print img
        img = img.split('/')
        img[1] = 'images_pool'
        img = '/'.join(img[1:])
        #print os.getcwd()+'/' + img
            
        one = '<img src="'+ img +'" class = "img_thumbnail" style="width:200px;height:200px">\n'
        body += one 
    return body + '</div>'

def index(request):
    res = pb.retrieve('a/300.jpg', 'texture_lbp')
    html = head_html + select_weight + tail_html
    #print request.POST
    #html = "hello"
    return HttpResponse(html)

def image(request, path, document_root):
    #print request, path
    imgpath = WEB_ROOT+'images_pool/' + path
    one = '<img src="'+ imgpath +'" class = "img_thumbnail">\n'
    return HttpResponse(one)

def retrieve(request):
    global target_image_name
    if not request.POST.has_key('img_target'):
        return index(request)

    img_target = request.POST['img_target']
    ch, cc = float(request.POST['ch']), float(request.POST['cc'])
    cm, ccv = float(request.POST['cm']), float(request.POST['ccv'])
    tl, tg = float(request.POST['tl']), float(request.POST['tg'])
    sh, se = float(request.POST['sh']), float(request.POST['se'])
    #normalize
    maxm = max(np.array([ch,cc,cm,ccv,tl,tg,sh,se]) )
    cc /= maxm
    cm /= maxm
    ccv /= maxm
    tl /= maxm
    tg /= maxm
    sh / maxm
    se /= maxm
    ch /= maxm
    if abs(maxm) < 1e-6:
        return index(request)

    ratio = {'color_histogram':ch, 'color_correlogram':cc,\
        'color_moment':cm, 'color_coherence_vector':ccv,\
        'texture_lbp':tl, 'texture_gabor':tg,\
        'shape_hog':sh, 'shape_eoh':se}

    methods = pb.config["extraction_method"]
    ranks = {}
    
    for method in methods:
        if np.abs(ratio[method]) < 1e-6:
            continue 
        realpath = settings.BASE_DIR + '/demo/images_pool/'+img_target
        print realpath
        res_list = pb.retrieve(realpath, method)
        for ind in range(len(res_list)):
            img = res_list[ind]
            if not ranks.has_key(img):
                ranks[img] = 100.0*len(methods)
            ranks[img] -= float(pb.config["topK"] - ind) * ratio[method]

    print 'ranks'
    print ranks
    sorted_dic = sorted(ranks.iteritems(), key=lambda s:s[1])
    res = []
    i = 0
    for k in sorted_dic:
        res.append(k[0])
        i += 1
        if (i >= pb.config["topK"]):
            break
    print res 
    html = head_html + select_weight + get_target_html(img_target) + gen_body(res) + tail_html
    return HttpResponse(html)

#upload pics
class ImageUploadForm(forms.Form):
    image = forms.ImageField()
class ExampleModel(models.Model):
    model_pic = models.ImageField(upload_to = 'upload_folder', default = 'upload_folder/no_img.jpg')
def  get_tip(img_name):
    tip = '\n\
    <div class="alert alert-info">Use THE '+ img_name+' as target_image\'s name to retrive!</div>\n\
'
    return tip

def upload_img(request):
    #return HttpResponse(request.POST)
    if request.method == 'POST':
        print request.POST
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            f = request.FILES["image"]
            fn = request.POST["newname"]
            parser = ImageFile.Parser()
            for chunk in f.chunks():
                parser.feed(chunk)
            img = parser.close()
            img.save(settings.BASE_DIR + '/demo/images_pool/' + fn)
            html = head_html + select_weight + get_tip(fn) + tail_html 
            return HttpResponse(html)
            #return HttpResponse('Image upload success!')
    return HttpResponse('Allowed only via POST')