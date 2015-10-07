# -*- coding: utf-8 -*-
import os
import glob
import commands
import sqlite3
import numpy as np
import string
import time
import codecs
import shutil

config = {}
image_dir = '../images/'
images_pool = '../images_pool/'

#read configuration
def parse_config():
    global config
    fd = open('config.cfg')
    lines = fd.readlines()
    for line in lines :
        if line[0] == '#' or line.strip() == '':
            continue
        tmp = line.split(':')
        try:            
            config[tmp[0].strip()].append(tmp[1].strip())
        except:
            config[tmp[0].strip()] = []
            config[tmp[0].strip()].append(tmp[1].strip())
    config["batch_size"] = int(config["batch_size"][0])
    config["sleep"] = float(config["sleep"][0])
    config["topK"] = int(config["topK"][0])
    #print config

#crawler ready
def crawler_ready():
    if (os.path.exists(image_dir + 'ready')):
        print('Warning: Crawler has put images ready!\n')
    else:
        os.mknod(image_dir + 'ready')
        print('Crawler has put images ready!\n')

#crawler is not ready
def crawler_unready():
    if (os.path.exists(image_dir + 'ready')):
        os.remove(image_dir + 'ready')
    else:
        print('Warning: ready file already deleted!')
        pass

#check executor has finished
def check_block_exist():
    if (os.path.isfile(image_dir + 'lock')) or \
    (os.path.isfile(image_dir + 'ready')):
        return True
    else:
        return False

#clean
def clean_image_dir():
    print 'Warning: DIR\'' + image_dir + '\' will be clean!'
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.mkdir(image_dir)

#loop to extract featrues and put them into databases
def loop():
    global config
    types = config["type"]
    
    while True:
        time.sleep(config["sleep"])
        if check_block_exist():
            continue

        clean_image_dir()
        files_grabbed = []
        for files in types:
            files_grabbed.extend(glob.glob(images_pool + '*.' +files))
        
        if (len(files_grabbed) < config["batch_size"]):
            continue
        
        for i in range(len(files_grabbed)):
            fn = files_grabbed[i]
            if (i+1) % config["batch_size"] == 0:
                while check_block_exist():
                    time.sleep(config["sleep"])
                #print fn ,image_dir + fn.split('/')[-1]
                os.symlink(fn, image_dir + fn.split('/')[-1])
                print image_dir + fn.split('/')[-1] + ' has been put in ' + image_dir 
                crawler_ready()


if __name__ == '__main__' :
    #global conn, cursor, config
    print "main"
    parse_config()
    clean_image_dir()
    loop()
    # try:
    #     loop()
    # except:
    #     for method in config["extraction_method"]:
    #         cursor[method].close()
    #         conn[method].close()
    #     print 'Someting happend, exit with databases close correctly.'
    