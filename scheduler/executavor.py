# -*- coding: utf-8 -*-
import os
import glob
import commands
import sqlite3
import numpy as np
import string
import time
import codecs

config = {}
conn = {}
cursor = {}
feature_dict = {}
image_dir = '../images/'
bins_dir = '../feature_extractor/bins/'
database_dir = '../database'

#read configuration
def parse_config():
    global config, feature_dict
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
    for method in config["extraction_method"]:
        feature_dict[method] = set()
    #print config

#check whether feature extractor exists
def check_feature_extractors_exist():
    global config
    for method in config["extraction_method"]:
        if not os.path.isfile(bins_dir + 'method'):
            print 'feature extractor: '+ method + ' not exists!'
            exit(-1)

#check whether crawler has put new images in
def producer_ready():
    if os.path.isfile(image_dir + 'ready'):
        return True
    else:
        return False

#block crawler
def block_crawler():
    if (os.path.isfile(image_dir + 'lock')):
        print('Error: lock aleady in!\n')
    else:
        os.mknod(image_dir + 'lock')

#unblock_crawler
def unblock_crawler():
    if (os.path.isfile(image_dir + 'lock')):
        os.remove(image_dir + 'lock')
        if (os.path.isfile(image_dir + 'ready')):
            os.remove(image_dir + 'ready')
    else:
        print('Warning: lock not in')
        pass

#init databases
def init_db():
    global config, conn, cursor, feature_dict
    for method in config["extraction_method"]:
        conn[method] = sqlite3.connect(database_dir + method + '.db')
        cursor[method] = conn[method].cursor()
        cursor[method].execute('create table if not exists ' + method + \
        ' (name_id integer primary key AUTOINCREMENT, image_name varchar(1000) NOT NULL, feature varchar(50000) NOT NULL)')
        cursor[method].execute('select image_name from ' + method)
        logs = cursor[method].fetchall()
        for log in logs:
            try:
                feature_dict[method].add(log[0])
            except:
                feature_dict[method] = set()
                feature_dict[method].add(log[0]) 

#get feature of a image
def get_feature(image_path, extraction_method):
    cmd = bins_dir + extraction_method + ' ' + image_path
    cmd_stat, cmd_result = commands.getstatusoutput(cmd)
    if not cmd_stat:
        return cmd_result[:-1]

#transfer strings to numpy.array
def str_to_np(cmd_result):
    return np.array(map(float, cmd_result.split()))

#loop to extract featrues and put them into databases
def loop():
    global config, conn, cursor
    types = config["type"]
    temp_list = {}
    extraction_method = config["extraction_method"]

    while True:
        time.sleep(config["sleep"])
        files_grabbed = []
        if producer_ready():
            #tell crawler that he CAN NOT put new iamges in image_dir
            block_crawler()

            for files in types:
                files_grabbed.extend(glob.glob(image_dir + '*.' +files))
            
            #debug: the images to be process
            #print files_grabbed
            #return 1
            print extraction_method
            for method in extraction_method:
                print method
                for image_name in files_grabbed:
                    if image_name not in feature_dict[method]:
                        feature = get_feature(image_name, method)
                        try:
                            temp_list[method].append((image_name, feature))
                            feature_dict[method].add(image_name)
                        except:
                            temp_list[method] = []
                            temp_list[method].append((image_name, feature))
                            feature_dict[method].add(image_name)
                    else:
                        print image_name + ' already in ' + method + ' database'
                
                #check whether temp_list[method] is not in
                try:
                    temp_list[method]
                except:
                    continue

                if (len(temp_list[method]) >= config["batch_size"]):
                    cursor[method].executemany('insert into ' + method + '(image_name, feature)' + ' values (?,?)', temp_list[method])
                    conn[method].commit()
                    print 'Update \'' + str(len(temp_list[method])) + '\' features in \'' + method + '\' database' 
                    temp_list[method] = []
            
            for method in extraction_method:
                try:
                    if len(temp_list[method]) < config["batch_size"]:
                        continue
                except:
                    pass
            
            #tell crawler that he CAN put new iamges in image_dir
            unblock_crawler()
        #debug: no loop
        #return 1



if __name__ == '__main__' :
    #global conn, cursor, config
    print "main"
    parse_config()
    print producer_ready()
    #block_crawler()
    init_db()
    loop()
     try:
    #     loop()
    # except:
    #     for method in config["extraction_method"]:
    #         cursor[method].close()
    #         conn[method].close()
    #     print 'Someting happend, exit with databases close correctly.'
    