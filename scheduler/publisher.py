# -*- coding: utf-8 -*-
import os
import glob
import commands
import sqlite3
import numpy as np
import string
import time
import codecs
import sys
import heapq

config = {}
image_dir = '../images/'
image_dir_in_db = '../images/'
cache_dir = '../cache/'
database_dir = '../database/'
bins_dir = '../feature_extractor/bins/'

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

#read configuration
def parse_config_path(path):
    global config
    fd = open(path)
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

#whether cache has the image to retrieve
def cache_hit(img_name, method):
    conn = sqlite3.connect(cache_dir + method + '_cache.db')
    cursor = conn.cursor()
    img_name = image_dir_in_db + img_name.split('/')[-1]
    sql = 'select image_name_sim from ' + method + ' where image_name = \'' + img_name + '\''
    cursor.execute(sql)
    res = cursor.fetchone()
    cursor.close()
    conn.close()
    if res == None:
        return (False, None)
    else:
        res_list = res[0].split(' ')
        return (True, res_list)

#get feature of a image
def get_feature(image_path, extraction_method):
    cmd = bins_dir + extraction_method + ' ' + image_path
    cmd_stat, cmd_result = commands.getstatusoutput(cmd)
    if not cmd_stat:
        return cmd_result[:-1]

#retrieve from database
def retrieve_from_database(img_name, method):
    global config
    
    feature = get_feature(img_name, method)
    #print feature
    tf = np.array(map(float, feature.split(' ')))
    #print tf
    conn = sqlite3.connect(database_dir + method + '.db')
    cursor = conn.cursor()
    sql = 'select * from ' + method
    cursor.execute(sql)
    allfeatures = cursor.fetchall()
    cursor.close()
    conn.close()
    
    res_dict = []
    for df in allfeatures:
        na, fea = df[1], np.array(map(float, feature.split(' ')))
        val = np.sqrt(sum((tf-fea)**2))
        res_dict.append({"name":na, "value":val})

    #print res_dict
    topK = config["topK"]
    res_list = []
    tops = heapq.nsmallest(topK, res_dict, key=lambda s:s['value'])
    for v in tops:
        #print v, 'heapq'
        res_list.append(v["name"])
    
    return res_list

#retrieve a image, return the most similarest 20 images
def retrieve(img_name, method):
    hit, imgs = cache_hit(img_name, method)
    if hit:
        print "Hit!"
        #print imgs
    else:
        print "Miss! Retrieving from database."
        imgs = retrieve_from_database(img_name, method)

    return imgs

if __name__ == "__main__":
    #print sys.argv[2]
    img_name = sys.argv[1]
    #method = sys.argv[2]
    method = 'color_histogram'
    parse_config()
    try:
        print retrieve(img_name, method)
    except:
        print ''

if __name__ == "scheduler.publisher":
    parse_config_path('../scheduler/config.cfg')
    # image_dir = '../../images/'
    # cache_dir = '../../cache/'
    # database_dir = '../../database/'
    # bins_dir = '../../feature_extractor/bins/'
    
