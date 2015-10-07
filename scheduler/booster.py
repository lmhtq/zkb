# -*- coding: utf-8 -*-
import os
import glob
import commands
import sqlite3
import numpy as np
import string
import time
import codecs
import heapq
import shutil

config = {}
conn = {}
cursor = {}
feature_dict = {}
features = {}
image_dir = '../images/'
cache_dir = '../cache/'
database_dir = '../database/'

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

#check whether feature extractor exists
def check_feature_extractors_exist():
    global config
    for method in config["extraction_method"]:
        if not os.path.isfile(bins_dir + 'method'):
            print 'feature extractor: '+ method + ' not exists!'
            exit(-1)

#copy a database for reading
def database_cpoy(extraction_method):
    db = database_dir + extraction_method + '.db'
    if os.path.isfile(db):
        shutil.copy(db, db + '.read')
        return True
    else:
        return False

#block crawler
def block_crawler():
    if (os.path.exists(image_dir + 'lock')):
        print('Error: lock aleady in!\n')
    else:
        os.mknod(image_dir + 'lock')

#unblock_crawler
def unblock_crawler():
    if (os.path.exists(image_dir + 'lock')):
        os.remove('../iamges/lock')
    else:
        print('Warning: lock not in')
        pass

#init cache databases
def init_cache_db():
    global config, conn, cursor, feature_dict
    for method in config["extraction_method"]:
        conn[method] = sqlite3.connect(cache_dir + method + '_cache.db.write')
        cursor[method] = conn[method].cursor()
        cursor[method].execute('create table if not exists ' + method + \
        ' (name_id integer primary key AUTOINCREMENT, image_name varchar(1000) NOT NULL, image_name_sim varchar(50000) NOT NULL)')
        cursor[method].execute('select image_name from ' + method)
        logs = cursor[method].fetchall()
        for log in logs:
            try:
                feature_dict[method].add(log[0])
            except:
                feature_dict[method] = set()
                feature_dict[method].add(log[0]) 

#calc a image's most similarest 20 images
def cache_a_log(log, method):
    global features, config, conn, cursor
    topK = config["topK"]
    tn, tf = log[1], np.array(map(float, log[2].split(' ')))
    res_dict = []
    
    for v in features:
        name, feature = v[1], np.array(map(float, v[2].split(' ')))
        value = np.sqrt(sum((tf - feature)**2))
        res_dict.append({"name":name, "value":value})

    #print res_dict
    res_list = []
    tops = heapq.nsmallest(topK, res_dict, key=lambda s:s['value'])
    for v in tops:
        #print v, 'heapq'
        res_list.append(v["name"])
    
    res_str = ' '.join(res_list) 
    cursor[method].execute('insert into ' + method + '(image_name, image_name_sim)' + ' values (?,?)', (tn, res_str))
    conn[method].commit()
    
    #for debug
    #exit(-1)

#read a database and cache its result
def cache_db(method):
    global features
    conn = sqlite3.connect(database_dir + method + '.db.read')
    cursor = conn.cursor()
    cursor.execute('select * from ' + method)
    features = cursor.fetchall()
    
    for log in features:
        cache_a_log(log, method)

    shutil.move(cache_dir + method + '_cache.db.write', cache_dir + method + '_cache.db')
    print 'database ' + method + 'cached! ' + str(len(features)) + ' images!'

#loop to generate cache databases
def loop():
    global config, conn
    while True:
        time.sleep(config["sleep"])
        init_cache_db()
        
        for method in config["extraction_method"]:
            database_cpoy(method)
            cache_db(method)
            conn[method].close()


if __name__ == '__main__' :
    #global conn, cursor, config
    print "main"
    parse_config()
    loop()
    # try:
    #     loop()
    # except:
    #     for method in config["extraction_method"]:
    #         cursor[method].close()
    #         conn[method].close()
    #     print 'Someting happend, exit with databases close correctly.'
    