#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 23:19:25 2018

@author: Xiaobo
"""
import logging, logging.config
from datetime import datetime
import json

class Logger(object):
    
    def __init__(self,log_dir,new=True):

        log_conf = log_dir+'logging.conf'
        if new:
            dt = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            log_file = log_dir+'results_'+dt+'.log'
            conf_dict = { 'version': 1,
             'formatters': {
                     'formatter1': {                
                             'class': 'logging.Formatter',
                             'format': '%(asctime)s %(message)s'
                                 }
                     },
                     
             'handlers': {
                     'file': {
                             'class':'logging.FileHandler',
                             'filename': log_file,
                             'formatter':'formatter1',                            
                             }
                     },
                     
            'loggers': {
                    },
        'root': {
            'level': 'DEBUG',
            'handlers': [ 'file']
        },
        }
            with open(log_conf,'w') as conf:
                json.dump(conf_dict,conf)
        else:
            with open(log_conf,'r') as conf:
                conf_dict = json.load(conf)
        logging.config.dictConfig(conf_dict)
        self.logger = logging.getLogger()
    
    def get_logger(self):
        return self.logger
    
    

