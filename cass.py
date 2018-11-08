#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:15:12 2018

@author: wuxian
"""

import logging

log = logging.getLogger()


#from cassandra.cluster import Cluster
#from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster

KEYSPACE = "imgspace"

def createKeySpace():
    cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
    session = cluster.connect()

    log.info("Creating keyspace...")
    try:
        session.execute("""
            CREATE KEYSPACE %s
            WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
            """ % KEYSPACE)

        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)

        log.info("creating table...")
        session.execute("""
            CREATE TABLE mytable (
                time text,
                name text,
                result text,
                PRIMARY KEY (time, name)
            )
            """)
    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)

createKeySpace();

def insertData():
    cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
    session = cluster.connect()

    log.info("setting keyspace...")
    session.set_keyspace(KEYSPACE)

    prepared = session.prepare("""
    INSERT INTO mytable (time, name, result)
    VALUES (?, ?, ?)
    """)
    
    fname = 'flask.txt'
    with open(fname, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
    
    l = last_line.split(',')
    log.info("inserting data")
    session.execute(prepared.bind((l[0],l[1],l[2].rstrip("\n"))))
    
