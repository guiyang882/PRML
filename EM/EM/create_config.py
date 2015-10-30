#!/usr/bin/env python
# coding=utf-8

import json

def create_ConfigParameter():
    config = {}
    config["name"] = "Describe the distribution about the data set"
    config["num_class"] = 2
    config["parameters"] = {}

    params = {}
    params["delta"] = []
    for i in range(0,config["num_class"]-1):
        params["delta"].append(1.0/config["num_class"])
    params["delta"].append(1-sum(params["delta"]))
    
    params["termination"] = 100

    params["data"] = []
    part1 = {}
    part1["mu"] = [0,0]
    part1["sigma"] = [[1,0],[0,1]]
    
    part2 = {}
    part2["mu"] = [1,1]
    part2["sigma"] = [[1,0],[0,1]]
    params["data"] = [part1,part2]

    config["parameters"] = params
    dump_obj = json.dumps(config,indent=4)
    fout = open("em_config.json","w")
    fout.write(dump_obj)
    fout.close()

create_ConfigParameter()
