#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
import rasterio as rio
import datetime
import numpy as np
import pandas as pd
import geowombat as gw
import geopandas as gpd

def get_center_coord(img):
    
    img0 = rio.open(img)
    xy = img0.xy(img0.height // 2, img0.width // 2)
    return xy
    
def get_ts_at_cent(pt_dir, spec_index, start_yr, end_yr):

    ts_stack = []
    ds_stack = []
    for img in os.listdir(pt_dir):
        if img.startswith(spec_index) and img.endswith('.tif'):
            ## ts images are named YYYYDDD with YYYY=year and DDD=doy
            img_date = pd.to_datetime(img.split('_')[1][:7],format='%Y%j')
            img_yr = int(img.split('_')[1][:4])
            img_doy = int(img.split('_')[1][4:7])
            if ((img_yr > start_yr) or (img_yr == start_yr and img_doy >= 150)) and (
                (img_yr < end_yr) or (img_yr == end_yr and img_doy <= 150)):
                ts_stack.append(os.path.join(pt_dir,img))
                ds_stack.append(img_date)

    xy = get_center_coord(ts_stack[0])

    with gw.open(ts_stack, time_names = ds_stack) as src:
        cent = src.sel(x=xy[0], y=xy[1], method='nearest').squeeze()

    tsda = pd.DataFrame(list(zip(cent.values,cent.time.values)), columns = ['val','dt'])
    tsda.set_index('dt', inplace=True)
    
    return tsda


def get_values_at_cent(img):
    
    xy = get_center_coord(img)
    with gw.open(img) as src:
        vals = src.sel(x=xy[0], y=xy[1], method='nearest').squeeze() 
    bands=[d[2:-2] for d in vals.descriptions]
    valdict = dict(zip(bands,vals.values))
    
    return valdict