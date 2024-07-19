#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
import datetime
import rasterio as rio
import numpy as np
import geowombat as gw
import xarray as xr
import pandas as pd
import bottleneck

def add_var_to_stack(arr, var, attrs, out_dir, band_names, ras_list, **gw_args):
    
    ras = os.path.join(out_dir,f'{var}.tif')
    #if os.path.exists(ras) == False:  
    arr.attrs = attrs
    print(f'making {var} raster') 
    arr.gw.to_raster(ras,**gw_args)
    ras_list.append(ras)
    band = str(var.split('_')[0:1])
    band_names.append(band)

def find_peak_simp(ts_stack,ds_stack):
    
    with gw.open(ts_stack, time_names = ds_stack) as src:
        highs = src.where(src >= 3000)
    #highs_masked = highs.where(src < 1000).all("time")
    peaksimp = highs.idxmax(dim='time',skipna=True)

    return peaksimp

def revise_peaks(t,vals,invert):
    i = -1 if invert == True else 1
    fill = 32767 if invert == True else 0
    f = 10000 if invert == True else 1
    
    if t == 1:
        peakcheck1 = vals.where(((i*(vals - vals.shift(time=-1).fillna(fill)) >= 0) | (vals==f)),fill).fillna(fill)
        peakcheck = peakcheck1.where(((i*(peakcheck1 - peakcheck1.shift(time=1).fillna(fill)) >= 0) |
                                      (peakcheck1==f)),fill).fillna(fill)
    else:
        peakcheck1 = vals.where(((i*(vals - vals.shift(time=-t).fillna(fill)) >= 0 ) | 
                                 (vals.shift(time=-(t-1)).fillna(fill) == f) | (vals==f)),fill).fillna(fill)
        peakcheck = peakcheck1.where(((i*(peakcheck1 - peakcheck1.shift(time=t).fillna(fill)) >= 0 ) | (peakcheck1.shift(time=(t-1)).fillna(fill) == f) | (peakcheck1==f)),fill).fillna(fill)
    
    return peakcheck

def find_peaks_robust(ts_stack, ds_stack,peak_thresh,base_thresh,invert):
    with gw.open(ts_stack, time_names = ds_stack) as valsin:
        attrs = valsin.attrs.copy()
    ''' if invert == True, returns troughs
        returns number of peaks and day and value of first and last peak (/trough) for stack
    '''
    
    ## base the input data, shuch that all values <= base threshold are flagged as 1 (these separate potential peaks)
    #i = -1 if invert == True else 1
    if invert == False:
        peakbase = valsin.where(valsin > base_thresh, 1)
        ## check whether each pixel is local max value by subtracting the next in sequence (both forward and backward) and retaining
        ##    only those with positive results. if the original pixel value was 1, it is retainied as 1 to flag troughs separating peaks
    
        peakcheck1 = peakbase.where(((peakbase - peakbase.shift(time=-1).fillna(0) >= 0) | (peakbase==1)),0).fillna(0)
        peakcheck = peakcheck1.where(((peakcheck1 - peakcheck1.shift(time=1).fillna(0) >= 0) | (peakcheck1==1)),0).fillna(0)
        ## repeat the process at increasing time steps. If the value before the next time step is 1, there is a trough separating
        ##    the two peaks and both should be retained.
        for t in range(2,10):
            peaktcheck = revise_peaks(t,peakcheck,invert=False)
         ## TODO: break ties
    
        true_peaks = peakcheck.where(peakcheck >= peak_thresh)
    
    else:
        peakbase = valsin.where(valsin < base_thresh, 10000)
        peakcheck1 = peakbase.where(((peakbase - peakbase.shift(time=-1).fillna(32767) <= 0) | (peakbase==10000)),32767).fillna(32767)
        peakcheck = peakcheck1.where(((peakcheck1 - peakcheck1.shift(time=1).fillna(32767) <= 0) | (peakcheck1==10000)),32767).fillna(32767)
        for t in range(2,10):
            peaktcheck = revise_peaks(t,peakcheck,invert=True)
        true_peaks = peakcheck.where(peakcheck <= peak_thresh)
         
    numpeaks = true_peaks.count(dim='band').sum(dim='time').fillna(0).astype('int16')
    ## get first peak (minimum time of valid peaks). First fill nas with last possible day because skipna doesn't work with datetime 
    peak0d = true_peaks["time"].where(~true_peaks.isnull()).fillna(true_peaks.time[-1].values).min(dim="time")
    peak0v = valsin.sel(time=peak0d, method='nearest').astype('int16')
    ## get last trough (maximum time of valid troughs). First fill nas with first possible day because skipna doesn't work with datetime
    peak9d = true_peaks["time"].where(~true_peaks.isnull()).fillna(true_peaks.time[1].values).max(dim="time")
    peak9v = valsin.sel(time=peak9d, method='nearest').astype('int16').squeeze()
    
    return numpeaks, peak0d, peak0v, peak9d, peak9v


def find_peaks_deriv(ts_stack,ds_stack,band_names,peak_thresh,base_thresh,ras_list,out_dir, **gw_args):
    '''
    method to find the first peak of a season (in case more than one peaks occur)
    '''
    with gw.open(ts_stack, time_names = ds_stack) as src1:
        attrs = src1.attrs.copy()
    src_c = src1.chunk({"time": -1})
    
    deriv = src_c.differentiate("time")
    maxima = src1.where((deriv < 0) & (deriv.shift(time=1) > 0) & (src1 >= peak_thresh))
    minima = src1.where((deriv > 0) & (deriv.shift(time=1) < 0) & (src1 <= base_thresh))
    minmax_arr = xr.concat((minima,maxima),'band').mean(dim='band')
    minmaxf = minmax_arr.ffill(dim='time')
    minmaxff = minmaxf.bfill(dim='time')
    minmax_step = minmaxff - minmaxff.shift(time=1) 
    #falsepeaks = max_arr.where((minmax_step > 0) & (minmax_step < 2000))
    #falsepeak1 = falsepeaks.idxmax(dim="time",skipna=True)
    #minmax_arr.loc[dict(time=falsepeak1)] = 'nan' # this doesn't work because lost one timestep.
    true_peaks = maxima.where((np.abs(minmax_step) >= (peak_thresh - base_thresh)) | (minmax_step == 0))
    numpeaks = true_peaks.count(dim='band').sum(dim='time').fillna(0).astype('int16')
    ## get minimum time where peak condition is met 
    ##    (note need to fill nan with last time stamp because skipna doens't work with datetime64 -- will always give 1st date)
    peakds = true_peaks["time"].where(~true_peaks.isnull()).fillna(true_peaks.time[-1].values).min(dim="time")
    ## remask where no peaks were found
    mask = true_peaks.isnull().all("time")
    peak1d = peakds.where(~mask)
    ## get peak the simple way (max value above threshold) in case slopes don't create proper maxima/minima
    altpeakd = find_peak_simp(ts_stack,ds_stack)
    altpeaknum = altpeakd.count(dim='band').fillna(0).astype('int16')
    ## if a peak was found with the first method, give that, otherwise give simple peak
    peakout = peak1d.where(peak1d.notnull(), altpeakd)
    numpeaks_all = numpeaks.where(numpeaks > 0, altpeaknum).fillna(0).astype('int16')
    masked2 = src1.where(src1 < base_thresh).all("time")
    numpeaks_allm = numpeaks_all.where(~masked2, 0).fillna(0).astype('int16')

    return peakout, numpeaks_all
    
def get_greenup(ts_stack, ds_stack, peak_time, method='step'):
    with gw.open(ts_stack, time_names = ds_stack) as src:
        attrs = src.attrs.copy()
    prepeak1 = src.where(src['time'] < peak_time)
    if method == 'step':
        endslope = prepeak1.where(prepeak1 - prepeak1.shift(time=1) < 0)
        mask = endslope.isnull().all("time")
        sos0 = endslope['time'].where(~endslope.isnull()).fillna(prepeak1.time[-1].values).min(dim='time')
        sos=sos0.where(~mask, prepeak1.time[0].values)
    elif method == 'thresh':
        thresh = 2000
        endslope = prepeak1.where(prepeak1 < thresh)
        mask = endslope.isnull().all("time")
        sos0 = endslope['time'].where(~endslope.isnull()).fillna(prepeak1.time[0].values).max(dim='time')
        sos=sos0.where(~mask, prepeak1.time[0].values)
    elif method == 'deriv':
        prechunk = prepeak1.chunk({"time": -1})
        green_deriv = prechnuk.differentiate("time")
        pos_green_deriv = green_deriv.where(green_deriv > 0)
        pos_greenup = prechunk.where(~np.isnan(pos_green_deriv))
        med_g = pos_greenup.median("time")
        dist = np.abs(pos_greenup - med_g)
        mask = dist.isnull().all("time")
        distfill = dist.fillna(dist.max() + 1)
        sos0 = distfill.idxmin(dim="time",skipna=True).where(~mask)
    sosv = prepeak1.sel(time=sos, method='nearest').astype('int16')
    #return ts_stack, peak_time, prepeak1, endslope, mask, sos0, sos, sosv
    return sos, sosv

def get_senescence(ts_stack, ds_stack, peak_time, method='step'):
    with gw.open(ts_stack, time_names = ds_stack) as src:
        attrs = src.attrs.copy()
    postpeak1 = src.where(src['time'] > peak_time)
    if method == 'step':
        endslope = postpeak1.where(postpeak1 - postpeak1.shift(time=1) > 0)
        #mask = endslope.isnull().all("time")
        eos = endslope['time'].where(~endslope.isnull()).fillna(postpeak1.time[-1].values).min(dim="time")
    elif method == 'thresh':
        thresh = 2000
        endslope = postpeak1.where(postpeak1 < thresh)
        #mask = endslope.isnull().all("time")
        eos = endslope['time'].where(~endslope.isnull()).fillna(postpeak1.time[-1].values).min(dim='time')
    elif method == 'deriv':
        postchunk = postpeak1.chunk({"time": -1})
        brown_deriv = postchnuk.differentiate("time")
        neg_brown_deriv = brown_deriv.where(brown_deriv < 0)
        neg_brownup = postchunk.where(~np.isnan(neg_brown_deriv))
        med_b = neg_brown.median("time")
        dist = np.abs(neg_brown - med_b)
        mask = dist.isnull().all("time")
        distfill = dist.fillna(dist.max() + 1)
        eos0 = distfill.idxmin(dim="time",skipna=True).where(~mask)
    eosv = postpeak1.sel(time=eos, method='nearest').astype('int16')
    return eos, eosv
            
def prep_pheno_bands(pheno_vars,ts_stack,ds_stack,ts_stack_padded, ds_stack_padded, out_dir,start_yr, 
                     temp,start_doy,sigdif,band_names,ras_list,**gw_args):
   
    with gw.open(ts_stack, time_names = ds_stack) as src:
        attrs = src.attrs.copy()
    
    if f'maxv_{temp}' in pheno_vars or f'amp_{temp}' in pheno_vars:
        mmax = src.max(dim='time').astype('int16')
        add_var_to_stack(mmax,f'maxv_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args) 
    if f'minv_{temp}' in pheno_vars or f'amp_{temp}' in pheno_vars:
        mmin = src.min(dim='time').astype('int16')
        add_var_to_stack(mmin,f'minv_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args) 
    if f'amp_{temp}' in pheno_vars:
        aamp =  (mmax - mmin)
        add_var_to_stack(aamp,f'amp_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
        
    mmed = src.median(dim='time').astype('int16')
    if f'med_{temp}' in pheno_vars:
        add_var_to_stack(mmed,f'med_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
    
    if f'slp_{temp}' in pheno_vars:
        slp_path = os.path.join(out_dir,f'slp_{temp}_{start_yr}.tif')
        #if os.path.exists(slp_path) == False:
        src.coords['ordinal_day'] = (('time', ), (src.time - src.time.min()).values.astype('timedelta64[D]').astype(int))
        vslope = src.swap_dims({'time': 'ordinal_day'}).polyfit('ordinal_day', 
                                                                deg=1,skipna=True).polyfit_coefficients[0].astype('int16')    
        #else:
        #    vslope = slp_path
        add_var_to_stack(vslope,f'slp_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
    
    peaks = find_peaks_robust(ts_stack,ds_stack, mmed+int(sigdif), mmed, invert=False)
    ## peaks returns number of peaks, date-of-1st-peak, val-of-first-peak, date-of-last-peak, val-of-last-peak
   
    if f'numrot_{temp}' in pheno_vars:
        add_var_to_stack(peaks[0],f'numrot_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
    
    if f'posd_{temp}' in pheno_vars:
        posd_path = os.path.join(out_dir,f'posd_{temp}_{start_yr}.tif')
        #if os.path.exists(posd_path) == False:
        peak1s = peaks[1].where(peaks[0] > 0).squeeze().dt.dayofyear  # returns na if no peaks (to be filled with 0)
        ## add 365 to doy if it passed into the next year to avoid jump in values from Dec31 to Jan 1
        ## (keep everything >= start_doy (-pad) as is. If passes into next year, doy will be < start_doy, so add 365)
        posd2 = peak1s.where(peak1s >= start_doy, (peak1s + 365)).fillna(0).astype('int16')     
        #else:
        #    posd2 = posd_path
        add_var_to_stack(posd2,f'posd_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
    
    posv = peaks[2].where(peaks[0] > 0, mmed).astype('int16').squeeze() 
    if f'posv_{temp}' in pheno_vars:                       
        add_var_to_stack(posv,f'posv_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)        
        
    if any(v in pheno_vars for v in [f'tosd_{temp}', f'tosv_{temp}',f'numlow_{temp}',f'p1amp_{temp}']):
        tosd_path = os.path.join(out_dir,f'tosd_{temp}_{start_yr}.tif')
        tosv_path = os.path.join(out_dir,f'tosv_{temp}_{start_yr}.tif')
        numlow_path = os.path.join(out_dir,f'numlow_{temp}_{start_yr}.tif')
        #if os.path.exists(tosd_path) == False or os.path.exists(tosv_path) == False or os.path.exists(numlow_path) == False:
        troughs = find_peaks_robust(ts_stack,ds_stack, mmed-int(sigdif), mmed, invert=True)
        tosd1 = troughs[3].where(troughs[0] > 0).squeeze().dt.dayofyear
        ## add 365 to doy if it passed into the next year to avoid jump in values from Dec31 to Jan 1
        ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
        tosd2 = tosd1.where(tosd1 >= start_doy, (tosd1 + 365)).fillna(0).astype('int16')
        tosv = peaks[4].where(peaks[0] > 0, mmed).astype('int16') 
        numlow = troughs[0]
        #else: 
        #    tosd2 = tosd2_path
        #    tosv = tosv_path
        #    numlow = numlow_path
        if f'numlow_{temp}' in pheno_vars:
            add_var_to_stack(numlow,f'numlow_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
        if f'tosd_{temp}' in pheno_vars:
            add_var_to_stack(tosd2,f'tosd_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
        if f'tosv_{temp}' in pheno_vars:      
            add_var_to_stack(tosv,f'tosv_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
        if f'p1amp_{temp}' in pheno_vars:                 
            p1amp00 = (posv - tosv).where(peaks[0] > 0, (mmed - tosv))  
            p1amp0 = p1amp00.where(numlow > 0, (posv - mmed))
            p1amp = p1amp0.where(((p1amp0 > 0) & ((peaks[0] > 0) | (numlow > 0))), 0).fillna(0).astype('int16').squeeze() 
            add_var_to_stack(p1amp,f'p1amp_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)

    if any(v in pheno_vars for v in [f'sosd_{temp}',f'sosv_{temp}',f'rog_{temp}', f'los_{temp}']):
        sosv_path = os.path.join(out_dir,f'sosv_{temp}_{start_yr}.tif')
        sosd_path = os.path.join(out_dir,f'sosd_{temp}_{start_yr}.tif')
        #if os.path.exists(sosd_path) == False or os.path.exists(sosv_path) == False:            
        sos, sosv = get_greenup(ts_stack_padded, ds_stack_padded, peaks[1], method='thresh')
        sosd = sos.dt.dayofyear
        ## add 365 to doy if it passes into the next year to avoid jump in values from Dec31 to Jan 1
        ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
        sosd1 = sosd.where(sosd >= start_doy -40, sosd + 365)
        sosd2 = sosd1.where(peaks[0] > 0, 0).astype('int16')   
        #else:
        #    sosd2 = sosd_path
        #    sosv = sosv_path
        if f'sosd_{temp}' in pheno_vars :
            add_var_to_stack(sosd2,f'sosd_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)       
        if f'sosv_{temp}' in pheno_vars:
            add_var_to_stack(sosv,f'sosv_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
    if any(v in pheno_vars for v in [f'eosd_{temp}',f'eosv_{temp}',f'ros_{temp}', f'los_{temp}']):
        eosv_path = os.path.join(out_dir,f'eosv_{temp}_{start_yr}.tif')
        eosd_path = os.path.join(out_dir,f'eosd_{temp}_{start_yr}.tif')
        #if os.path.exists(eosd_path) == False or os.path.exists(eosv_path) == False:
        eos, eosv = get_senescence(ts_stack_padded, ds_stack_padded, peaks[1], method='thresh')
        eosd = eos.dt.dayofyear
        ## add 365 to doy if it passes into the next year to avoid jump in values from Dec31 to Jan 1
        ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
        eosd1 = eosd.where(eosd >= start_doy, eosd + 365)
        eosd2 = eosd1.where(peaks[0] > 0, 0).astype('int16')  
        #else:
        #    eosd2 = eosd_path
        #    eosv = eosv_path
        if f'eosd_{temp}' in pheno_vars:
            add_var_to_stack(eosd2,f'eosd_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
        if f'eosv_{temp}' in pheno_vars:
            add_var_to_stack(eosv,f'eosv_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
 
    if f'rog_{temp}' in pheno_vars or f'ros_{temp}' in pheno_vars:
        with gw.open(os.path.join(out_dir,f'posd_{temp}_{start_yr}.tif')) as posd2:
            pass
        if f'rog_{temp}' in pheno_vars or f'los_{temp}' in pheno_vars:
            with gw.open(os.path.join(out_dir,f'sosd_{temp}_{start_yr}.tif')) as sosd2:
                with gw.open(os.path.join(out_dir,f'sosv_{temp}_{start_yr}.tif')) as sosv:
                    rog = sosd2.where(sosd2 == 0, (posv - sosv) / (posd2 - sosd2))
                    rog = rog.astype('int16')
            add_var_to_stack(rog,f'rog_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
        if f'ros_{temp}' in pheno_vars or f'ros_{temp}' in pheno_vars:
            with gw.open(os.path.join(out_dir,f'eosd_{temp}_{start_yr}.tif')) as eosd2:      
                with gw.open(os.path.join(out_dir,f'eosv_{temp}_{start_yr}.tif')) as eosv:
                    ros = eosd2.where(eosd2 == 0, (posv - eosv) / (posd2 - eosd2)) 
                    ros = ros.astype('int16')
            add_var_to_stack(ros,f'ros_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)
        if f'los_{temp}' in pheno_vars:
            with gw.open(os.path.join(out_dir,f'sosd_{temp}_{start_yr}.tif')) as sosd2:
                with gw.open(os.path.join(out_dir,f'eosd_{temp}_{start_yr}.tif')) as eosd2:
                    los = eosd2 - sosd2
            add_var_to_stack(los,f'los_{temp}_{start_yr}',attrs,out_dir,band_names,ras_list,**gw_args)

    return peaks

def make_pheno_vars(img_dir, out_dir, start_yr, start_mo, spec_index, pheno_vars, sigdif, pad_days):
    '''
    The code expects that the wet season is the primary season of interest and the {start_mo} is the first month of 
    the wet season of interest. If the season of interest spans two calendar years, the first year is the {start_year}.
    The other season is the full season prior to the start year. For now this divides into two seasons, each of six months 
    The {pad_days} can be set to allow for some overlap between seasons when fitting curve tails).
        {pad_days} is a list [l,r] with l as padding into left side and r as padding into right side.
    '''
    if not os.path.exists(out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print('made new directory: {}'.format(out_dir))

    ## get stack from images in time-series directory that match year and month
    ts_stack_wet = []
    ts_stack_dry = []
    ds_stack_wet = []
    ds_stack_dry = []
    ts_stack_wet_padded = []
    ds_stack_wet_padded = []
    ts_stack_dry_padded = []
    ds_stack_dry_padded = []
            
    pt_name = os.path.basename(img_dir).split('_')[0]
    start_doy = int (30.5 * start_mo) - 30
    
    if pad_days is None or pad_days == " ":
        pad_days = [0,0]
    else:
        print('padded will add {} days on left and {} days on right \n'.format(pad_days[0],pad_days[1]))
    print('start doy is: {}, padded start_doy is {}'.format(start_doy, start_doy - pad_days[1]))
    for img in sorted(os.listdir(img_dir)):
        if img.split('_')[0] == spec_index and img.endswith('.tif'):
            ## ts images are named YYYYDDD with YYYY=year and DDD=doy
            img_date = pd.to_datetime(img.split('_')[1][:7],format='%Y%j')
            ## use year to filter to mapping year and doy to parse seasons regardless of year
            img_yr = int(img.split('_')[1][:4])
            img_doy = int(img.split('_')[1][4:7])
            if img_yr == int(start_yr):
                if (img_doy >= (start_doy - 183) - pad_days[1]) & (img_doy < (start_doy - 183)):
                    ts_stack_dry_padded.append(os.path.join(img_dir,img))
                    ds_stack_dry_padded.append(img_date)
                elif (img_doy >= (start_doy - 183)) and (img_doy < start_doy - pad_days[0]):
                    ts_stack_dry.append(os.path.join(img_dir,img))
                    ds_stack_dry.append(img_date)
                    ts_stack_dry_padded.append(os.path.join(img_dir,img))
                    ds_stack_dry_padded.append(img_date)
                elif (img_doy >= start_doy - pad_days[0]) and (img_doy < start_doy):
                    ts_stack_dry.append(os.path.join(img_dir,img))
                    ds_stack_dry.append(img_date)
                    ts_stack_dry_padded.append(os.path.join(img_dir,img))
                    ds_stack_dry_padded.append(img_date)      
                    ts_stack_wet_padded.append(os.path.join(img_dir,img))
                    ds_stack_wet_padded.append(img_date)         
                elif (img_doy >= start_doy) & (img_doy < start_doy + pad_days[0]):
                    ts_stack_wet.append(os.path.join(img_dir,img))
                    ds_stack_wet.append(img_date)
                    ts_stack_dry_padded.append(os.path.join(img_dir,img))
                    ds_stack_dry_padded.append(img_date)      
                    ts_stack_wet_padded.append(os.path.join(img_dir,img))
                    ds_stack_wet_padded.append(img_date)   
                elif (img_doy >= start_doy + pad_days[0]):        
                    ts_stack_wet.append(os.path.join(img_dir,img))
                    ds_stack_wet.append(img_date)
                    ts_stack_wet_padded.append(os.path.join(img_dir,img))
                    ds_stack_wet_padded.append(img_date)   
            elif img_yr == int(start_yr) + 1:     
                if (img_doy <= (start_doy - 183)):
                    ts_stack_wet.append(os.path.join(img_dir,img))
                    ds_stack_wet.append(img_date)
                    ts_stack_wet_padded.append(os.path.join(img_dir,img))
                    ds_stack_wet_padded.append(img_date)   
                elif img_doy < (start_doy - 183) + pad_days[1] :    
                    ts_stack_wet_padded.append(os.path.join(img_dir,img))
                    ds_stack_wet_padded.append(img_date)   
                    
    ts_stack = list(set(ts_stack_dry_padded + ts_stack_wet_padded))
    ts_stack.sort()
    ds_stack = list(set(ds_stack_dry_padded + ds_stack_wet_padded))
    ds_stack.sort()
    ras_list = []
    band_names = []
                
    gw_args = {'verbose':0,'n_workers':1,'n_threads':1,'n_chunks':254, 'gdal_cache':64,'overwrite':True}
    
    ## Convert image stack to XArray using geowombat:
    yr_bands = [b for b in pheno_vars if "_" not in b or b.split("_")[1] == 'yr']
    if len(yr_bands) > 0:
        print('getting full year variables')
        sys.stderr.write('working on yr_band {} \n'.format(yr_bands))
        pheno_comp = prep_pheno_bands(pheno_vars, ts_stack, ds_stack, None, None, 
                                      out_dir,start_yr,'yr',start_doy, sigdif, band_names, ras_list, **gw_args)
        
    
    wet_bands = [b for b in pheno_vars if "_" in b and b.split("_")[1] == 'wet']
    if len(wet_bands) > 0:
        print('getting wet season variables')
        print(ts_stack_wet)
        print(ds_stack_wet)
        pheno_comp = prep_pheno_bands(pheno_vars, ts_stack_wet, ds_stack_wet, ts_stack_wet_padded, ds_stack_wet_padded,
                                      out_dir,start_yr,'wet', start_doy, sigdif, band_names, ras_list, **gw_args)
        
        sys.stderr.write('writing stack for wet pheno_vars:{}'.format(band_names))
        if len(ras_list)<len(pheno_vars):
            sys.stderr.write('oops--got an unknown band')
         
        else:
            ##Start writing output composite
            with rio.open(ras_list[0]) as src0:
                meta = src0.meta
                meta.update(count = len(ras_list))
            
            # Read each layer and write it to stack
        
            out_ras = os.path.join(out_dir,'{}_{}_{}_Phen_wet.tif'.format(pt_name,start_yr,spec_index))

            with rio.open(out_ras, 'w', **meta) as dst:
                for id, layer in enumerate(ras_list, start=1):
                    with rio.open(layer) as src1:
                        dst.write(src1.read(1),id)
                dst.descriptions = tuple(band_names)
   
    dry_bands = [b for b in pheno_vars if "_" in b and b.split("_")[1] == 'dry']
    print('getting dry season variables')
    if len(dry_bands) > 0:
        pheno_comp = prep_pheno_bands(pheno_vars, ts_stack_dry, ds_stack_dry, ts_stack_dry_padded, ds_stack_dry_padded, 
                         out_dir,start_yr,'dry', start_doy, sigdif, band_names, ras_list, **gw_args)
    

    return out_ras
