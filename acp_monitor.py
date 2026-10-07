"""USGS M>=5 -> NOAA ACP -> synchronized MP4 -> X. See README.md."""
import sys, subprocess, os, json, time, math, io, hashlib
from pathlib import Path
from requests_oauthlib import OAuth1
import requests, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio_ffmpeg
from matplotlib.animation import FFMpegWriter
from eccodes import codes_new_from_message, codes_get, codes_get_values, codes_release
from tqdm.auto import tqdm

# CONFIGURATION
PUBLISH_TO_X = os.environ.get('PUBLISH_TO_X','true').lower() == 'true'
FPS = 5                      # data scenes per second; 720 scenes => 144 seconds
LAT_RADIUS, LON_RADIUS = 20, 10
STEP = 0.25
# Change to a mounted Google Drive directory if cache must survive Colab resets.
ROOT = Path(os.environ.get('ACP_OUTPUT', 'output'))
MIN_COVERAGE_TO_PUBLISH = 0.95 # avoid publishing a mostly empty analysis
ROOT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / 'cache'; CACHE.mkdir(exist_ok=True)
STATE = Path(os.environ.get('ACP_STATE', '.state')); STATE.mkdir(parents=True,exist_ok=True)
HTTP = requests.Session()
HTTP.headers['User-Agent'] = 'ACP-Monitor/1.0 research'
HOUR = pd.Timedelta(hours=1)


def get(url, **kwargs):
    """Retry read-only requests; honor throttling without cycling IPs."""
    for attempt in range(4):
        try:
            r = HTTP.get(url, timeout=(20, 150), **kwargs)
            if r.status_code == 404:
                r.close(); raise FileNotFoundError(url)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = r.headers.get('Retry-After', '')
                wait = float(wait) if wait.isdigit() else 3 * 2**attempt
                r.close()
                if attempt == 3 or wait > 120:
                    raise RuntimeError('Source throttled/unavailable. Rerun later; cached frames are retained.')
                time.sleep(wait); continue
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 3: raise
            time.sleep(3 * 2**attempt)


def latest_earthquake():
    now = pd.Timestamp.now(tz='UTC')
    for days in (7, 30, 365):
        r = get('https://earthquake.usgs.gov/fdsnws/event/1/query', params={
            'format':'geojson','starttime':(now-pd.Timedelta(days=days)).isoformat(),
            'endtime':now.isoformat(),'minmagnitude':5,'eventtype':'earthquake',
            'orderby':'time','limit':1})
        features = r.json()['features']
        if features: return features[0]
    raise RuntimeError('USGS returned no M >= 5 earthquake in the past year.')


def source_url(t):
    cycle = t.floor('6h'); lead = int((t-cycle)/HOUR)
    return (f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{cycle:%Y%m%d}/'
            f'{cycle:%H}/atmos/gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}')


def decode_field(url, lines, name, latitudes, longitudes, t):
    matches = [i for i, line in enumerate(lines) if f':{name}:2 m above ground:' in line]
    if len(matches) != 1 or matches[0]+1 >= len(lines):
        raise RuntimeError('Required NOAA field missing or ambiguous in index.')
    i = matches[0]; lo=int(lines[i].split(':')[1]); hi=int(lines[i+1].split(':')[1])-1
    r=get(url+f'?part={lo}', headers={'Range':f'bytes={lo}-{hi}'}, stream=True)
    try:
        if r.status_code != 206: raise RuntimeError('Server ignored Range; full-file download stopped.')
        raw=r.content
    finally: r.close()
    if len(raw)!=hi-lo+1: raise RuntimeError('Truncated GRIB message.')
    gid=codes_new_from_message(raw)
    try:
        if codes_get(gid,'gridType')!='regular_ll' or codes_get(gid,'scanningMode')!=0:
            raise RuntimeError('Unsupported NOAA grid ordering.')
        valid_time=pd.to_datetime(str(codes_get(gid,'validityDate'))+f"{codes_get(gid,'validityTime'):04d}",format='%Y%m%d%H%M',utc=True)
        if valid_time!=t: raise RuntimeError('Source valid time differs from requested time.')
        nx,ny=int(codes_get(gid,'Ni')),int(codes_get(gid,'Nj'))
        dx,dy=codes_get(gid,'iDirectionIncrementInDegrees'),codes_get(gid,'jDirectionIncrementInDegrees')
        x0,y0=codes_get(gid,'longitudeOfFirstGridPointInDegrees'),codes_get(gid,'latitudeOfFirstGridPointInDegrees')
        x=np.rint(((longitudes-x0)%360)/dx).astype(int)%nx
        y=np.rint((y0-latitudes)/dy).astype(int)
        if np.any(y<0) or np.any(y>=ny): raise RuntimeError('Invalid latitude grid.')
        field=codes_get_values(gid).reshape(ny,nx)[np.ix_(y,x)].astype(float)
        missing=codes_get(gid,'missingValue')
        field[(field==missing)|(~np.isfinite(field))]=np.nan
        return field
    finally: codes_release(gid)


def download_acp(t, lats, lons):
    url=source_url(t)
    key=hashlib.sha256((url+str((lats[0],lats[-1],lons[0],lons[-1],STEP))).encode()).hexdigest()
    fn=CACHE/(key+'.npy')
    if fn.exists():
        arr=np.load(fn, allow_pickle=False)
        if arr.shape==(len(lats),len(lons)): return arr
    lines=get(url+'.idx').text.strip().splitlines()
    temperature=decode_field(url,lines,'TMP',lats,lons,t)-273.15
    rh=decode_field(url,lines,'RH',lats,lons,t)
    mask=np.isfinite(temperature)&np.isfinite(rh)&(rh>0)&(rh<=100)
    arr=np.full(temperature.shape,np.nan,dtype=np.float32)
    arr[mask]=5.8e-10*(20*temperature[mask]+5463)**2*np.log(100/rh[mask])
    temp=fn.with_suffix('.tmp')
    with temp.open('wb') as f: np.save(f,arr)
    temp.replace(fn)
    return arr


def make_video(event, times, cube, lats, lons, out):
    when=pd.to_datetime(event['properties']['time'],unit='ms',utc=True)
    lon,lat,_=event['geometry']['coordinates']
    ix=int(np.argmin(np.abs(lons-lon))); iy=int(np.argmin(np.abs(lats-lat)))
    series=cube[:,iy,ix]
    samples=series[np.isfinite(series)].astype(float)
    threshold=float(samples.mean()+3*samples.std(ddof=1)) if samples.size>=2 else None
    (out/'statistics.json').write_text(json.dumps({'threshold_eV':threshold,'definition':'mean + 3 sample standard deviations','valid_samples':int(samples.size)},indent=2))
    pd.DataFrame({'time_utc':times,'ACP_eV':series,'threshold_eV':threshold,
        'source':['analysis' if t.hour%6==0 else f'{t.hour%6}h forecast' for t in times]
    }).to_csv(out/'hourly_ACP.csv',index=False)
    np.savez_compressed(out/'ACP_grids.npz',ACP_eV=cube,latitude=lats,longitude=lons,time_utc=times.astype(str).to_numpy(dtype=str))
    valid=cube[np.isfinite(cube)]
    vmax=max(float(valid.max()),0.001)
    cm=plt.get_cmap('jet').copy();cm.set_bad('#c7cdd0')
    fig=plt.figure(figsize=(9.6,9.92),dpi=100,facecolor='white')
    gs=fig.add_gridspec(2,2,height_ratios=[2.4,1],width_ratios=[1,.045],hspace=.32,wspace=.12)
    proj=ccrs.PlateCarree(central_longitude=lon)
    ax=fig.add_subplot(gs[0,0],projection=proj); cbax=fig.add_subplot(gs[0,1]); graph=fig.add_subplot(gs[1,:])
    # Coordinates relative to epicenter avoid dateline discontinuities.
    x=lons-lon
    ax.set_extent([max(-180,x.min()-.125),min(180,x.max()+.125),max(-90,lats.min()-.125),min(90,lats.max()+.125)],crs=proj)
    mesh=ax.pcolormesh(x,lats,np.ma.masked_invalid(cube[0]),cmap=cm,vmin=0,vmax=vmax,shading='nearest',transform=proj)
    ax.coastlines(resolution='110m',linewidth=.7)
    ax.add_feature(cfeature.BORDERS,linewidth=.5)
    ax.plot(0,lat,'*',color='#ffe04b',markeredgecolor='black',markersize=15,transform=proj,zorder=6)
    grid=ax.gridlines(draw_labels=True,linewidth=.3,alpha=.5);grid.top_labels=False;grid.right_labels=False
    fig.colorbar(mesh,cax=cbax,label='ACP (eV)')
    graph.plot(times,series,color='#127660',lw=1,label='Hourly ACP')
    if threshold is not None:
        graph.axhline(threshold,color='#c93636',ls='--',lw=1.5,label=f'Threshold: mean + 3 SD = {threshold:.5f} eV')
        graph.set_ylim(0,max(float(samples.max()),threshold,1e-6)*1.12)
    graph.legend(loc='upper left',fontsize=7)

    graph.set(xlim=(when-pd.Timedelta(days=30),when),ylabel='ACP (eV)',xlabel='Day (UTC) — hourly samples')
    graph.grid(alpha=.2);graph.xaxis.set_major_locator(mdates.DayLocator(interval=3));graph.xaxis.set_major_formatter(mdates.DateFormatter('%d %b',tz=when.tz))
    graph.tick_params(axis='x',labelsize=8)
    cursor=graph.axvline(times[0],color='#c44c28',lw=1.8)
    dot,=graph.plot([],[],'o',color='#c44c28',ms=4)
    graph.set_title(f'Grid point nearest epicenter: {lats[iy]:.2f}°, {((lons[ix]+180)%360)-180:.2f}°',fontsize=10)
    title=fig.suptitle('',fontsize=12,color='#16382f')
    fig.text(.1,.025,'Exploratory ACP · Statistical threshold, not a validated seismic alarm',fontsize=8)
    matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    # Supply 5 scenes/sec; FFmpeg duplicates frames to a standard 30 fps stream.
    writer=FFMpegWriter(fps=FPS,codec='libx264',bitrate=3500,extra_args=['-r','30','-pix_fmt','yuv420p','-movflags','+faststart','-profile:v','high'])
    path=out/'ACP_30days_5fps.mp4'
    with writer.saving(fig,str(path),100):
        for k,t in enumerate(tqdm(times,desc='Rendering MP4')):
            mesh.set_array(np.ma.masked_invalid(cube[k]).ravel())
            cursor.set_xdata([t,t]);dot.set_data([t],[series[k]])
            title.set_text(f"Atmosphere Chemical Potential Monitor\nM {event['properties']['mag']:.1f} · {event['properties'].get('place','')[:65]}\nEarthquake {when:%Y-%m-%d %H:%M UTC} | Map {t:%Y-%m-%d %H:%M UTC}")
            if k==len(times)-1: fig.savefig(out/'ACP_map_chart.png',dpi=150)
            writer.grab_frame()
    plt.close(fig)
    return path, float(np.isfinite(series).mean())


def commit_state():
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        subprocess.run(['git','add','--all'],cwd=STATE,check=True)
        changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=STATE).returncode
        if changed:
            subprocess.run(['git','commit','-m','Update ACP publication state'],cwd=STATE,check=True,stdout=subprocess.DEVNULL)
            subprocess.run(['git','push','origin','HEAD:acp-state'],cwd=STATE,check=True,stdout=subprocess.DEVNULL)


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def publish(path, event, out):
    state_file=STATE/(event['id']+'.json')
    if state_file.exists():
        state=json.loads(state_file.read_text())
        if state.get('post_url'):
            print('Already published:',state['post_url']);return
        if state.get('status')=='posting':
            raise RuntimeError('Previous post outcome is uncertain. Check X before removing publication.json and retrying.')
    names=['X_API_KEY','X_API_SECRET','X_ACCESS_TOKEN','X_ACCESS_TOKEN_SECRET']
    credentials=[os.environ.get(n,'') for n in names]
    if not all(credentials): raise RuntimeError('Missing X OAuth 1.0a repository secrets: '+', '.join(n for n,v in zip(names,credentials) if not v))
    session=requests.Session();session.auth=OAuth1(*credentials)
    def call(method,path,**kwargs):
        # No automatic POST retries: avoid duplicate public posts.
        r=session.request(method,'https://api.x.com/2'+path,timeout=(20,180),**kwargs)
        if not r.ok: raise RuntimeError(f'X HTTP {r.status_code}: {r.reason}')
        return r.json() if r.content else {}
    init=call('POST','/media/upload/initialize',json={'media_type':'video/mp4','total_bytes':path.stat().st_size,'media_category':'tweet_video'})
    media_id=init['data']['id']
    with path.open('rb') as f:
        n=0
        while True:
            chunk=f.read(4*1024*1024)
            if not chunk: break
            call('POST',f'/media/upload/{media_id}/append',data={'segment_index':str(n)},files={'media':('chunk.mp4',chunk,'application/octet-stream')});n+=1
    status=call('POST',f'/media/upload/{media_id}/finalize').get('data',{})
    deadline=time.monotonic()+900
    while status.get('processing_info',{}).get('state') not in (None,'succeeded'):
        info=status['processing_info']
        if info.get('state')=='failed': raise RuntimeError('X video processing failed: '+str(info.get('error','')))
        if time.monotonic()>deadline: raise RuntimeError('X video processing timed out; local MP4 is retained.')
        time.sleep(min(60,max(1,info.get('check_after_secs',5))))
        status=call('GET','/media/upload',params={'command':'STATUS','media_id':media_id}).get('data',{})
    when=pd.to_datetime(event['properties']['time'],unit='ms',utc=True)
    lon,lat,_=event['geometry']['coordinates']
    coords=f"{abs(lat):.3f}°{'N' if lat>=0 else 'S'}, {abs(lon):.3f}°{'E' if lon>=0 else 'W'}"
    text=(f"M{event['properties']['mag']:.1f} · {(event['properties'].get('place') or 'Global earthquake')[:50]}\n"
          f"{when:%Y-%m-%d %H:%M UTC}\nEpicenter: {coords}\n"
          "30-day hourly ACP evolution.\nExploratory, not a seismic prediction.\n"
          f"https://earthquake.usgs.gov/earthquakes/eventpage/{event['id']}")
    (out/'post_text.txt').write_text(text)
    atomic_json(state_file,{'status':'posting','event_id':event['id'],'media_id':media_id});commit_state()
    result=call('POST','/tweets',json={'text':text,'media':{'media_ids':[media_id]}})
    url='https://x.com/i/web/status/'+result['data']['id']
    atomic_json(state_file,{'status':'published','post_url':url,'event_id':event['id']});commit_state()
    print('Published:',url)


def process_event(event):
    when=pd.to_datetime(event['properties']['time'],unit='ms',utc=True)
    lon,lat,_=event['geometry']['coordinates']
    out=ROOT/event['id'];out.mkdir(exist_ok=True)
    (out/'event.json').write_text(json.dumps(event,indent=2))
    print(f"Processing USGS M >= 5: {event['id']} | M{event['properties']['mag']} | {when} | {event['properties'].get('place','')}")
    # Fix the chosen event for the full run; a later earthquake does not replace it mid-analysis.
    lats=np.arange(math.ceil(max(-90,lat-LAT_RADIUS)/STEP),math.floor(min(90,lat+LAT_RADIUS)/STEP)+1)*STEP
    lons=np.arange(math.ceil((lon-LON_RADIUS)/STEP),math.floor((lon+LON_RADIUS)/STEP)+1)*STEP
    times=pd.date_range((when-pd.Timedelta(days=30)).ceil('h'),when, freq='h',inclusive='left')
    assert len(times)==720
    cube=np.full((len(times),len(lats),len(lons)),np.nan,dtype=np.float32)
    missing=[]
    for i,t in enumerate(tqdm(times,desc='NOAA hourly ACP')):
        try: cube[i]=download_acp(t,lats,lons)
        except FileNotFoundError: missing.append(t.isoformat())
    (out/'missing_times.json').write_text(json.dumps(missing,indent=2))
    coverage=float(np.isfinite(cube).mean())
    if not np.isfinite(cube).any(): raise RuntimeError('No NOAA data recovered; no video or post produced.')
    path,point_coverage=make_video(event,times,cube,lats,lons,out)
    print(f'MP4 saved: {path} | 144 seconds | map coverage {coverage:.1%} | chart coverage {point_coverage:.1%}')
    if PUBLISH_TO_X:
        if min(coverage,point_coverage)<MIN_COVERAGE_TO_PUBLISH:
            raise RuntimeError('Insufficient coverage for publication; cached data and output retained.')
        publish(path,event,out)
    else:
        print('DRY RUN: publication disabled.')


def main():
    queue_path=STATE/'queue.json'
    state=json.loads(queue_path.read_text()) if queue_path.exists() else {'initialized':False,'events':{}}
    now=pd.Timestamp.now(tz='UTC')
    # Re-scan 7 days for revised magnitudes and delayed reports; first run is 24 h.
    start=now-pd.Timedelta(days=7 if state['initialized'] else 1)
    offset=1
    while True:
        result=get('https://earthquake.usgs.gov/fdsnws/event/1/query',params={
            'format':'geojson','starttime':start.isoformat(),'endtime':now.isoformat(),
            'minmagnitude':5,'eventtype':'earthquake','orderby':'time-asc','limit':20000,'offset':offset}).json()
        items=result.get('features',[])
        for event in items: state['events'][event['id']]=event
        if len(items)<20000: break
        offset+=20000
    state['initialized']=True
    if PUBLISH_TO_X: atomic_json(queue_path,state);commit_state()
    pending=[]
    for event in sorted(state['events'].values(),key=lambda e:e['properties']['time']):
        sf=STATE/(event['id']+'.json')
        prior=json.loads(sf.read_text()) if sf.exists() else {}
        if prior.get('status') in ('published','posting'):
            if prior.get('status')=='posting': print('Manual review needed for uncertain previous post:',event['id'])
            continue
        pending.append(event)
    def attempts(e):
        f=STATE/(e['id']+'.json')
        return json.loads(f.read_text()).get('attempts',0) if f.exists() else 0
    pending.sort(key=lambda e:(attempts(e),e['properties']['time']))
    maximum=int(os.environ.get('MAX_EVENTS_PER_RUN','1'))
    print(f'{len(pending)} queued events; processing at most {maximum} this run.')
    failures=[]
    for event in pending[:maximum]:
        try: process_event(event)
        except Exception as exc:
            print('Event failed:',event['id'],str(exc));failures.append(event['id'])
            sf=STATE/(event['id']+'.json')
            prior=json.loads(sf.read_text()) if sf.exists() else {}
            if PUBLISH_TO_X and prior.get('status') not in ('posting','published'):
                atomic_json(sf,{'status':'failed','attempts':prior.get('attempts',0)+1});commit_state()
    if failures: raise RuntimeError('Incomplete events (retried on next run unless post outcome is uncertain): '+', '.join(failures))


if __name__=='__main__':
    main()
