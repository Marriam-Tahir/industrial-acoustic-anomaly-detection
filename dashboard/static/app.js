'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const state = {file:null,health:null,result:null,busy:false,url:null,wave:null,waveDuration:0,previewWave:null,previewDuration:0,previewToken:0,history:[],toastTimer:null};
  const HISTORY_KEY = 'smart-listener.history.v1';
  const titles = {analysis:['Sound analysis','Listen closely.','Detect early.',"Understand your machine's health, one recording at a time."],models:['Model library','Built to listen.','Ready to inspect.','Explore the trained models and their saved evaluation metrics.'],pipeline:['API & pipeline','From sound.','To a signal.','A transparent path from your recording to a machine health decision.'],history:['Session history','A record of','what you heard.','Review the results of your recent machine sound analyses.']};
  const node = (tag,text,cls) => { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n; };
  const icon = name => { const s=document.createElementNS('http://www.w3.org/2000/svg','svg');s.setAttribute('class','icon');s.setAttribute('aria-hidden','true');const u=document.createElementNS('http://www.w3.org/2000/svg','use');u.setAttribute('href','#i-'+name);s.append(u);return s; };
  const set = (id,value) => { $(id).textContent=value; };
  const cap = value => String(value||'Unknown').replace(/^./,s=>s.toUpperCase());
  const number = value => {const n=Number(value);if(!Number.isFinite(n))return '—';if(n===0)return '0';return Math.abs(n)<0.001||Math.abs(n)>=10000?n.toExponential(3):n.toLocaleString(undefined,{maximumFractionDigits:5});};
  const noise = value => ({'6_dB':'+6 dB','0_dB':'0 dB','-6_dB':'−6 dB'}[value]||String(value||'—'));
  const message = text => {set('global-message',text||'');$('global-message').hidden=!text;};
  const toast = text => {clearTimeout(state.toastTimer);set('toast',text);$('toast').hidden=false;state.toastTimer=setTimeout(()=>$('toast').hidden=true,3500);};

  function openView(view){
    if(!titles[view])view='analysis';
    document.querySelectorAll('.view').forEach(section=>section.hidden=section.id!=='view-'+view);
    document.querySelectorAll('[data-view]').forEach(button=>{const selected=button.dataset.view===view;button.classList.toggle('active',selected);if(selected)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');});
    const t=titles[view];set('breadcrumb-title',t[0]);$('page-title').replaceChildren(document.createTextNode(t[1]+' '),node('span',t[2]));set('page-subtitle',t[3]);
    if(location.hash!=='#'+view)history.replaceState(null,'','#'+view);
    requestAnimationFrame(drawAll);
  }
  function availableModel(){
    if(!state.health)return false;const machine=$('machine-type').value;
    return (state.health.models||[]).some(model=>(machine==='auto'||model.machine_type===machine)&&model.snr===$('snr').value);
  }
  function updateButton(){
    const button=$('analyze-button');button.disabled=state.busy||!state.file||!availableModel();
    button.querySelector('span').textContent=state.busy?'Analyzing recording…':!state.health?'API unavailable':state.file&&!availableModel()?'Model unavailable':'Analyze recording';
    ['machine-type','snr','browse-file','remove-file','audio-file'].forEach(id=>$(id).disabled=state.busy);
    button.setAttribute('aria-busy',String(state.busy));
  }
  function renderModels(){
    const models=state.health?.models||[];$('model-grid').replaceChildren();set('nav-model-count',models.length);set('library-count',models.length+' exported models');
    if(!models.length){$('model-grid').append(node('p','No ready model exports are available. Check the API status.','message error'));return;}
    models.forEach(model=>{
      const card=node('article',undefined,'card model-card');const mark=node('div',undefined,'model-icon');mark.append(icon('machine'));
      card.append(mark,node('h3',cap(model.machine_type)+' · '+noise(model.snr)),node('p',model.id||'Exported anomaly detector'));
      const details=node('div',undefined,'model-meta');
      const add=(label,value)=>{const row=node('div');row.append(node('span',label),node('strong',value));details.append(row);};
      add('Decision threshold',number(model.threshold));add('Window aggregation',cap(model.aggregation||'max'));
      ['auc','f1','precision','recall'].forEach(key=>{const value=model.metrics?.[key];if(typeof value==='number'&&Number.isFinite(value))add('Test '+key.toUpperCase(),value.toFixed(3));});
      card.append(details,node('span','EXPORTED MODEL','badge success'));$('model-grid').append(card);
    });
  }
  function renderConfig(){
    const p=state.health?.preprocessing||{};const verified=p.verified===true;
    set('preprocessing-status',verified?'Training-aligned preprocessing':'Preprocessing verification pending');
    const summary=p.sample_rate?(p.sample_rate/1000)+' kHz mono · '+(p.clip_seconds||10)+' s · '+(p.n_windows||19)+' windows':'Configuration information is unavailable.';
    set('preprocessing-detail',summary);set('pipeline-config-summary',summary+'. Features use the exported training settings.');
    const rows=[['Verification',verified?'Code hash + parameters verified':p.verified===false?'Not verified':'Unavailable'],['Sample rate',p.sample_rate?number(p.sample_rate)+' Hz':'—'],['Analysis duration',p.clip_seconds?p.clip_seconds+' seconds':'—'],['Feature shape',Array.isArray(p.output_shape)?p.output_shape.join(' × '):'—'],['Configuration fingerprint',p.fingerprint||'—']];
    $('config-details').replaceChildren();rows.forEach(([label,value])=>{const row=node('div');row.append(node('dt',label),node('dd',value));$('config-details').append(row);});
    if(state.health?.classifier?.detail)set('classifier-detail',state.health.classifier.detail);
    const max=state.health?.max_upload_bytes;if(max)set('file-hint','WAV audio · Up to '+Math.round(max/1048576)+' MiB · '+(p.clip_seconds||10)+'-second analysis');
  }
  async function loadHealth(){
    $('refresh-health').disabled=true;const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),15000);
    try{
      const response=await fetch('/api/health',{cache:'no-store',signal:controller.signal});const data=await response.json();if(!Array.isArray(data.models))throw new Error('Invalid health response');state.health=data;
      const ready=data.status==='ready';set('connection-text',ready?'System ready':'System needs attention');set('side-status-text',ready?'Inference API online':'API needs attention');
      $('connection-pill').className='connection-pill'+(ready?'':' degraded');$('connection-pill').querySelector('.status-dot').className='status-dot'+(ready?'':' pending');$('side-status-dot').className='status-dot'+(ready?'':' pending');
      if(!ready&&Array.isArray(data.errors)&&data.errors.length)message(data.errors.join(' '));
      renderModels();renderConfig();
    }catch(error){
      state.health=null;set('connection-text','API offline');set('side-status-text','Inference API offline');$('connection-pill').className='connection-pill offline';$('connection-pill').querySelector('.status-dot').className='status-dot offline';$('side-status-dot').className='status-dot offline';renderModels();renderConfig();
    }finally{clearTimeout(timer);$('refresh-health').disabled=false;updateButton();drawAll();}
  }
  function clearResult(){
    state.result=null;$('result-content').hidden=true;$('result-loading').hidden=true;$('result-empty').hidden=false;$('result-warnings').hidden=true;$('result-warnings').replaceChildren();
    set('result-state','Awaiting audio');$('result-state').className='badge neutral';$('score-empty').hidden=false;$('spectrogram-empty').hidden=false;set('window-count','Overlapping audio windows');set('aggregation-info','Model-specific threshold');
    state.wave=state.previewWave;state.waveDuration=state.previewDuration;drawAll();
  }
  function removeFile(){
    state.previewToken++;if(state.url)URL.revokeObjectURL(state.url);state.url=null;state.file=null;state.previewWave=null;state.previewDuration=0;state.wave=null;
    $('audio-file').value='';$('audio-player').pause();$('audio-player').removeAttribute('src');$('audio-player').load();$('upload-empty').hidden=false;$('upload-selected').hidden=true;$('audio-preview').hidden=true;
    clearResult();message('');updateButton();
  }
  async function selectFile(file){
    if(state.busy||!file)return;
    removeFile();const max=state.health?.max_upload_bytes||26214400;
    if(!/\.wav$/i.test(file.name)){message('Please choose a WAV recording. Other file formats are not supported.');return;}
    if(file.size===0){message('This recording is empty. Choose a WAV file containing audio.');return;}
    if(file.size>max){message('This recording exceeds the '+Math.round(max/1048576)+' MiB upload limit. Choose a smaller WAV file.');return;}
    state.file=file;state.url=URL.createObjectURL(file);const token=++state.previewToken;
    set('file-name',file.name);set('file-meta',(file.size/1024).toLocaleString(undefined,{maximumFractionDigits:1})+' KiB · WAV audio');set('preview-duration','Loading preview');
    $('upload-empty').hidden=true;$('upload-selected').hidden=false;$('audio-preview').hidden=false;$('audio-player').src=state.url;updateButton();
    let context;
    try{
      const AudioContext=window.AudioContext||window.webkitAudioContext;if(!AudioContext)throw new Error('Preview unavailable');context=new AudioContext();
      const buffer=await context.decodeAudioData(await file.arrayBuffer());if(token!==state.previewToken)return;
      const channels=Array.from({length:buffer.numberOfChannels},(_,i)=>buffer.getChannelData(i));const size=700;const wave=[];
      for(let i=0;i<size;i++){
        const start=Math.floor(i*buffer.length/size),end=Math.max(start+1,Math.floor((i+1)*buffer.length/size));let peak=0;const step=Math.max(1,Math.floor((end-start)/160));
        for(let j=start;j<Math.min(end,buffer.length);j+=step){let sum=0;for(const channel of channels)sum+=channel[j];peak=Math.max(peak,Math.abs(sum/channels.length));}wave.push(peak);
      }
      state.previewWave=wave;state.previewDuration=buffer.duration;state.wave=wave;state.waveDuration=buffer.duration;
      set('preview-duration','Full recording · '+buffer.duration.toFixed(2)+' s');
      set('file-meta',(file.size/1024).toLocaleString(undefined,{maximumFractionDigits:1})+' KiB · '+buffer.duration.toFixed(2)+' s · '+(buffer.numberOfChannels===1?'Mono':buffer.numberOfChannels+' channels'));drawWave();
    }catch(error){if(token===state.previewToken)set('preview-duration','Preview unavailable · API will validate audio');}
    finally{if(context)context.close().catch(()=>{});}
  }
  function renderResult(result){
    state.result=result;const abnormal=result.is_anomaly===true||result.condition==='abnormal';$('result-empty').hidden=true;$('result-loading').hidden=true;$('result-content').hidden=false;
    set('result-state','Analysis complete');$('result-state').className='badge '+(abnormal?'danger':'success');$('diagnosis-banner').className='diagnosis-banner'+(abnormal?' abnormal':'');$('diagnosis-icon').setAttribute('href',abnormal?'#i-alert':'#i-check');
    set('condition-text',abnormal?'Abnormal':'Normal');set('condition-detail',abnormal?'The anomaly score exceeds this model’s threshold.':'The anomaly score is within this model’s threshold.');
    set('result-machine',cap(result.machine_type));const method=result.machine_identification?.method;set('identification-method',method==='filename'?'Identified from filename':'Selected by operator');$('identification-method').title=result.machine_identification?.detail||'';
    set('result-elapsed',number(Number(result.elapsed_ms)/1000)+' s');set('result-score',number(result.score));set('result-threshold',number(result.threshold));
    const ratio=Number(result.threshold)>0?Number(result.score)/Number(result.threshold):null;set('score-ratio',ratio===null?'Raw model score':ratio.toFixed(2)+'× the threshold');
    const maximum=Math.max(Number(result.score),Number(result.threshold),0.000000001)*1.25;$('score-fill').style.width=(100*Math.max(0,Number(result.score))/maximum)+'%';$('score-fill').style.background=abnormal?'#d7a56d':'#a2bf83';$('threshold-marker').style.left=(100*Math.max(0,Number(result.threshold))/maximum)+'%';
    set('result-model',noise(result.snr)+' · '+(result.aggregation||'max')+' aggregation');$('result-model').title=result.model_id||'';
    const warnings=Array.isArray(result.warnings)?result.warnings:[];$('result-warnings').replaceChildren();warnings.forEach(warning=>$('result-warnings').append(node('p',String(warning))));$('result-warnings').hidden=warnings.length===0;
    const scores=Array.isArray(result.window_scores)?result.window_scores:[];$('score-empty').hidden=scores.length>0;$('spectrogram-empty').hidden=!!result.spectrogram?.values?.length;set('window-count',scores.length+' overlapping windows');set('aggregation-info',cap(result.aggregation||'max')+' window score · '+noise(result.snr));
    if(!state.previewWave&&Array.isArray(result.audio?.waveform)&&result.audio.waveform.length){state.wave=result.audio.waveform;state.waveDuration=10;set('preview-duration','Waveform: analyzed 10 s · playback: original file');}
    drawAll();
  }
  async function analyze(){
    if(state.busy||!state.file)return;
    message('');clearResult();state.busy=true;updateButton();$('result-empty').hidden=true;$('result-loading').hidden=false;set('result-state','Listening');$('result-state').className='badge neutral';
    const started=performance.now();const ticker=setInterval(()=>set('processing-time',((performance.now()-started)/1000).toFixed(1)+' s'),100);set('processing-time','0.0 s');const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),180000);
    try{
      const form=new FormData();form.append('file',state.file);form.append('machine_type',$('machine-type').value);form.append('snr',$('snr').value);
      const response=await fetch('/api/predict',{method:'POST',body:form,signal:controller.signal});let result;
      try{result=await response.json();}catch(error){throw new Error('The API returned an unreadable response. Check the server logs and try again.');}
      if(!response.ok){const detail=result.error?.message||'The recording could not be analyzed.';throw new Error(detail+(result.request_id?' (Request '+result.request_id+')':''));}
      if(typeof result.score!=='number'||!Number.isFinite(result.score)||typeof result.threshold!=='number'||!Number.isFinite(result.threshold)||!result.machine_type)throw new Error('The API returned an incomplete prediction. Check the model export and server logs.');
      renderResult(result);saveHistory(result);
    }catch(error){clearResult();message(error.name==='AbortError'?'The analysis took too long. Check that the API is running, then try again.':error.message||'Could not reach the API. Check the local server and try again.');set('result-state','Needs attention');$('result-state').className='badge danger';}
    finally{clearInterval(ticker);clearTimeout(timer);state.busy=false;updateButton();}
  }
  function loadHistory(){try{const entries=JSON.parse(localStorage.getItem(HISTORY_KEY)||'[]');state.history=Array.isArray(entries)?entries.filter(entry=>entry&&typeof entry==='object'&&typeof entry.filename==='string').slice(0,40):[];}catch(error){state.history=[];}renderHistory();}
  function saveHistory(result){
    state.history.unshift({request_id:result.request_id,filename:result.filename||state.file?.name||'Recording',machine_type:result.machine_type,condition:result.condition,score:result.score,threshold:result.threshold,snr:result.snr,timestamp:result.timestamp||new Date().toISOString()});state.history=state.history.slice(0,40);
    try{localStorage.setItem(HISTORY_KEY,JSON.stringify(state.history));}catch(error){toast('Analysis complete. Browser history storage is unavailable.');}renderHistory();
  }
  function renderHistory(){
    const hasEntries=state.history.length>0;$('history-empty').hidden=hasEntries;$('history-table-wrap').hidden=!hasEntries;$('clear-history').disabled=!hasEntries;$('history-body').replaceChildren();
    state.history.forEach(entry=>{
      const row=node('tr');const condition=entry.condition==='abnormal'?'Abnormal':'Normal';const first=node('td',entry.filename);first.title=entry.filename;
      row.append(first,node('td',cap(entry.machine_type)));const status=node('td');status.append(node('span',condition,'badge '+(entry.condition==='abnormal'?'danger':'success')));
      row.append(status,node('td',number(entry.score)+' / '+number(entry.threshold)),node('td',noise(entry.snr)));const date=new Date(entry.timestamp);row.append(node('td',Number.isNaN(date.getTime())?'—':date.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'})));$('history-body').append(row);
    });
  }
  function downloadReport(){
    if(!state.result)return;const blob=new Blob([JSON.stringify(state.result,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=node('a');a.href=url;
    a.download='smart-listener-'+state.result.machine_type+'-'+state.result.condition+'-'+String(state.result.request_id||Date.now()).replace(/[^a-zA-Z0-9_-]/g,'').slice(0,40)+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('Analysis report downloaded.');
  }
  function surface(id){
    const canvas=$(id),rect=canvas.getBoundingClientRect();if(!rect.width||!rect.height)return null;const dpr=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);return {ctx,w:rect.width,h:rect.height};
  }
  function drawWave(){
    const s=surface('waveform-canvas');if(!s)return;const {ctx,w,h}=s;ctx.clearRect(0,0,w,h);ctx.strokeStyle='#e9eee1';ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();
    const data=state.wave;if(!data?.length)return;const peak=Math.max(...data.map(value=>Math.abs(Number(value)||0)),0.001);ctx.strokeStyle='#9bb780';ctx.lineWidth=1.5;const bars=Math.floor(w/3);ctx.beginPath();
    for(let i=0;i<bars;i++){const index=Math.min(data.length-1,Math.floor(i*data.length/bars));const height=Math.max(1,Math.abs(Number(data[index])||0)/peak*(h-12)/2);const x=i*w/bars;ctx.moveTo(x,h/2-height);ctx.lineTo(x,h/2+height);}ctx.stroke();
  }
  function chartBase(s){
    const {ctx,w,h}=s;const p={x:49,y:16,w:w-62,h:h-43};ctx.clearRect(0,0,w,h);ctx.font='10px Segoe UI,Arial,sans-serif';ctx.fillStyle='#75836a';ctx.strokeStyle='#eef1e8';ctx.lineWidth=1;
    for(let i=0;i<4;i++){const y=p.y+i*p.h/3;ctx.beginPath();ctx.moveTo(p.x,y);ctx.lineTo(p.x+p.w,y);ctx.stroke();}return p;
  }
  function drawScores(){
    const s=surface('score-canvas');if(!s)return;const {ctx}=s,p=chartBase(s),r=state.result;
    const duration=Number(r?.preprocessing?.clip_seconds)||Number(state.health?.preprocessing?.clip_seconds)||10;ctx.fillStyle='#75836a';ctx.textAlign='center';[0,.5,1].forEach(v=>ctx.fillText(number(v*duration)+' s',p.x+v*p.w,p.y+p.h+19));
    const data=r?.window_scores;if(!Array.isArray(data)||!data.length)return;const values=data.map(Number);if(values.some(value=>!Number.isFinite(value)))return;const threshold=Number(r.threshold);const maximum=Math.max(...values,threshold,0.000000001)*1.18;
    ctx.textAlign='right';for(let i=0;i<4;i++)ctx.fillText((maximum*(1-i/3)).toFixed(4),p.x-7,p.y+i*p.h/3+3);
    const y=value=>p.y+p.h-(Math.max(0,value)/maximum)*p.h;
    const x=index=>{const t=r.window_times?.[index];const time=t?(Number(t.start)+Number(t.end))/2:index*duration/Math.max(1,values.length-1);return p.x+Math.min(duration,Math.max(0,time))/duration*p.w;};
    ctx.save();ctx.setLineDash([4,4]);ctx.strokeStyle='#c09661';ctx.beginPath();ctx.moveTo(p.x,y(threshold));ctx.lineTo(p.x+p.w,y(threshold));ctx.stroke();ctx.restore();
    const gradient=ctx.createLinearGradient(0,p.y,0,p.y+p.h);gradient.addColorStop(0,'#b7d39a55');gradient.addColorStop(1,'#b7d39a00');ctx.beginPath();ctx.moveTo(x(0),p.y+p.h);values.forEach((value,index)=>ctx.lineTo(x(index),y(value)));ctx.lineTo(x(values.length-1),p.y+p.h);ctx.closePath();ctx.fillStyle=gradient;ctx.fill();
    ctx.beginPath();values.forEach((value,index)=>index?ctx.lineTo(x(index),y(value)):ctx.moveTo(x(index),y(value)));ctx.strokeStyle='#78975f';ctx.lineWidth=1.7;ctx.stroke();
    values.forEach((value,index)=>{ctx.beginPath();ctx.arc(x(index),y(value),2.3,0,2*Math.PI);ctx.fillStyle=value>threshold?'#cc9b65':'#8baa70';ctx.fill();});
    $('score-canvas').setAttribute('aria-label','Anomaly scores across '+values.length+' windows. Threshold '+number(threshold)+'. Aggregation '+r.aggregation+'.');
  }
  function drawSpectrogram(){
    const s=surface('spectrogram-canvas');if(!s)return;const {ctx}=s,p=chartBase(s);const spec=state.result?.spectrogram;const duration=Number(spec?.duration_seconds)||10;
    ctx.fillStyle='#75836a';ctx.textAlign='center';[0,.5,1].forEach(v=>ctx.fillText(number(v*duration)+' s',p.x+v*p.w,p.y+p.h+19));
    const values=spec?.values;if(!Array.isArray(values)||!values.length||!Array.isArray(values[0])||!values[0].length)return;const rows=values.length,cols=values[0].length;let min=Infinity,max=-Infinity;
    values.forEach(row=>row.forEach(value=>{if(Number.isFinite(Number(value))){min=Math.min(min,Number(value));max=Math.max(max,Number(value));}}));if(!Number.isFinite(min))return;
    const off=document.createElement('canvas');off.width=cols;off.height=rows;const offctx=off.getContext('2d'),image=offctx.createImageData(cols,rows);const colors=[[32,53,59],[55,104,100],[112,168,131],[196,211,137],[226,179,104]];
    for(let row=0;row<rows;row++){for(let col=0;col<cols;col++){
      const value=Number(values[row][col]);const t=Math.min(1,Math.max(0,(value-min)/Math.max(max-min,0.000001)));const scaled=t*(colors.length-1),a=Math.floor(scaled),b=Math.min(a+1,colors.length-1),mix=scaled-a;const i=((rows-row-1)*cols+col)*4;
      for(let c=0;c<3;c++)image.data[i+c]=Math.round(colors[a][c]*(1-mix)+colors[b][c]*mix);image.data[i+3]=255;
    }}
    offctx.putImageData(image,0,0);ctx.imageSmoothingEnabled=true;ctx.drawImage(off,p.x,p.y,p.w,p.h);ctx.fillStyle='#75836a';ctx.textAlign='right';const frequencies=spec.frequencies_hz||[];
    [0,.5,1].forEach(v=>{const index=Math.round(v*(rows-1));const hz=Number(frequencies[index]);if(Number.isFinite(hz))ctx.fillText(hz>=1000?(hz/1000).toFixed(1)+'k':Math.round(hz),p.x-7,p.y+(1-v)*p.h+3);});
  }
  function drawAll(){drawWave();drawScores();drawSpectrogram();}
  document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>openView(button.dataset.view)));
  document.querySelectorAll('[data-open-view]').forEach(button=>button.addEventListener('click',()=>openView(button.dataset.openView)));
  window.addEventListener('hashchange',()=>openView(location.hash.slice(1)));
  $('browse-file').addEventListener('click',()=>$('audio-file').click());$('audio-file').addEventListener('change',event=>selectFile(event.target.files[0]));$('remove-file').addEventListener('click',removeFile);
  const drop=$('drop-zone');['dragenter','dragover'].forEach(event=>drop.addEventListener(event,e=>{e.preventDefault();if(!state.busy)drop.classList.add('dragover');}));
  ['dragleave','drop'].forEach(event=>drop.addEventListener(event,e=>{e.preventDefault();drop.classList.remove('dragover');}));
  drop.addEventListener('drop',e=>{if(!state.busy){if(e.dataTransfer.files.length>1)message('Choose one recording at a time.');else selectFile(e.dataTransfer.files[0]);}});
  ['machine-type','snr'].forEach(id=>$(id).addEventListener('change',()=>{message('');clearResult();updateButton();}));
  $('analyze-button').addEventListener('click',analyze);$('refresh-health').addEventListener('click',loadHealth);$('download-report').addEventListener('click',downloadReport);
  $('clear-history').addEventListener('click',()=>{state.history=[];try{localStorage.removeItem(HISTORY_KEY);}catch(error){}renderHistory();toast('Browser analysis history cleared.');});
  set('curl-example',['curl -X POST "'+location.origin+'/api/predict" '+String.fromCharCode(92),'  -F "file=@recording.wav" '+String.fromCharCode(92),'  -F "machine_type=fan" '+String.fromCharCode(92),'  -F "snr=6_dB"'].join(String.fromCharCode(10)));
  $('copy-curl').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('curl-example').textContent);toast('API request copied.');}catch(error){toast('Copy is unavailable. Select the request text to copy it.');}});
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(drawAll,100);});
  window.addEventListener('beforeunload',()=>{if(state.url)URL.revokeObjectURL(state.url);});
  loadHistory();openView(location.hash.slice(1)||'analysis');loadHealth();setInterval(()=>{if(!document.hidden&&!state.busy)loadHealth();},60000);
})();
