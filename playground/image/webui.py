"""Small text-to-image UI backed by one local ComfyUI process."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local Model Playground</title>
<style>
:root{color-scheme:dark;--bg:#111318;--panel:#191c23;--line:#303541;--text:#f2f3f5;--muted:#9aa3b2;--accent:#ff8a4c;--soft:#272c36;--ok:#62d394;--bad:#ff6b6b}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 75% 0,#23212a 0,#111318 38%);font:14px/1.45 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--text)}button,input,textarea,select{font:inherit}.top{height:58px;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 22px;gap:14px;background:#111318dd;position:sticky;top:0;z-index:3}.brand{font-weight:700;font-size:16px}.badge{background:var(--soft);border:1px solid var(--line);padding:5px 9px;border-radius:999px;color:#d9dde4}.status{margin-left:auto;color:var(--ok)}.link{color:#c7cdd7;text-decoration:none}.layout{display:grid;grid-template-columns:minmax(330px,410px) 1fr;min-height:calc(100vh - 58px)}.controls{padding:20px;border-right:1px solid var(--line);background:#15181ecc}.viewer{padding:24px;min-width:0}.label{display:flex;justify-content:space-between;margin:15px 0 7px;font-weight:600}.hint{color:var(--muted);font-weight:400;font-size:12px}textarea,input,select{width:100%;color:var(--text);background:#101218;border:1px solid var(--line);border-radius:9px;padding:10px;outline:none}textarea:focus,input:focus,select:focus{border-color:#a95f3c}textarea{resize:vertical;min-height:116px}.seg{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.seg button,.ratio button,.ghost{background:var(--soft);border:1px solid var(--line);color:var(--text);border-radius:8px;padding:9px;cursor:pointer}.seg button.active,.ratio button.active{background:#6f3d29;border-color:var(--accent)}.ratio{display:flex;gap:7px;flex-wrap:wrap}.ratio button{padding:7px 11px}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.advanced{margin-top:16px;border-top:1px solid var(--line);padding-top:12px}.advanced summary{cursor:pointer;font-weight:650;padding:5px 0}.advice{margin:14px 0;background:#20242d;border:1px solid #353b48;border-radius:10px;padding:12px;color:#cbd1da}.advice strong{display:block;color:#fff;margin-bottom:3px}.generate{width:100%;border:0;border-radius:10px;padding:13px;margin-top:15px;background:var(--accent);color:#17100d;font-weight:800;cursor:pointer}.generate:disabled{opacity:.55;cursor:default}.cancel{width:100%;margin-top:8px;background:transparent;color:#ff9999;border:1px solid #713b3b;border-radius:10px;padding:10px;cursor:pointer}.hidden{display:none!important}.stage{display:flex;align-items:center;gap:10px;margin-bottom:9px;color:var(--muted)}.spinner{width:12px;height:12px;border:2px solid #555;border-top-color:var(--accent);border-radius:50%;animation:s .8s linear infinite}@keyframes s{to{transform:rotate(360deg)}}.progress{height:8px;background:var(--soft);border:1px solid var(--line);border-radius:99px;overflow:hidden;margin:0 0 16px}.progress div{height:100%;width:0;background:linear-gradient(90deg,#ff7442,var(--accent));border-radius:99px;transition:width .25s ease}.canvas{min-height:520px;border:1px solid var(--line);border-radius:14px;background:#0c0e12;display:grid;place-items:center;overflow:hidden;position:relative}.canvas img{max-width:100%;max-height:74vh;display:block}.preview-label{position:absolute;left:12px;bottom:12px;background:#101218dd;border:1px solid var(--line);border-radius:999px;padding:6px 10px;color:#d7dbe2;font-size:12px}.empty{text-align:center;color:var(--muted)}.empty b{display:block;color:#d5d9e0;margin-bottom:6px}.meta{margin-top:12px;color:var(--muted);display:flex;gap:14px;flex-wrap:wrap}.result-path{margin-top:10px;display:flex;align-items:center;gap:8px;min-width:0;color:var(--muted)}.result-path code{background:var(--soft);border:1px solid var(--line);padding:7px 9px;border-radius:7px;overflow:auto;white-space:nowrap;flex:1}.result-path button{background:var(--soft);border:1px solid var(--line);color:var(--text);border-radius:7px;padding:7px 10px;cursor:pointer}.history-title{margin:24px 0 10px;font-weight:700}.history{display:flex;gap:10px;overflow:auto;padding-bottom:7px}.thumb{flex:0 0 112px;background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden;cursor:pointer}.thumb img{width:112px;height:92px;object-fit:cover;display:block}.thumb span{display:block;padding:6px;font-size:11px;color:var(--muted)}.error{color:var(--bad);margin-top:10px;white-space:pre-wrap}@media(max-width:820px){.layout{grid-template-columns:1fr}.controls{border-right:0;border-bottom:1px solid var(--line)}.viewer{padding:16px}.canvas{min-height:360px}}
</style><style>
.history-toolbar{display:flex;align-items:center;gap:8px;margin:24px 0 10px}.history-toolbar strong{margin-right:auto}.history-toolbar button{background:var(--soft);border:1px solid var(--line);color:var(--text);border-radius:7px;padding:6px 9px;cursor:pointer}.history-toolbar button.danger{color:#ffaaaa;border-color:#633}.history-toolbar button:disabled{opacity:.45}.history{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;overflow:visible}.thumb{position:relative;min-width:0;display:flex;flex-direction:column}.thumb img{width:100%;height:120px}.thumb .select{position:absolute;left:7px;top:7px;background:#111b;padding:4px;border-radius:6px}.thumb .select input{width:auto;margin:0}.thumb .delete-one{position:absolute;right:7px;top:7px;background:#111d;border:1px solid #633;color:#ffb0b0;border-radius:6px;cursor:pointer}.thumb .prompt{color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.thumb small{padding:0 6px 7px;color:var(--muted)}
</style></head><body>
<header class="top"><div class="brand">Local Model Playground</div><div id="model" class="badge"></div><div id="ready" class="status">● Ready</div><a id="comfy" class="link" target="_blank">Open ComfyUI ↗</a></header>
<main class="layout"><section class="controls">
<div class="label"><span>Prompt</span><span id="promptCount" class="hint">0 / 4000</span></div><textarea id="prompt" maxlength="4000" placeholder="A small red fox sitting in a sunlit meadow, detailed natural photograph"></textarea>
<div class="label"><span>Preset</span><span id="presetHint" class="hint"></span></div><div id="presets" class="seg"></div>
<div class="label"><span>Aspect ratio</span><span id="sizeHint" class="hint"></span></div><div id="ratios" class="ratio"></div>
<div class="label"><span>Seed</span><span class="hint"><label><input id="random" type="checkbox" checked style="width:auto"> random each run</label></span></div><input id="seed" type="number" min="0" step="1" value="20260921" disabled>
<details class="advanced"><summary>Advanced parameters</summary>
<div class="row"><div><div class="label"><span>Steps</span><span id="stepsGuide" class="hint"></span></div><input id="steps" type="number" min="1" max="60"><div id="stepsEffect" class="hint" style="margin-top:6px"></div></div><div><div class="label"><span>CFG</span><span id="cfgGuide" class="hint"></span></div><input id="cfg" type="number" min="0" max="20" step="0.1"><div id="cfgEffect" class="hint" style="margin-top:6px"></div></div></div>
<div class="row"><div><div class="label"><span>Width</span></div><input id="width" type="number" min="256" max="1536" step="8"></div><div><div class="label"><span>Height</span></div><input id="height" type="number" min="256" max="1536" step="8"></div></div>
<div class="row"><div><div class="label"><span>Sampler</span><span id="samplerGuide" class="hint"></span></div><select id="sampler"></select><div id="samplerEffect" class="hint" style="margin-top:6px"></div></div><div><div class="label"><span>Scheduler</span><span id="schedulerGuide" class="hint"></span></div><select id="scheduler"></select><div id="schedulerEffect" class="hint" style="margin-top:6px"></div></div></div>
<details style="margin-top:10px"><summary>Sampler and scheduler guide</summary><div id="optionReference" class="advice" style="margin-bottom:0"></div></details>
<div id="negativeWrap"><div class="label"><span>Negative prompt</span><span id="negativeCount" class="hint">0 / 4000 · optional</span></div><textarea id="negative" maxlength="4000" style="min-height:70px" placeholder="What should be avoided"></textarea></div>
<button id="reset" class="ghost" style="margin-top:10px;width:100%">Reset recommended parameters</button>
</details>
<div id="advice" class="advice"></div><button id="generate" class="generate">Generate</button><button id="cancel" class="cancel hidden">Cancel current run</button><div id="error" class="error"></div>
</section><section class="viewer"><div id="stage" class="stage">Ready for a new image</div><div id="progress" class="progress hidden" role="progressbar" aria-label="Generation progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div id="progressFill"></div></div><button id="returnCurrent" class="ghost hidden" style="margin:0 0 12px">← Back to current generation</button><div id="canvas" class="canvas"><div class="empty"><b>Your result appears here</b>Parameters remain editable for the next run while this model works.</div></div><div id="meta" class="meta"></div><div id="resultPath" class="result-path hidden"><code id="pathText"></code><button id="copyPath">Copy path</button><a id="openWorkflow" class="ghost hidden" target="_blank" rel="noopener" style="text-decoration:none;white-space:nowrap">Open this workflow ↗</a></div><div class="history-toolbar"><strong>History <span id="historyCount" class="hint"></span></strong><button id="selectAll">Select all</button><button id="deleteSelected" class="danger" disabled>Delete selected</button><button id="deleteAll" class="danger">Delete all</button></div><div id="history" class="history"></div></section></main>
<script>
const $=id=>document.getElementById(id);let config,currentJob=null,currentPreset="balanced",started=0,progressValue=null,progressStarted=0,stageName='Queued',historyItems=[],viewingHistory=false,livePreviewUrl=null,previewVersion=0,activeResult=null,activeParams=null,selectedViewId=null,saveTimer=null;
async function api(path,options){const r=await fetch(path,options);const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j}
function duration(seconds){seconds=Math.max(0,Math.round(seconds));if(seconds<60)return seconds+'s';const m=Math.floor(seconds/60),s=seconds%60;return m+'m '+s+'s'}
function setStage(name){stageName=name;const elapsed=(Date.now()-started)/1000;$('stage').innerHTML=`<span class="spinner"></span>${name} · ${duration(elapsed)} elapsed${name==='Decoding image'||name==='Saving image'?' · finishing':''}`}
function setProgress(value,max,diffusionElapsed=null){const now=Date.now();if(diffusionElapsed!==null)progressStarted=now-diffusionElapsed*1000;else if(!progressStarted)progressStarted=now;const pct=max?Math.max(0,Math.min(100,value/max*100)):0,elapsed=(now-started)/1000,sampling=(now-progressStarted)/1000,remaining=value>0?sampling/value*(max-value):0;progressValue=pct;stageName='Diffusing';$('progress').classList.remove('hidden');$('progressFill').style.width=pct+'%';$('progress').setAttribute('aria-valuenow',String(Math.round(pct)));$('stage').innerHTML=`<span class="spinner"></span>Diffusing · step ${value} / ${max} · ${Math.round(pct)}% · ${duration(elapsed)} elapsed · about ${duration(remaining)} left`}
function renderPreview(){if(livePreviewUrl)$('canvas').innerHTML=`<img src="${livePreviewUrl}" alt="Live diffusion preview"><div class="preview-label">Live preview · still generating</div>`;else $('canvas').innerHTML='<div class="empty"><b>Current generation is preparing</b>The first diffusion preview will appear here.</div>';$('meta').innerHTML='';$('resultPath').classList.add('hidden')}
function updateGuides(){for(const name of ['steps','cfg']){const value=Number($(name).value),guide=config.parameter_guides[name].find(x=>value<=x.max)||config.parameter_guides[name].at(-1);$(name+'Guide').textContent=guide.label;$(name+'Effect').textContent=guide.text}for(const name of ['sampler','scheduler']){const guide=config.option_guides[name][$(name).value];$(name+'Guide').textContent=guide.label;$(name+'Effect').textContent=guide.text}updateNegative()}
function optionList(el,values){el.innerHTML=values.map(x=>`<option value="${x}">${x}</option>`).join('')}
function selectPreset(name){const p=config.presets[name];currentPreset=name;[...$('presets').children].forEach(b=>b.classList.toggle('active',b.dataset.name===name));$('steps').value=p.steps;$('cfg').value=p.cfg;$('sampler').value=p.sampler;$('scheduler').value=p.scheduler;$('presetHint').textContent=p.label;$('advice').innerHTML=`<strong>${p.label}: ${p.steps} steps · CFG ${p.cfg}</strong>${p.advice}`;updateGuides()}
function markCustom(){currentPreset='custom';[...$('presets').children].forEach(b=>b.classList.remove('active'));$('presetHint').textContent='Custom';$('advice').innerHTML=`<strong>Custom parameters</strong>${config.parameter_advice}`;updateGuides()}
function setSize(w,h,label){$('width').value=w;$('height').value=h;$('sizeHint').textContent=`${w} × ${h}`;[...$('ratios').children].forEach(b=>b.classList.toggle('active',b.dataset.label===label))}
function updateNegative(){$('negativeWrap').classList.toggle('hidden',config.hide_negative_at_cfg_one&&Number($('cfg').value)<=1)}
function updateCounts(){for(const name of ['prompt','negative']){const count=$(name).value.length,out=$(name+'Count');out.textContent=`${count} / 4000${name==='negative'?' · optional':''}`;out.style.color=count>=3800?'var(--bad)':''}}
function params(){return{prompt:$('prompt').value.trim(),negative_prompt:$('negative').value.trim(),width:Number($('width').value),height:Number($('height').value),steps:Number($('steps').value),cfg:Number($('cfg').value),sampler:$('sampler').value,scheduler:$('scheduler').value,seed:$('random').checked?Math.floor(Math.random()*4294967295):Number($('seed').value),preset:currentPreset}}
function stateParams(){const p=params();if($('random').checked)p.seed=Number($('seed').value);p.random_seed=$('random').checked;return p}
async function saveUI(){saveTimer=null;try{await api('/api/ui-state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({params:stateParams(),current_job:currentJob,view_id:selectedViewId})})}catch(_){}}
function scheduleSave(){clearTimeout(saveTimer);saveTimer=setTimeout(saveUI,250)}
function showResult(item){selectedViewId=item.id;$('canvas').innerHTML=`<img src="${item.image_url}?t=${Date.now()}" alt="Generated image">`;$('meta').innerHTML=`<span>${item.width}×${item.height}</span><span>${item.steps} steps</span><span>CFG ${item.cfg}</span><span>Seed ${item.seed}</span><span>${item.seconds.toFixed(1)} s</span>`;$('pathText').textContent=item.image_path;$('resultPath').classList.remove('hidden');$('openWorkflow').classList.toggle('hidden',!item.workflow_url);if(item.workflow_url)$('openWorkflow').href=item.workflow_url}
function escapeHTML(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function selectedIds(){return [...document.querySelectorAll('.history-check:checked')].map(x=>x.value)}
function updateHistoryButtons(){$('deleteSelected').disabled=selectedIds().length===0;$('selectAll').textContent=historyItems.length&&selectedIds().length===historyItems.length?'Clear selection':'Select all';$('deleteAll').disabled=historyItems.length===0}
function applyParameters(x){$('prompt').value=x.prompt;$('negative').value=x.negative_prompt||'';$('steps').value=x.steps;$('cfg').value=x.cfg;$('width').value=x.width;$('height').value=x.height;$('seed').value=x.seed;$('random').checked=false;$('seed').disabled=false;$('sampler').value=x.sampler;$('scheduler').value=x.scheduler;markCustom();updateCounts()}
function restoreParameters(x){applyParameters(x);$('random').checked=Boolean(x.random_seed);$('seed').disabled=$('random').checked;currentPreset=x.preset||'custom';[...$('presets').children].forEach(b=>b.classList.toggle('active',b.dataset.name===currentPreset));if(config.presets[currentPreset]){$('presetHint').textContent=config.presets[currentPreset].label;$('advice').innerHTML=`<strong>${config.presets[currentPreset].label}: ${x.steps} steps · CFG ${x.cfg}</strong>${config.presets[currentPreset].advice}`}updateGuides()}
function reuseHistory(x){if(currentJob||activeResult){viewingHistory=true;$('returnCurrent').classList.remove('hidden');$('returnCurrent').textContent=currentJob?'← Back to current generation':'← View newest result'}showResult(x);applyParameters(x);scheduleSave()}
async function removeHistory(ids,all=false){await api('/api/history/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids,all})});if(activeResult&&(all||ids.includes(activeResult.id))){activeResult=null;$('returnCurrent').classList.add('hidden')}await loadHistory()}
async function loadHistory(){const j=await api('/api/history');historyItems=j.items;$('historyCount').textContent=`(${historyItems.length})`;$('history').innerHTML=historyItems.map((x,i)=>`<div class="thumb" data-i="${i}"><label class="select" title="Select"><input class="history-check" type="checkbox" value="${escapeHTML(x.id)}"></label><button class="delete-one" data-id="${escapeHTML(x.id)}" title="Delete this result">Delete</button><img src="${x.image_url}" alt="Generated result"><span class="prompt" title="${escapeHTML(x.prompt)}">${escapeHTML(x.prompt)}</span><small>${x.width}×${x.height} · ${x.steps} steps · ${x.seconds.toFixed(0)}s</small></div>`).join('');[...$('history').children].forEach(el=>el.onclick=e=>{if(e.target.closest('button,label,input'))return;reuseHistory(historyItems[Number(el.dataset.i)])});document.querySelectorAll('.history-check').forEach(x=>x.onchange=updateHistoryButtons);document.querySelectorAll('.delete-one').forEach(x=>x.onclick=()=>removeHistory([x.dataset.id]));updateHistoryButtons()}
async function poll(){if(!currentJob)return;try{const j=await api('/api/jobs/'+currentJob);if(j.progress)setProgress(Number(j.progress.value),Number(j.progress.max),Number(j.progress.elapsed_seconds||0));else if(j.stage)setStage(j.stage);else if(progressValue===null)setStage(j.status==='queued'?'Queued':'Preparing model');if(j.preview_version&&j.preview_version!==previewVersion){previewVersion=j.preview_version;livePreviewUrl=`/api/jobs/${currentJob}/preview?v=${previewVersion}`;if(!viewingHistory)renderPreview()}if(j.status==='complete'){activeResult=j;livePreviewUrl=null;if(!viewingHistory)showResult(j);else $('returnCurrent').textContent='← View newest result';setProgress(1,1,1);finish('Completed · 100%');await loadHistory();return}if(j.status==='error'){throw new Error(j.error||'Generation failed')}setTimeout(poll,900)}catch(e){finish('Failed');$('error').textContent=e.message}}
function finish(label){currentJob=null;$('generate').disabled=false;$('cancel').classList.add('hidden');$('ready').textContent='● Ready';$('stage').textContent=label;if(!label.startsWith('Completed'))$('progress').classList.add('hidden');scheduleSave()}
async function generate(){try{$('error').textContent='';const p=params();if(!p.prompt)throw new Error('Enter a prompt first.');activeParams={...p};$('generate').disabled=true;$('cancel').classList.remove('hidden');$('ready').textContent='● Working';started=Date.now();progressValue=null;progressStarted=0;stageName='Queued';viewingHistory=false;activeResult=null;selectedViewId=null;livePreviewUrl=null;previewVersion=0;$('returnCurrent').classList.add('hidden');$('progress').classList.remove('hidden');$('progressFill').style.width='0%';$('progress').setAttribute('aria-valuenow','0');$('resultPath').classList.add('hidden');$('meta').innerHTML='';renderPreview();const j=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});currentJob=j.id;await saveUI();setStage('Queued');poll()}catch(e){finish('Ready');$('error').textContent=e.message}}
async function init(){config=await api('/api/config');$('model').textContent=config.label;$('comfy').href=config.comfy_url;optionList($('sampler'),config.samplers);optionList($('scheduler'),config.schedulers);$('optionReference').innerHTML=['sampler','scheduler'].map(name=>`<strong>${name[0].toUpperCase()+name.slice(1)}</strong>`+Object.entries(config.option_guides[name]).map(([key,value])=>`<div style="margin:5px 0"><b>${key}</b> · ${value.label} — ${value.text}</div>`).join('')).join('<br>');$('presets').innerHTML=Object.entries(config.presets).map(([k,v])=>`<button data-name="${k}">${v.short}</button>`).join('');[...$('presets').children].forEach(b=>b.onclick=()=>{selectPreset(b.dataset.name);scheduleSave()});$('ratios').innerHTML=config.sizes.map(s=>`<button data-label="${s.label}">${s.label}</button>`).join('');[...$('ratios').children].forEach((b,i)=>b.onclick=()=>{const s=config.sizes[i];setSize(s.width,s.height,s.label);scheduleSave()});selectPreset('balanced');const s=config.sizes[0];setSize(s.width,s.height,s.label);$('prompt').value=config.example_prompt;updateCounts();const saved=await api('/api/ui-state');if(saved.params)restoreParameters(saved.params);await loadHistory();if(saved.view_id){const item=historyItems.find(x=>x.id===saved.view_id);if(item){showResult(item);activeResult=item}}if(saved.current_job){try{const job=await api('/api/jobs/'+saved.current_job);if(job.status!=='complete'&&job.status!=='error'){currentJob=saved.current_job;activeParams=saved.params||null;started=Date.now()-Number(job.elapsed_seconds||0)*1000;$('generate').disabled=true;$('cancel').classList.remove('hidden');$('ready').textContent='● Working';renderPreview();poll()}else if(job.status==='complete'){showResult(job);activeResult=job}}catch(_){scheduleSave()}}}
$('generate').onclick=generate;$('cancel').onclick=async()=>{if(currentJob)await api('/api/jobs/'+currentJob+'/cancel',{method:'POST'});finish('Cancelled')};$('random').onchange=()=>{$('seed').disabled=$('random').checked;scheduleSave()};$('reset').onclick=()=>{selectPreset('balanced');scheduleSave()};['steps','cfg'].forEach(id=>$(id).oninput=()=>{markCustom();scheduleSave()});['width','height','sampler','scheduler'].forEach(id=>$(id).onchange=()=>{markCustom();scheduleSave()});init().catch(e=>$('error').textContent=e.message);
$('copyPath').onclick=async()=>{await navigator.clipboard.writeText($('pathText').textContent);$('copyPath').textContent='Copied';setTimeout(()=>$('copyPath').textContent='Copy path',1200)};
$('prompt').oninput=()=>{updateCounts();scheduleSave()};$('negative').oninput=()=>{updateCounts();scheduleSave()};$('seed').oninput=scheduleSave;
$('returnCurrent').onclick=()=>{viewingHistory=false;$('returnCurrent').classList.add('hidden');if(activeParams)applyParameters(activeParams);if(currentJob){selectedViewId=null;renderPreview()}else if(activeResult)showResult(activeResult);scheduleSave()};
$('selectAll').onclick=()=>{const checks=[...document.querySelectorAll('.history-check')],all=checks.length&&checks.every(x=>x.checked);checks.forEach(x=>x.checked=!all);updateHistoryButtons()};$('deleteSelected').onclick=()=>removeHistory(selectedIds());$('deleteAll').onclick=()=>{if(historyItems.length)removeHistory([],true)};
</script></body></html>'''


class ImageUIState:
    def __init__(self, backend_port: int, comfy_url: str, model_key: str, model: dict,
                 paths: dict[str, Path], output_directory: Path, workflow_builder,
                 history_path: Path, job_workflow_builder=None):
        self.backend_port = backend_port
        self.comfy_url = comfy_url
        self.model_key = model_key
        self.model = model
        self.paths = paths
        self.output_directory = output_directory.resolve()
        self.workflow_builder = workflow_builder
        self.job_workflow_builder = job_workflow_builder
        self.history_path = history_path
        self.ui_state_path = history_path.with_name("image-ui-state.json")
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def backend_json(self, path: str, payload=None, method=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.backend_port}{path}", data=data,
            headers={"Content-Type": "application/json"}, method=method,
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def config(self):
        stage_nodes = ({
            "1": "Loading diffusion model", "2": "Loading text encoder",
            "3": "Loading VAE", "4": "Encoding prompt", "5": "Encoding prompt",
            "6": "Preparing latent", "7": "Preparing noise", "8": "Preparing sampler",
            "9": "Preparing schedule", "10": "Preparing guidance",
            "11": "Diffusing", "12": "Decoding image", "13": "Saving image",
        } if self.model_key == "flux2" else {
            "1": "Loading diffusion model", "2": "Loading text encoder",
            "3": "Loading VAE", "4": "Encoding prompt", "5": "Diffusing",
            "6": "Decoding image", "7": "Saving image",
        })
        return {
            "model": self.model_key, "label": self.model["short_label"],
            "comfy_url": self.comfy_url,
            "presets": self.model["presets"],
            "sizes": self.model["sizes"], "samplers": self.model["samplers"],
            "schedulers": self.model["schedulers"],
            "parameter_advice": self.model["parameter_advice"],
            "parameter_guides": self.model["parameter_guides"],
            "option_guides": self.model["option_guides"],
            "stage_nodes": stage_nodes,
            "hide_negative_at_cfg_one": self.model_key == "qwen21",
            "example_prompt": "A small red fox sitting in a sunlit meadow, detailed natural photograph",
        }

    def validate(self, value):
        prompt = str(value.get("prompt", "")).strip()
        if not prompt or len(prompt) > 4000:
            raise ValueError("Prompt must contain 1 to 4000 characters.")
        result = {"prompt": prompt, "negative_prompt": str(value.get("negative_prompt", ""))[:4000]}
        for name, low, high in (("width", 256, 1536), ("height", 256, 1536),
                                ("steps", 1, 60)):
            number = int(value.get(name, 0))
            if not low <= number <= high or (name in ("width", "height") and number % 8):
                raise ValueError(f"{name} must be {low}-{high}" + (" in multiples of 8." if name != "steps" else "."))
            result[name] = number
        if self.model_key == "qwen21" and result["width"] != result["height"]:
            raise ValueError("Qwen-Image-2.1 currently requires a square size.")
        result["cfg"] = float(value.get("cfg", self.model["cfg"]))
        if not 0 <= result["cfg"] <= 20:
            raise ValueError("cfg must be between 0 and 20.")
        result["seed"] = int(value.get("seed", 0))
        if not 0 <= result["seed"] <= 2**64 - 1:
            raise ValueError("seed is outside the supported range.")
        for name in ("sampler", "scheduler"):
            result[name] = str(value.get(name, ""))
            if result[name] not in self.model[name + "s"]:
                raise ValueError(f"Unsupported {name}: {result[name]}")
        result["preset"] = str(value.get("preset", "custom"))
        return result

    def submit(self, raw):
        params = self.validate(raw)
        graph = self.workflow_builder(self.model_key, self.paths, **params)
        editable_workflow = (self.job_workflow_builder(self.model_key, self.paths, **params)
                             if self.job_workflow_builder else None)
        client_id = uuid.uuid4().hex
        job_id = str(uuid.uuid4())
        with self.lock:
            self.jobs[job_id] = {**params, "id": job_id, "status": "queued", "started": time.monotonic()}
        self.start_progress_monitor(job_id, client_id)
        try:
            payload = {"prompt": graph, "client_id": client_id, "prompt_id": job_id}
            if editable_workflow:
                payload["extra_data"] = {"extra_pnginfo": {"workflow": editable_workflow}}
            response = self.backend_json("/prompt", payload)
        except Exception:
            with self.lock:
                self.jobs.pop(job_id, None)
            raise
        if response["prompt_id"] != job_id:
            raise ValueError("ComfyUI returned an unexpected job id.")
        return job_id

    def start_progress_monitor(self, job_id, client_id):
        progress_socket, initial = self.open_progress_socket(client_id)
        thread = threading.Thread(
            target=self.monitor_progress, args=(job_id, progress_socket, initial), daemon=True)
        thread.start()

    def open_progress_socket(self, client_id):
        progress_socket = socket.create_connection(("127.0.0.1", self.backend_port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET /ws?clientId={urllib.parse.quote(client_id)} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.backend_port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
        progress_socket.sendall(request)
        response = b""
        while b"\r\n\r\n" not in response and len(response) < 65536:
            block = progress_socket.recv(4096)
            if not block:
                break
            response += block
        if not response.startswith(b"HTTP/1.1 101"):
            progress_socket.close()
            raise OSError("ComfyUI rejected the progress connection.")
        headers, initial = response.split(b"\r\n\r\n", 1)
        expected = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
        ).digest()).decode("ascii")
        if f"sec-websocket-accept: {expected}".lower() not in headers.decode("latin1").lower():
            progress_socket.close()
            raise OSError("ComfyUI returned an invalid progress connection.")
        progress_socket.settimeout(None)
        return progress_socket, initial

    @staticmethod
    def websocket_frame(progress_socket, buffered):
        def take(length):
            nonlocal buffered
            while len(buffered) < length:
                block = progress_socket.recv(max(4096, length - len(buffered)))
                if not block:
                    raise EOFError
                buffered += block
            result, buffered = buffered[:length], buffered[length:]
            return result

        header = take(2)
        opcode = header[0] & 0x0F
        length = header[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", take(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", take(8))[0]
        mask = take(4) if header[1] & 0x80 else None
        payload = take(length)
        if mask:
            payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        return opcode, payload, buffered

    def monitor_progress(self, job_id, progress_socket, buffered=b""):
        try:
            while True:
                opcode, payload, buffered = self.websocket_frame(progress_socket, buffered)
                if opcode == 8:
                    break
                if opcode == 2 and len(payload) > 8 and struct.unpack(">I", payload[:4])[0] == 1:
                    image_type = struct.unpack(">I", payload[4:8])[0]
                    with self.lock:
                        job = self.jobs.get(job_id)
                        if job:
                            job["_preview"] = payload[8:]
                            job["_preview_mime"] = "image/png" if image_type == 2 else "image/jpeg"
                            job["preview_version"] = job.get("preview_version", 0) + 1
                    continue
                if opcode != 1:
                    continue
                message = json.loads(payload.decode("utf-8"))
                data = message.get("data") or {}
                if data.get("prompt_id") not in (None, job_id):
                    continue
                event = message.get("type")
                with self.lock:
                    job = self.jobs.get(job_id)
                    if not job:
                        break
                    if event == "progress":
                        diffusion_started = job.setdefault("_diffusion_started", time.monotonic())
                        job.update(status="running", stage="Diffusing",
                                   progress={"value": data.get("value", 0), "max": data.get("max", 0),
                                             "elapsed_seconds": round(time.monotonic() - diffusion_started, 2)})
                    elif event == "executing" and data.get("node") is not None:
                        job["status"] = "running"
                        job["stage"] = self.config()["stage_nodes"].get(str(data["node"]), "Preparing")
                        if job["stage"] == "Diffusing":
                            job.setdefault("_diffusion_started", time.monotonic())
                    elif event in ("execution_error", "execution_interrupted"):
                        job.update(status="error", error="Generation was interrupted." if event.endswith("interrupted") else "ComfyUI reported a workflow error.")
                        break
                    elif event == "executing" and data.get("node") is None:
                        break
        except (EOFError, OSError, ValueError, json.JSONDecodeError):
            pass
        finally:
            progress_socket.close()

    @staticmethod
    def public_job(job):
        return {key: value for key, value in job.items() if not key.startswith("_")}

    def ui_state(self):
        try:
            value = json.loads(self.ui_state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def save_ui_state(self, value):
        params = value.get("params") if isinstance(value.get("params"), dict) else {}
        clean_params = {
            key: params[key] for key in (
                "prompt", "negative_prompt", "width", "height", "steps", "cfg",
                "sampler", "scheduler", "seed", "preset", "random_seed")
            if key in params
        }
        if len(str(clean_params.get("prompt", ""))) > 4000 or len(str(clean_params.get("negative_prompt", ""))) > 4000:
            raise ValueError("Saved prompts may not exceed 4000 characters.")
        clean = {
            "params": clean_params,
            "current_job": str(value.get("current_job"))[:64] if value.get("current_job") else None,
            "view_id": str(value.get("view_id"))[:128] if value.get("view_id") else None,
        }
        self.ui_state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.ui_state_path.write_text(
                json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return clean

    def preview(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id) or {}
            data = job.get("_preview")
            mime = job.get("_preview_mime", "image/jpeg")
        if not data:
            raise KeyError(job_id)
        return data, mime

    def job(self, job_id):
        with self.lock:
            job = dict(self.jobs.get(job_id) or {})
        if not job:
            raise KeyError(job_id)
        item = self.backend_json(f"/api/jobs/{urllib.parse.quote(job_id)}")
        workflow_id = item.get("workflow_id")
        if workflow_id:
            job["workflow_id"] = str(workflow_id)
            job["workflow_url"] = f"{self.comfy_url.rstrip('/')}/#{urllib.parse.quote(str(workflow_id), safe='')}"
        backend_status = item.get("status")
        if backend_status in ("pending", "in_progress"):
            job["status"] = "queued" if backend_status == "pending" else "running"
            job["elapsed_seconds"] = round(time.monotonic() - job["started"], 2)
            return self.public_job(job)
        if backend_status in ("failed", "cancelled"):
            job.update(status="error", error="Generation was cancelled." if backend_status == "cancelled"
                       else "ComfyUI reported a workflow error.")
            return self.public_job(job)
        for output in item.get("outputs", {}).values():
            images = output.get("images", [])
            if images:
                image = images[0]
                path = (self.output_directory / image.get("subfolder", "") / image["filename"]).resolve()
                path.relative_to(self.output_directory)
                job.update(status="complete", seconds=round(time.monotonic() - job["started"], 2),
                           completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           image_path=str(path), image_url=self.image_url(path))
                self.complete(job_id, job)
                return self.public_job(job)
        job["status"] = "running"
        return self.public_job(job)

    def complete(self, job_id, job):
        with self.lock:
            if self.jobs[job_id].get("status") == "complete":
                return
            clean_job = self.public_job(job)
            self.jobs[job_id] = clean_job
            history = [item for item in self.history()
                       if item.get("image_path") != job.get("image_path")]
            history.insert(0, {k: v for k, v in clean_job.items() if k not in ("started",)})
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")

    def history(self):
        try:
            value = json.loads(self.history_path.read_text(encoding="utf-8"))
            history = value if isinstance(value, list) else []
        except (OSError, ValueError, json.JSONDecodeError):
            history = []
        for item in history:
            self.add_workflow_link(item)
        known = {str(Path(item.get("image_path", "")).resolve()) for item in history}
        discovered = []
        if self.output_directory.is_dir():
            for path in self.output_directory.rglob("*.png"):
                resolved = str(path.resolve())
                if resolved not in known:
                    discovered.append(self.discover_image(path))
        discovered.sort(key=lambda item: item["completed_at"], reverse=True)
        return history + discovered

    def add_workflow_link(self, item: dict) -> dict:
        """Recover a direct ComfyUI workflow link from the PNG when history predates Jobs."""
        if item.get("workflow_url"):
            return item
        try:
            metadata, _ = self.png_text(Path(item["image_path"]))
            workflow = json.loads(metadata.get("workflow", "{}"))
            workflow_id = workflow.get("id") if isinstance(workflow, dict) else None
            if workflow_id:
                item["workflow_id"] = str(workflow_id)
                item["workflow_url"] = f"{self.comfy_url.rstrip('/')}/#{urllib.parse.quote(str(workflow_id), safe='')}"
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return item

    @staticmethod
    def png_text(path: Path) -> tuple[dict[str, str], tuple[int, int]]:
        text = {}
        size = (0, 0)
        with path.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                return text, size
            while True:
                header = stream.read(8)
                if len(header) != 8:
                    break
                length, kind = struct.unpack(">I4s", header)
                data = stream.read(length)
                stream.read(4)
                if kind == b"IHDR" and len(data) >= 8:
                    size = struct.unpack(">II", data[:8])
                elif kind == b"tEXt" and b"\0" in data:
                    key, value = data.split(b"\0", 1)
                    text[key.decode("latin1")] = value.decode("utf-8", "replace")
                if kind == b"IEND":
                    break
        return text, size

    def discover_image(self, path: Path) -> dict:
        metadata, image_size = self.png_text(path)
        try:
            graph = json.loads(metadata.get("prompt", "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            graph = {}
        nodes = graph if isinstance(graph, dict) else {}

        def first(node_type):
            return next((node for node in nodes.values()
                         if node.get("class_type") == node_type), {})

        sampler = first("KSampler")
        schedule = first("Flux2Scheduler")
        guider = first("CFGGuider")
        noise = first("RandomNoise")
        latent = first("EmptyFlux2LatentImage")
        qwen_text = first("TextEncodeQwenImage21")
        sampler_select = first("KSamplerSelect")
        positive = qwen_text.get("inputs", {}).get("prompt", "")
        negative = qwen_text.get("inputs", {}).get("negative_prompt", "")
        if not positive and guider:
            positive_id = str(guider.get("inputs", {}).get("positive", [""])[0])
            negative_id = str(guider.get("inputs", {}).get("negative", [""])[0])
            positive = nodes.get(positive_id, {}).get("inputs", {}).get("text", "")
            negative = nodes.get(negative_id, {}).get("inputs", {}).get("text", "")
        sampler_inputs = sampler.get("inputs", {})
        schedule_inputs = schedule.get("inputs", {})
        latent_inputs = latent.get("inputs", {})
        resolution = qwen_text.get("inputs", {}).get("resolution", 0)
        relative = path.resolve().relative_to(self.output_directory).as_posix()
        item = {
            "id": "discovered-" + hashlib.sha256(relative.encode()).hexdigest()[:16],
            "status": "complete", "prompt": positive or "Recovered image",
            "negative_prompt": negative, "width": int(latent_inputs.get("width") or resolution or image_size[0]),
            "height": int(latent_inputs.get("height") or resolution or image_size[1]),
            "steps": int(sampler_inputs.get("steps") or schedule_inputs.get("steps") or 0),
            "cfg": float(sampler_inputs.get("cfg") or guider.get("inputs", {}).get("cfg") or 0),
            "seed": int(sampler_inputs.get("seed") or noise.get("inputs", {}).get("noise_seed") or 0),
            "sampler": sampler_inputs.get("sampler_name") or sampler_select.get("inputs", {}).get("sampler_name") or "unknown",
            "scheduler": sampler_inputs.get("scheduler") or ("flux2" if schedule else "unknown"),
            "preset": "recovered", "seconds": 0.0,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
            "image_path": str(path.resolve()),
        }
        return self.add_workflow_link(item)

    def image_url(self, path):
        relative = Path(path).resolve().relative_to(self.output_directory)
        return "/images/" + urllib.parse.quote(relative.as_posix())

    def delete_history(self, ids=None, *, delete_all=False):
        selected = set(ids or [])
        with self.lock:
            history = self.history()
            removed = history if delete_all else [item for item in history if item.get("id") in selected]
            remaining = [] if delete_all else [item for item in history if item.get("id") not in selected]
            for item in removed:
                try:
                    path = Path(item.get("image_path", "")).resolve()
                    path.relative_to(self.output_directory)
                    path.unlink(missing_ok=True)
                except (OSError, ValueError):
                    continue
            for directory in sorted(self.output_directory.rglob("*"), reverse=True):
                if directory.is_dir():
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(
                json.dumps(remaining, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if delete_all:
                self.ui_state_path.unlink(missing_ok=True)
        return {"deleted": len(removed), "remaining": len(remaining)}

    def cancel(self, job_id):
        try:
            return self.backend_json(f"/api/jobs/{urllib.parse.quote(job_id)}/cancel", {}, "POST")
        except urllib.error.HTTPError:
            return self.backend_json("/interrupt", {}, "POST")


def make_handler(state: ImageUIState):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def body(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1024 * 1024:
                raise ValueError("Request is too large.")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            try:
                if path == "/":
                    data = HTML.encode("utf-8"); self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
                elif path == "/api/config": self.send_json(state.config())
                elif path == "/api/status": self.send_json({"ok": True, "model": state.model_key})
                elif path == "/api/ui-state": self.send_json(state.ui_state())
                elif path == "/api/history":
                    items = []
                    for item in state.history():
                        candidate = Path(item.get("image_path", ""))
                        if candidate.is_file():
                            item = dict(item); item["image_url"] = state.image_url(candidate); items.append(item)
                    self.send_json({"items": items})
                elif path.startswith("/api/jobs/") and path.endswith("/preview"):
                    job_id = urllib.parse.unquote(path.split("/")[-2])
                    data, mime = state.preview(job_id); self.send_response(200)
                    self.send_header("Content-Type", mime); self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
                elif path.startswith("/api/jobs/"):
                    self.send_json(state.job(urllib.parse.unquote(path.rsplit("/", 1)[-1])))
                elif path.startswith("/images/"):
                    relative = urllib.parse.unquote(path[len("/images/"):])
                    image = (state.output_directory / relative).resolve(); image.relative_to(state.output_directory)
                    data = image.read_bytes(); self.send_response(200); self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
                else: self.send_json({"error": "Not found"}, 404)
            except KeyError: self.send_json({"error": "Unknown job"}, 404)
            except (OSError, ValueError, urllib.error.URLError) as error: self.send_json({"error": str(error)}, 400)

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            try:
                if path == "/api/generate": self.send_json({"id": state.submit(self.body())}, 202)
                elif path == "/api/ui-state": self.send_json(state.save_ui_state(self.body()))
                elif path == "/api/history/delete":
                    value = self.body()
                    ids = value.get("ids", [])
                    if not isinstance(ids, list):
                        raise ValueError("ids must be a list.")
                    self.send_json(state.delete_history(ids, delete_all=bool(value.get("all"))))
                elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    job_id = urllib.parse.unquote(path.split("/")[-2]); self.send_json(state.cancel(job_id))
                else: self.send_json({"error": "Not found"}, 404)
            except (KeyError, OSError, ValueError, urllib.error.URLError) as error:
                self.send_json({"error": str(error)}, 400)

        def log_message(self, fmt, *args):
            return
    return Handler


def start_server(port: int, state: ImageUIState):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
