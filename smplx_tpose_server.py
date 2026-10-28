#!/usr/bin/env python3
"""Interactive SMPL-X T-pose viewer with automatic gender-model discovery.

Inference runs on CUDA when a working GPU is detected at startup, and falls
back to the original CPU path otherwise (see :func:`resolve_device`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import numpy as np
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMPL-X T-pose 预览</title>
<style>
  :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
  * { box-sizing: border-box; } html,body,#app { margin:0; width:100%; height:100%; overflow:hidden; }
  body { background:#111318; color:#edf0f5; } #app { display:grid; grid-template-columns:300px minmax(0,1fr); grid-template-rows:100%; }
  /* the console scrolls on its own so a growing panel never resizes the canvas */
  aside { overflow-y:auto; min-height:0; }
  aside { background:#1a1e25; border-right:1px solid #303642; padding:24px 18px; display:flex; flex-direction:column; gap:22px; }
  h1 { font-size:18px; margin:0; } .sub { color:#aeb7c6; font-size:13px; line-height:1.55; margin:8px 0 0; }
  .section-title { display:block; color:#f4f6fa; font-size:16px; font-weight:700; margin:0; }
  .panel { border:1px solid #2b313c; border-radius:8px; background:#171b22; padding:0 12px 12px; }
  .panel > summary { list-style:none; cursor:pointer; user-select:none; padding:12px 0 10px; display:flex; align-items:center; justify-content:space-between; }
  .panel > summary::-webkit-details-marker { display:none; }
  .panel > summary::after { content:'\25BE'; color:#8f99a9; font-size:11px; transition:transform .15s ease; }
  .panel:not([open]) > summary::after { transform:rotate(-90deg); }
  .panel[open] > summary { margin-bottom:10px; }
  .panel:not([open]) { padding-bottom:0; }
  .brand h1 { margin:0; } select { width:100%; min-height:43px; border:1px solid #414957; border-radius:6px; background:#252b35; color:#eff2f7; padding:0 11px; font-size:14px; cursor:pointer; }
  button { border:1px solid #414957; border-radius:6px; background:#252b35; color:#eff2f7; min-height:43px; padding:0 12px; font-size:14px; text-align:left; cursor:pointer; }
  button:hover:not(:disabled) { border-color:#7da6ef; background:#293544; } button.active { color:#fff; background:#24559a; border-color:#70a5f3; }
  button:disabled { color:#697383; cursor:not-allowed; } #status { margin-top:auto; color:#aeb7c6; font-size:12px; line-height:1.55; }
  main { min-width:0; min-height:0; position:relative; } canvas { display:block; width:100%; height:100%; }
  .hud { position:absolute; top:18px; right:18px; background:#1a1e25e8; border:1px solid #343b47; border-radius:6px; padding:11px 13px; color:#b9c2d0; font-size:12px; line-height:1.55; }
  .pose-controls { display:flex; gap:8px; align-items:stretch; } .toggle { display:flex; align-items:center; gap:7px; min-height:38px; flex:1; padding:0 10px; border:1px solid #414957; border-radius:6px; background:#252b35; font-size:14px; cursor:pointer; white-space:nowrap; } .toggle input { width:16px; height:16px; accent-color:#70a5f3; } #reset { min-height:38px; padding:0 10px; font-size:13px; white-space:nowrap; }
  .segment-empty { color:#8f99a9; font-size:13px; } .segment-name { color:#d9e5f8; font-size:13px; margin:0 0 10px; } .field { display:grid; grid-template-columns:62px minmax(0,1fr) 48px; gap:7px; align-items:center; margin:8px 0; font-size:12px; color:#b9c3d0; } input[type=range] { width:100%; accent-color:#70a5f3; } .value { color:#f1f4f8; text-align:right; font-variant-numeric:tabular-nums; } .segment-tools { display:flex; gap:8px; align-items:center; margin:10px 0 4px; } .segment-tools select { flex:1; min-height:34px; font-size:13px; } .small { min-height:34px; padding:0 10px; font-size:12px; } .selected-bone { color:#fff1b3; }
  .error { color:#ffabb8 !important; }
  #toasts { position:fixed; right:18px; bottom:18px; display:flex; flex-direction:column; gap:8px; z-index:60; pointer-events:none; align-items:flex-end; }
  .toast { background:#f2c94c; color:#241c00; font-size:13px; line-height:1.45; padding:10px 14px; border-radius:6px; box-shadow:0 6px 18px rgba(0,0,0,.35); max-width:320px; transition:opacity .25s ease, transform .25s ease; }
  .toast.out { opacity:0; transform:translateY(6px); }
  button.locked { opacity:.45; cursor:not-allowed; }
  .skeleton-locked #seg-length { opacity:.5; }
  .skeleton-locked #seg-length-row { cursor:not-allowed; } .ref-row { display:flex; gap:8px; align-items:stretch; } #ref-pick { flex:1 1 auto; min-width:0; text-align:center; } #ref-hide, #ref-clear { flex:0 0 62px; min-width:62px; text-align:center; } #ref-file { display:none; } .ref-info { margin:8px 0 0; } @media(max-width:700px) { #app{grid-template-columns:1fr;grid-template-rows:auto minmax(0,1fr)} aside{padding:14px; border-right:0; border-bottom:1px solid #303642; gap:12px}.gender{grid-template-columns:repeat(3,1fr)} button{font-size:12px;text-align:center} #status{margin-top:0} }
</style>
</head>
<body><div id="app"><aside>
  <div><h1>SMPL-X T-pose</h1><p class="sub">选择性别模型。鼠标左键旋转，滚轮缩放，右键平移。</p></div>
  <details class="panel" open><summary class="section-title">人体模型</summary><select id="gender" aria-label="人体模型"></select></details>
  <details class="panel" open><summary class="section-title">调整人体姿态</summary><div class="pose-controls"><label class="toggle"><input id="full-body-ik" type="checkbox"> <span>全身 IK</span></label><button id="reset" disabled>重置当前姿态</button></div><p class="sub">关闭时移动所选点及其子节点，父节点保持固定；开启后允许整条父链共同求解。</p></details>
  <details class="panel" open><summary class="section-title">骨段编辑</summary><div id="segment-editor" class="segment-empty">点击一条黄色骨骼连线以编辑该骨段。</div></details>
  <details class="panel" open><summary class="section-title">参考图</summary><div class="ref-row"><button id="ref-pick">选择图片</button><button id="ref-hide" class="small" disabled>隐藏</button><button id="ref-clear" class="small" disabled>移除</button></div><input id="ref-file" type="file" accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp,image/gif,image/tiff,image/*"><p class="sub ref-info" id="ref-info">未选择图片。上传后自动用 AnimePoseDetector 对齐双眼。</p></details>
  <div id="status">正在读取模型信息...</div>
</aside><main><canvas id="viewer"></canvas><div class="hud">点击黄色身体关节后拖动<br>空白处拖动旋转 · 滚轮缩放</div><div id="toasts"></div></main></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const canvas=document.querySelector('#viewer'), status=document.querySelector('#status'), choices=document.querySelector('#gender');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setClearColor(0x111318);
const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(38,1,.01,100); camera.position.set(3.2,1.7,5.6);
const controls=new OrbitControls(camera,renderer.domElement); controls.target.set(0,.82,0); controls.enableDamping=true; controls.minDistance=1.2; controls.maxDistance=10;
scene.add(new THREE.HemisphereLight(0xddeaff,0x20242c,2.0)); const key=new THREE.DirectionalLight(0xffffff,2.1); key.position.set(3,5,4); scene.add(key);
const grid=new THREE.GridHelper(8,16,0x4a596f,0x27303b); scene.add(grid); const axes=new THREE.AxesHelper(.45); axes.position.set(-2.5,.01,2.5); scene.add(axes);
let meshRoot=null, skeletonRoot=null, selected=null, jointPickers=[], segmentLines=null, skeletonData=null, drag=null, selectedSegment=null, revision=0;
const raycaster=new THREE.Raycaster(), pointer=new THREE.Vector2(), dragPlane=new THREE.Plane();
function resize(){const r=canvas.getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix()} addEventListener('resize',resize); resize();
function render(){controls.update();renderer.render(scene,camera);requestAnimationFrame(render)} render();
function setStatus(text,err=false){status.textContent=text;status.className=err?'error':''}
function frame(root){const box=new THREE.Box3().setFromObject(root), c=box.getCenter(new THREE.Vector3()), s=box.getSize(new THREE.Vector3()); controls.target.set(c.x,c.y,c.z); camera.position.set(c.x,c.y+s.y*.10,c.z+s.y*2.05); controls.update();}
// ---------------------------- reference picture ----------------------------
// The picture lives in the xy plane.  AnimePoseDetector reports the eye
// midpoint in picture pixels plus a pixels-per-metre scale, so the plane is
// sized to human scale and translated until the picture's eye midpoint lands on
// the model's eye midpoint (SMPL-X joints 23/24).  The model is never edited.
let referenceRoot=null, referencePlane=null, referenceLayout=null, referenceURL=null;
let referenceHidden=false, referenceStatus='', groundOffset=0;
let referenceTargets=null, referenceToken=0, referenceEpoch=0;
let referenceSolvedKey=null, referenceSolveInfo=null, referenceLocked=false;
const EYE_LEFT_JOINT=23, EYE_RIGHT_JOINT=24;
function refreshReferenceInfo(){
  const info=document.querySelector('#ref-info');
  if(!info)return;
  if(!referenceRoot){info.textContent='未选择图片。上传后自动用 AnimePoseDetector 对齐双眼。';return;}
  const bits=[];
  if(referenceLayout){
    const L=referenceLayout;
    bits.push(`${L.label} · ${L.imgW}×${L.imgH} px · ${L.pxPerM.toFixed(1)} px/m`);
    if(referenceStatus)bits.push(referenceStatus);
    if(referenceSolveInfo){
      bits.push(`骨骼残差 ${referenceSolveInfo.rms.toFixed(1)} px（长度拟合 ${referenceSolveInfo.lengths.toFixed(1)} px）`);
      if(referenceSolveInfo.z!==undefined&&referenceSolveInfo.z!==null)bits.push(`z 漂移 ${(referenceSolveInfo.z*1000).toFixed(2)} mm`);
    }
  }else{
    bits.push('未检测到双眼，按 1 px = 1 mm 放在模型中心');
  }
  if(referenceHidden)bits.push('已隐藏');
  info.textContent=bits.join(' · ');
}
function updateReferenceButtons(){
  const hide=document.querySelector('#ref-hide'), clear=document.querySelector('#ref-clear');
  const has=!!referencePlane;
  if(hide){hide.disabled=!has;hide.classList.toggle('active',has&&referenceHidden);}
  if(clear)clear.disabled=!has;
}
function modelEyeWorld(){
  if(!skeletonData||!skeletonData.joints||skeletonData.joints.length<=EYE_RIGHT_JOINT)return null;
  const a=skeletonData.joints[EYE_LEFT_JOINT], b=skeletonData.joints[EYE_RIGHT_JOINT];
  return new THREE.Vector3((a[0]+b[0])/2,(a[1]+b[1])/2+groundOffset,(a[2]+b[2])/2);
}
function alignReference(){
  if(!referenceRoot||!referencePlane||!referenceLayout)return null;
  const eye=modelEyeWorld();
  if(!eye)return null;
  const L=referenceLayout, mpp=1/L.pxPerM;
  const localX=(L.eyePx[0]-L.imgW/2)*mpp;
  const localY=(L.imgH/2-L.eyePx[1])*mpp;
  // The plane keeps its XY orientation (no rotation at all); it is translated
  // along z as well so the picture's eye midpoint and the model's eye midpoint
  // coincide in 3D, not just in the projection.
  referencePlane.position.set(eye.x-localX,eye.y-localY, 0);
  referenceRoot.updateMatrixWorld(true);
  const probe=referencePlane.localToWorld(new THREE.Vector3(localX,localY,0));
  // the picture is pinned to the xy plane, so only the in-plane offset can be
  // driven to zero -- compare in xy, not in 3D (the eyes sit ~6 cm in front).
  return Math.hypot(probe.x-eye.x,probe.y-eye.y);
}
function realignReference(){
  if(!referenceRoot||!referenceLayout)return;
  const err=alignReference();
  if(err!==null)referenceStatus=`双眼对齐误差 ${(err*1000).toFixed(4)} mm（xy 平面内）`;
  refreshReferenceInfo();
}
function disposeReference(){
  if(referenceRoot){
    referenceRoot.traverse((object)=>{if(object.isMesh){object.geometry.dispose();if(object.material.map)object.material.map.dispose();object.material.dispose();}});
    scene.remove(referenceRoot);
  }
  if(referenceURL){URL.revokeObjectURL(referenceURL);referenceURL=null;}
  referenceRoot=null;referencePlane=null;referenceLayout=null;
  referenceHidden=false;referenceStatus='';
  referenceTargets=null;referenceSolvedKey=null;referenceSolveInfo=null;
  setReferenceLock(false);
}
function clearReference(){
  disposeReference();
  refreshReferenceInfo();
  updateReferenceButtons();
}
function setReferenceLock(locked){
  if(referenceLocked===locked)return;
  referenceLocked=locked;
  document.body.classList.toggle('skeleton-locked',locked);
  const reset=document.querySelector('#reset');
  if(reset)reset.classList.toggle('locked',locked);
  segmentEditor();
}
function pushToast(text){
  const box=document.querySelector('#toasts');
  if(!box)return;
  const toast=document.createElement('div');
  toast.className='toast';
  toast.textContent=text;
  box.appendChild(toast);
  while(box.children.length>3)box.removeChild(box.firstChild);   // at most 3 stacked
  setTimeout(()=>{toast.classList.add('out');setTimeout(()=>toast.remove(),260);},3000);
}
function warnLocked(){
  pushToast('骨架已处于锁定状态；请移除图片以继续编辑骨架');
}
// Signal hook for future head / eye editors: those move the eyes, so the whole
// bone-length fit has to be recomputed on the next load.  Call this from any
// panel that edits the head or the eyes.
function invalidateReferenceAlignment(){
  if(!referenceTargets)return;
  referenceEpoch+=1;
  referenceSolvedKey=null;
}
// Public signal hook: any panel that moves the head / eyes (or otherwise changes
// joint positions) calls this, and the next load re-fits the bone lengths.
window.smplxInvalidateReferenceAlignment=invalidateReferenceAlignment;
async function maybeSolveReference(){
  if(!referenceTargets||!selected)return;
  const key=referenceTargets.token+'|'+selected+'|'+referenceEpoch;
  if(referenceSolvedKey===key)return;
  referenceSolvedKey=key;
  setStatus('正在按参考图调整骨骼长度...');
  try{
    const response=await fetch('/api/reference/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      gender:selected,
      pixels_per_meter:referenceTargets.mpp,
      eye_midpoint:referenceTargets.eye,
      joints:referenceTargets.joints,
    })});
    if(!response.ok)throw Error((await response.text()).slice(0,160));
    const data=await response.json();
    await load(selected,true,true);
    referenceSolveInfo={rms:data.rms_px,max:data.max_px,count:data.num_joints,lengths:data.lengths_rms_px,moved:(data.pose_moved||[]).length,full:data.full_body,z:data.z_drift_m};
    refreshReferenceInfo();
    setStatus(`骨架已按参考图锁定（${data.num_joints} 关节；长度拟合 ${data.lengths_rms_px.toFixed(1)} px → ${data.full_body?'全身':'局部'} IK 后 ${data.rms_px.toFixed(1)} px / 最大 ${data.max_px.toFixed(1)} px）`);
  }catch(error){
    referenceSolveInfo=null;
    refreshReferenceInfo();
    setStatus('参考图骨骼对齐失败：'+error.message,true);
  }
}
async function showReferenceFile(file){
  const name=file.name||'image';
  if(!/^image\//.test(file.type)&&!/\.(png|jpe?g|webp|bmp|gif|tiff?)$/i.test(name)){setStatus('不支持的图片格式：'+(file.type||name),true);return;}
  setStatus('正在用 AnimePoseDetector 分析参考图 '+name+' ...');
  const url=URL.createObjectURL(file);
  let image;
  try{
    image=await new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(Error('图片解码失败'));img.src=url;});
  }catch(error){URL.revokeObjectURL(url);setStatus('参考图载入失败：'+error.message,true);return;}
  let eyes=null, detectorError='';
  try{
    const response=await fetch('/api/reference',{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream'},body:file});
    if(!response.ok)throw Error((await response.text()).slice(0,160));
    eyes=await response.json();
  }catch(error){detectorError=error.message;}
  const imgW=eyes?eyes.image_width:image.naturalWidth;
  const imgH=eyes?eyes.image_height:image.naturalHeight;
  const pxPerM=eyes&&eyes.pixels_per_meter>1?eyes.pixels_per_meter:1000;
  const mpp=1/pxPerM;
  disposeReference();
  referenceURL=url;
  const texture=new THREE.Texture(image);
  texture.colorSpace=THREE.SRGBColorSpace;
  texture.anisotropy=renderer.capabilities.getMaxAnisotropy();
  texture.needsUpdate=true;
  const material=new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,depthWrite:false,alphaTest:0.001,toneMapped:false});
  const plane=new THREE.Mesh(new THREE.PlaneGeometry(imgW*mpp,imgH*mpp),material);
  plane.renderOrder=-10;          // drawn first, never hides the body
  referencePlane=plane;
  referenceRoot=new THREE.Group();
  referenceRoot.add(plane);
  scene.add(referenceRoot);
  if(eyes){
    referenceLayout={label:name,imgW:imgW,imgH:imgH,eyePx:eyes.eye_midpoint,pxPerM:pxPerM};
    referenceStatus='';
    referenceTargets={token:++referenceToken,mpp:pxPerM,eye:eyes.eye_midpoint,joints:eyes.joints||{}};
    referenceEpoch=0;
    referenceSolvedKey=null;      // a new picture must always be solved again
    referenceSolveInfo=null;
    realignReference();
    setReferenceLock(true);
    setStatus('已载入参考图并对齐双眼（joint 23/24）');
    maybeSolveReference();          // fit the bone lengths to the picture's joints
  }else{
    referenceLayout=null;
    const box=new THREE.Box3().setFromObject(meshRoot||scene), centre=box.getCenter(new THREE.Vector3());
    plane.position.set(centre.x,centre.y,centre.z);
    refreshReferenceInfo();
    setStatus('未检测到双眼，参考图按 1 px = 1 mm 放在模型中心：'+detectorError,true);
  }
  updateReferenceButtons();
}
function segmentEditor(){
  const host=document.querySelector('#segment-editor');
  if(selectedSegment===null||!skeletonData){host.className='segment-empty';host.textContent='点击一条黄色骨骼连线以编辑该骨段。';updateSelectionVisuals();return;}
  const child=selectedSegment,parent=skeletonData.parents[child];
  const cfg=skeletonData.segments[String(child)]||{length:1,radii:[1,1,1]};
  const names=skeletonData.names;
  host.className='';
  const lengthEnabled=(child===4||child===5||child===7||child===8||child===18||child===19||child===20||child===21);
  const lengthMin=Math.min(0.50,cfg.length), lengthMax=Math.max(1.80,cfg.length);
  const lengthControl=lengthEnabled?`<div class="field" id="seg-length-row"><span>长度</span><input id="seg-length" type="range" min="${lengthMin}" max="${lengthMax}" step="0.01" value="${cfg.length}" ${referenceLocked?'disabled':''}><span class="value" id="seg-length-value">${cfg.length.toFixed(2)}</span></div>`:`<p class="sub">当前骨段仅支持横截面调整。</p>`;
  host.innerHTML=`<p class="segment-name"><span class="selected-bone">${names[parent]} → ${names[child]}</span></p>${lengthControl}<div class="segment-tools"><span>横截面</span><select id="seg-count">${Array.from({length:11},(_,i)=>i+2).map(n=>`<option value="${n}" ${n===cfg.radii.length?'selected':''}>${n} 段</option>`).join('')}</select><button class="small" id="seg-reset">重置此段</button></div><div id="radius-fields"></div>`;
  const renderRadii=(radii)=>{
    const fields=document.querySelector('#radius-fields');
    fields.innerHTML=radii.map((v,i)=>`<div class="field"><span>${Math.round(i/(radii.length-1)*100)}%</span><input data-radius="${i}" type="range" min="0.3" max="2.5" step="0.01" value="${v}"><span class="value">${v.toFixed(2)}</span></div>`).join('');
    fields.querySelectorAll('input').forEach((input)=>{input.oninput=()=>{input.nextElementSibling.textContent=Number(input.value).toFixed(2);};input.onchange=()=>apply(currentRadii());});
  };
  const currentRadii=()=>[...document.querySelectorAll('[data-radius]')].map((input)=>Number(input.value));
  // locked: keep the exact fitted length, never the slider's 2-decimal rounding
  const currentLength=()=>referenceLocked?Number(cfg.length):(lengthEnabled?Number(document.querySelector('#seg-length').value):1);
  const apply=(radii)=>applySegment(child,currentLength(),radii);
  renderRadii(cfg.radii);
  if(referenceLocked){
    const row=document.querySelector('#seg-length-row');
    if(row)row.onclick=warnLocked;
  }
  if(lengthEnabled){const length=document.querySelector('#seg-length');length.oninput=()=>{document.querySelector('#seg-length-value').textContent=Number(length.value).toFixed(2);};length.onchange=()=>apply(currentRadii());}
  document.querySelector('#seg-count').onchange=(event)=>{
    const count=Number(event.target.value),old=currentRadii();
    const next=Array.from({length:count},(_,i)=>old[Math.min(old.length-1,Math.round(i/(count-1)*(old.length-1)))]);
    renderRadii(next);apply(next);
  };
  document.querySelector('#seg-reset').onclick=async()=>{
    if(referenceLocked){warnLocked();return;}
    setStatus('正在重置当前骨段...');
    try{
      const response=await fetch('/api/segment/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gender:selected,child})});
      if(!response.ok)throw Error(await response.text());
      const result=await response.json();
      if(!result.ok)throw Error('骨段重置未完成');
      await load(selected,true,true);
      selectedSegment=child;
      segmentEditor();
      setStatus('已重置当前骨段');
    }catch(error){setStatus('骨段重置失败：'+error.message,true);}
  };
  updateSelectionVisuals();
}
async function applySegment(child,length,radii){setStatus('正在更新骨段 Mesh...');try{const r=await fetch('/api/segment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gender:selected,child,length,radii})});if(!r.ok)throw Error(await r.text());await load(selected,true,true);selectedSegment=child;segmentEditor()}catch(e){setStatus('骨段更新失败：'+e.message,true)}}
// ---------------------------- bone selection highlight ---------------------
// Selecting a bone thickens it, tints its own skinned vertices and marks the
// interior division points of the current cross-section profile.
const MESH_BASE_COLOUR=new THREE.Color(0x9ab7df);
const MESH_SELECT_COLOUR=new THREE.Color(0xffff00);
const SELECTED_BONE_COLOUR=0xff6a3d;
const DIVIDER_COLOUR=0x5ce1ff;
let segmentOwners=null, segmentOwnersGender=null, selectionGroup=null;

function paintMesh(){
  if(!meshRoot)return;
  meshRoot.traverse((object)=>{
    if(!object.isMesh)return;
    const geometry=object.geometry, count=geometry.attributes.position.count;
    const colours=new Float32Array(count*3);
    const selected=(selectedSegment!==null&&segmentOwners&&segmentOwners.length===count)?selectedSegment:-1;
    for(let i=0;i<count;i++){
      const colour=(segmentOwners&&segmentOwners[i]===selected)?MESH_SELECT_COLOUR:MESH_BASE_COLOUR;
      colours[i*3]=colour.r;colours[i*3+1]=colour.g;colours[i*3+2]=colour.b;
    }
    geometry.setAttribute('color',new THREE.BufferAttribute(colours,3));
    object.material.vertexColors=true;
    object.material.color.set(0xffffff);
    object.material.needsUpdate=true;
  });
}
async function ensureOwners(gender){
  if(segmentOwners&&segmentOwnersGender===gender)return;
  segmentOwners=null;segmentOwnersGender=null;
  try{
    const response=await fetch('/api/weights/'+gender+'?revision='+(++revision),{cache:'no-store'});
    if(!response.ok)throw Error(await response.text());
    const owners=new Uint8Array(await response.arrayBuffer());
    if(!meshRoot)return;
    segmentOwners=owners;
    segmentOwnersGender=gender;
    paintMesh();
    updateSelectionVisuals();
  }catch(error){segmentOwners=null;}
}
function updateSelectionVisuals(){
  if(selectionGroup&&skeletonRoot)skeletonRoot.remove(selectionGroup);
  selectionGroup=null;
  paintMesh();
  if(selectedSegment===null||!skeletonData||!skeletonRoot)return;
  const child=selectedSegment, parent=skeletonData.parents[child];
  if(parent===undefined||parent<0)return;
  const a=new THREE.Vector3(...skeletonData.joints[parent]);
  const b=new THREE.Vector3(...skeletonData.joints[child]);
  const direction=new THREE.Vector3().subVectors(b,a);
  const length=direction.length();
  if(length<1e-6)return;
  const group=new THREE.Group();
  const bone=new THREE.Mesh(
    new THREE.CylinderGeometry(.005,.005,length,10,1,true),
    new THREE.MeshBasicMaterial({color:SELECTED_BONE_COLOUR,transparent:true,opacity:.95,depthTest:false,depthWrite:false,side:THREE.DoubleSide})
  );
  bone.position.copy(a).add(b).multiplyScalar(.5);
  bone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),direction.clone().normalize());
  bone.renderOrder=4;
  group.add(bone);
  const config=skeletonData.segments[String(child)];
  const divisions=Math.max(2,(config&&Array.isArray(config.radii))?config.radii.length:3);
  // transparent:true keeps the markers in the same (transparent) queue as the
  // highlighted bone, so renderOrder decides and they are drawn on top of it.
  const markerMaterial=new THREE.MeshBasicMaterial({color:DIVIDER_COLOUR,transparent:true,opacity:1,depthTest:false,depthWrite:false});
  for(let index=1;index<divisions-1;index+=1){
    const t=index/(divisions-1);
    const marker=new THREE.Mesh(new THREE.SphereGeometry(.004,12,8),markerMaterial);
    marker.position.lerpVectors(a,b,t);
    marker.renderOrder=5;
    group.add(marker);
  }
  selectionGroup=group;
  skeletonRoot.add(group);
}
function clearSegmentSelection(){
  if(selectedSegment===null)return;
  selectedSegment=null;
  segmentEditor();
  setStatus('已取消骨段选择');
}

function addSkeleton(data,groundOffsetY){
  if(skeletonRoot)scene.remove(skeletonRoot);
  skeletonData=data;
  skeletonRoot=new THREE.Group();
  skeletonRoot.position.y=groundOffsetY;
  jointPickers=[];
  const joints=data.joints.map((p)=>new THREE.Vector3(...p));
  const points=[],children=[];
  for(let i=0;i<data.parents.length;i++){const parent=data.parents[i];if(parent>=0&&parent<joints.length){points.push(joints[parent],joints[i]);children.push(i);}}
  const lineGeometry=new THREE.BufferGeometry().setFromPoints(points);
  segmentLines=new THREE.LineSegments(lineGeometry,new THREE.LineBasicMaterial({color:0xffc64d,depthTest:false,depthWrite:false,transparent:true,opacity:1}));
  segmentLines.userData.children=children;
  segmentLines.renderOrder=2;
  skeletonRoot.add(segmentLines);
  const jointMat=new THREE.MeshBasicMaterial({color:0xfff0a6,depthTest:false,depthWrite:false});
  for(let i=0;i<joints.length;i++){
    const dot=new THREE.Mesh(new THREE.SphereGeometry(.006,6,4),jointMat);
    dot.position.copy(joints[i]);
    dot.renderOrder=3;
    dot.userData.joint=i;
    skeletonRoot.add(dot);
    if(i>0&&i<=21)jointPickers.push(dot);
  }
  scene.add(skeletonRoot);
  segmentEditor();
  realignReference();
}
// OBJLoader expands a mesh into one vertex per triangle corner (3 x 20908 =
// 62724 vertices), which breaks the 1:1 mapping onto the 10475 SMPL-X skinning
// weights.  Parse the OBJ ourselves and keep it indexed so vertex index ==
// model vertex index -- that is what makes per-vertex skin tinting possible.
function parseObjGeometry(text){
  const positions=[], indices=[];
  const lines=text.split('\n');
  for(let i=0;i<lines.length;i+=1){
    const line=lines[i];
    if(line.charCodeAt(0)===118&&line.charCodeAt(1)===32){            // 'v x y z'
      const p=line.split(/\s+/);
      positions.push(+p[1],+p[2],+p[3]);
    }else if(line.charCodeAt(0)===102&&line.charCodeAt(1)===32){      // 'f a b c'
      const p=line.split(/\s+/);
      for(let k=1;k<p.length;k+=1){
        const slash=p[k].indexOf('/');
        indices.push(parseInt(slash<0?p[k]:p[k].slice(0,slash),10)-1);
      }
    }
  }
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

async function load(gender,edited=false,force=false){
  if(selected===gender&&!force)return;
  const shouldFrame=selected!==gender||!meshRoot;
  selected=gender;choices.value=gender;
  setStatus('正在加载 '+gender+' 模型...');
  try{
    const suffix='?revision='+(++revision);
    const [meshText,skeleton]=await Promise.all([
      fetch('/api/model/'+gender+suffix,{cache:'no-store'}).then((r)=>{if(!r.ok)throw Error('网格读取失败');return r.text();}),
      fetch('/api/skeleton/'+gender+suffix,{cache:'no-store'}).then((r)=>{if(!r.ok)throw Error('骨架数据读取失败');return r.json();})
    ]);
    if(meshRoot)scene.remove(meshRoot);
    const material=new THREE.MeshStandardMaterial({color:0xffffff,vertexColors:true,roughness:.72,metalness:.03,transparent:true,opacity:.42,depthWrite:false,side:THREE.DoubleSide});
    const obj=new THREE.Mesh(parseObjGeometry(meshText),material);
    meshRoot=obj;
    groundOffset=.02-new THREE.Box3().setFromObject(obj).min.y;
    obj.position.y=groundOffset;
    scene.add(obj);
    paintMesh();
    addSkeleton(skeleton,groundOffset);
    ensureOwners(gender);
    if(shouldFrame)frame(obj);
    document.querySelector('#reset').disabled=false;
    setStatus(referenceLocked?'已加载 '+gender+'（骨架已锁定，移除图片后可编辑）':'已加载 '+gender+'，可拖动黄色身体关节');
    maybeSolveReference();
  }catch(e){setStatus('加载失败：'+e.message,true);}
}
function updatePointer(event){const r=canvas.getBoundingClientRect();pointer.set((event.clientX-r.left)/r.width*2-1,-((event.clientY-r.top)/r.height)*2+1);raycaster.setFromCamera(pointer,camera)}
let gesture=null;
canvas.addEventListener('pointerdown',(event)=>{
  gesture={x:event.clientX,y:event.clientY,moved:false,boneHit:false,joint:false};
  if(!selected)return;
  updatePointer(event);
  const hit=raycaster.intersectObjects(jointPickers,false)[0];
  if(hit){
    if(referenceLocked){warnLocked();event.preventDefault();return;}
    const joint=hit.object;
    drag={joint:joint.userData.joint,dot:joint};
    gesture.joint=true;
    const worldPosition=joint.getWorldPosition(new THREE.Vector3());
    dragPlane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()),worldPosition);
    controls.enabled=false;
    canvas.setPointerCapture(event.pointerId);
    setStatus('拖动关节 '+drag.joint+'，松开后应用 '+(document.querySelector('#full-body-ik').checked?'全身 IK':'局部旋转'));
    event.preventDefault();
    return;
  }
  if(segmentLines){
    raycaster.params.Line.threshold=.035;
    const lineHit=raycaster.intersectObject(segmentLines,false)[0];
    if(lineHit){
      const child=segmentLines.userData.children[Math.floor(lineHit.index/2)];
      if(child<=21){
        selectedSegment=child;
        gesture.boneHit=true;
        segmentEditor();
        setStatus('已选择骨段 '+skeletonData.names[skeletonData.parents[child]]+' → '+skeletonData.names[child]);
        event.preventDefault();
      }
    }
  }
});
canvas.addEventListener('pointermove',(event)=>{
  if(gesture&&!gesture.moved&&(Math.abs(event.clientX-gesture.x)>4||Math.abs(event.clientY-gesture.y)>4))gesture.moved=true;
  if(!drag)return;
  updatePointer(event);
  const p=new THREE.Vector3();
  if(raycaster.ray.intersectPlane(dragPlane,p))drag.dot.position.copy(skeletonRoot.worldToLocal(p.clone()));
});
canvas.addEventListener('pointerup',async (event)=>{
  const finished=gesture;
  gesture=null;
  if(drag){
    const moved=drag;
    drag=null;
    controls.enabled=true;
    canvas.releasePointerCapture(event.pointerId);
    const fullBody=document.querySelector('#full-body-ik').checked;
    setStatus('正在求解 '+(fullBody?'全身 IK...':'局部旋转...'));
    try{
      const r=await fetch('/api/pose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gender:selected,joint:moved.joint,target:moved.dot.position.toArray(),full_body:fullBody})});
      if(!r.ok)throw Error(await r.text());
      await r.json();
      clearSegmentSelection();
      await load(selected,true,true);
    }catch(e){
      setStatus('IK 求解失败：'+e.message,true);
      clearSegmentSelection();
      await load(selected,true,true);
    }
    return;
  }
  // A plain click that changed nothing deselects; an orbit / pan drag counts as
  // a modification and leaves the selection alone.
  if(finished&&!finished.moved&&!finished.boneHit&&!finished.joint)clearSegmentSelection();
});
document.querySelector('#reset').onclick=async()=>{if(!selected)return;if(referenceLocked){warnLocked();return;}const button=document.querySelector('#reset');button.disabled=true;setStatus('正在重置当前姿态...');try{const response=await fetch('/api/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gender:selected})});if(!response.ok)throw Error(await response.text());const result=await response.json();if(!result.ok)throw Error('重置未完成');selectedSegment=null;await load(selected,true,true);setStatus('已重置为默认 SMPL-X 姿态');}catch(error){setStatus('重置失败：'+error.message,true);button.disabled=false;}};
const refInput=document.querySelector('#ref-file'), refPick=document.querySelector('#ref-pick'), refClear=document.querySelector('#ref-clear'), refHide=document.querySelector('#ref-hide');
refPick.onclick=()=>refInput.click();
refInput.onchange=()=>{const file=refInput.files&&refInput.files[0];refInput.value='';if(file)showReferenceFile(file);};
refHide.onclick=()=>{
  if(!referenceRoot)return;
  referenceHidden=!referenceHidden;
  referenceRoot.visible=!referenceHidden;
  updateReferenceButtons();
  refreshReferenceInfo();
  setStatus(referenceHidden?'参考图已隐藏':'参考图已显示');
};
refClear.onclick=()=>{clearReference();setStatus('已移除参考图');};
async function init(){const r=await fetch('/api/models');if(!r.ok)throw Error('无法读取模型信息');const data=await r.json();for(const item of data.models){const option=document.createElement('option');option.value=item.gender;option.textContent=item.label+(item.available?'':'（未安装）');option.disabled=!item.available;choices.append(option)}choices.onchange=()=>load(choices.value,true);const first=data.models.find(x=>x.available);if(first)load(first.gender,true);else setStatus('未找到可用 SMPL-X 模型文件',true)}
init().catch(e=>setStatus(e.message,true));
</script></body></html>"""

GENDERS = {
    "female": "女性",
    "male": "男性",
    "neutral": "中性",
}

JOINT_NAMES = [
    "骨盆", "左髋", "右髋", "脊柱下", "左膝", "右膝", "脊柱中", "左踝", "右踝", "脊柱上",
    "左脚", "右脚", "颈部", "左锁骨", "右锁骨", "头部", "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
]


DEVICE_CHOICES = ("auto", "cpu", "cuda")

#: Pose-refinement pass of the reference alignment.  ``False`` runs the
#: per-joint IK with the "全身 IK" chain disabled (each joint is placed by
#: rotating its direct parent only); set it to ``True`` to re-run the same
#: sweep with the full parent chain solved together.
REFERENCE_IK_FULL_BODY = True


def resolve_device(requested: str = "auto") -> tuple[str, str]:
    """Decide where SMPL-X inference runs, before any weights are loaded.

    ``auto`` keeps CUDA only when the driver, the CUDA-enabled PyTorch build and
    a real tensor operation all succeed; anything else falls back to the CPU,
    which is the behaviour of the original code.  The probe is one tiny
    allocation plus one elementwise op, so a broken driver surfaces here at
    startup instead of inside the first request.

    Returns ``(device, reason)``; ``reason`` is meant for the startup log.
    """
    requested = (requested or "auto").lower()
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {requested!r}; expected one of {DEVICE_CHOICES}")
    if requested == "cpu":
        return "cpu", "requested on the command line"

    import torch

    if not torch.cuda.is_available():
        return "cpu", (
            f"torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, CUDA build {torch.version.cuda})"
        )

    try:
        probe = torch.zeros(8, dtype=torch.float32, device="cuda") + 1.0
        torch.cuda.synchronize()
        if not bool(torch.isfinite(probe).all()):
            raise RuntimeError("probe produced non-finite values")
    except Exception as exc:
        return "cpu", f"CUDA probe failed: {type(exc).__name__}: {exc}"

    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        return "cuda", f"{name} (sm_{major}{minor}, CUDA {torch.version.cuda})"
    except Exception:
        return "cuda", f"CUDA device (CUDA {torch.version.cuda})"


class ModelCache:
    def __init__(self, model_dir: Path, cache_dir: Path, device: str = "cpu",
                 detector_dir: str = ""):
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.detector_dir = detector_dir
        self._detector = None
        self.device = device if device in ("cpu", "cuda") else "cpu"
        self.lock = threading.Lock()
        self.poses = {}
        self.global_orients = {}
        self.models = {}
        self.outputs = {}
        self.segment_configs = {}
        self.geometry_cache = {}

    def source(self, gender: str) -> Path:
        return self.model_dir / f"SMPLX_{gender.upper()}.npz"

    def available(self, gender: str) -> bool:
        return self.source(gender).is_file()

    def _model(self, gender: str):
        if gender not in self.models:
            import smplx
            model = smplx.create(
                str(self.source(gender)), model_type="smplx", gender=gender,
                use_pca=False, flat_hand_mean=True, batch_size=1,
            )
            self.models[gender] = self._place_model(model)
        return self.models[gender]

    def _place_model(self, model):
        """Move a newly created model onto the selected device.

        A failed transfer (driver/architecture mismatch, out of memory) must not
        take the viewer down, so the session retreats to the CPU -- the original
        behaviour -- and reports why.
        """
        if self.device == "cpu":
            return model
        try:
            return model.to(self.device)
        except Exception as exc:
            print(
                f"[device] could not move SMPL-X to {self.device}: "
                f"{type(exc).__name__}: {exc} -> falling back to cpu",
                flush=True,
            )
            # Anything already cached has to come back with us, otherwise a CUDA
            # pose tensor would meet a CPU model on the next request.
            for cached in self.models.values():
                try:
                    cached.to("cpu")
                except Exception:
                    pass
            self.poses.clear()
            self.outputs.clear()
            self.geometry_cache.clear()
            self.device = "cpu"
            try:
                return model.to("cpu")
            except Exception:
                return model

    def _pose(self, gender: str, model):
        import torch
        if gender not in self.poses:
            self.poses[gender] = torch.zeros(
                (1, model.NUM_BODY_JOINTS * 3),
                dtype=torch.float32,
                device=self.device,
            )
        return self.poses[gender]

    def _orient(self, gender: str, model):
        import torch
        if gender not in self.global_orients:
            self.global_orients[gender] = torch.zeros(
                (1, 3), dtype=torch.float32, device=self.device
            )
        return self.global_orients[gender]

    def _output(self, gender: str):
        import torch
        if gender in self.outputs:
            return self._model(gender), self.outputs[gender]
        model = self._model(gender)
        with torch.no_grad():
            output = model(
                body_pose=self._pose(gender, model),
                global_orient=self._orient(gender, model),
            )
        self.outputs[gender] = output
        return model, output

    def _segments(self, gender: str):
        return self.segment_configs.setdefault(gender, {})

    def _geometry(self, gender: str):
        if gender in self.geometry_cache:
            return self.geometry_cache[gender]
        model, output = self._output(gender)
        base_vertices = output.vertices[0].cpu().numpy().astype(np.float32)
        base_joints = output.joints[0, :len(model.parents)].cpu().numpy().astype(np.float32)
        parents = model.parents.cpu().numpy().astype(np.int32)
        joints = base_joints.copy()
        segments = self._segments(gender)
        # No segment edits must mean the exact, unmodified SMPL-X output.
        # This avoids even harmless numerical drift in the canonical T-pose.
        if not segments:
            result = (model, base_vertices, joints, parents)
            self.geometry_cache[gender] = result
            return result
        for child in range(1, len(parents)):
            parent = int(parents[child])
            if parent < 0:
                continue
            factor = float(segments.get(str(child), {}).get("length", 1.0))
            joints[child] = joints[parent] + (base_joints[child] - base_joints[parent]) * factor

        vertices = base_vertices.copy()
        weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
        dominant_joint = np.argmax(weights[:, :len(joints)], axis=1)
        body_children = [index for index in range(1, 22) if parents[index] >= 0]
        nearest_distance = np.full(len(base_vertices), np.inf, dtype=np.float32)
        nearest_segment = np.full(len(base_vertices), -1, dtype=np.int32)
        for candidate in body_children:
            parent = int(parents[candidate])
            start, end = base_joints[parent], base_joints[candidate]
            vector = end - start
            length_sq = float(vector @ vector)
            if length_sq < 1e-10:
                continue
            relative = base_vertices - start
            t0 = np.clip((relative @ vector) / length_sq, 0.0, 1.0)
            distance_sq = ((base_vertices - (start + t0[:, None] * vector)) ** 2).sum(axis=1)
            replace = distance_sq < nearest_distance
            nearest_distance[replace] = distance_sq[replace]
            nearest_segment[replace] = candidate

        def descendants(root: int) -> set[int]:
            result = {root}
            changed = True
            while changed:
                changed = False
                for index, parent in enumerate(parents):
                    if parent in result and index not in result:
                        result.add(index)
                        changed = True
            return result

        for child_text, config in segments.items():
            child = int(child_text)
            parent = int(parents[child])
            if child > 21 or parent < 0:
                continue
            start0, end0 = base_joints[parent], base_joints[child]
            axis0 = end0 - start0
            base_length = float(np.linalg.norm(axis0))
            if base_length < 1e-6:
                continue
            axis0 /= base_length
            relative0 = base_vertices - start0
            raw_t = relative0 @ axis0 / base_length
            feather = 0.08
            t = np.clip(raw_t, 0.0, 1.0)
            axial_mask = np.clip((raw_t + feather) / feather, 0.0, 1.0)
            axial_mask *= np.clip((1.0 + feather - raw_t) / feather, 0.0, 1.0)
            segment_owner = (nearest_segment == child).astype(np.float32) * axial_mask
            if child in (18, 19):
                # Upper-arm seams share vertices with torso and forearm. Gate
                # the continuous upper-arm deformation by this side's
                # shoulder+elbow weights so torso surface cannot be pulled
                # into the arm, and so the distal forearm remains a child
                # branch that translates with the elbow.
                if child == 18:
                    arm_weight = weights[:, 16] + weights[:, 18]
                    neighbor_weight = weights[:, 13] + weights[:, 20]
                else:
                    arm_weight = weights[:, 17] + weights[:, 19]
                    neighbor_weight = weights[:, 14] + weights[:, 21]
                upper_surface = (arm_weight > 0.03) & (arm_weight >= neighbor_weight * 0.55) & (raw_t <= 1.0)
                segment_owner = upper_surface.astype(np.float32) * axial_mask
            if child in (7, 8):
                # At the ankle seam, nearest-bone classification leaves some
                # genuine shin vertices assigned to thigh/foot. Use the local
                # SMPL-X knee+ankle weights, constrained against neighboring
                # same-side weights, so the opposite calf stays untouched.
                if child == 7:
                    calf_weight = weights[:, 4] + weights[:, 7]
                    neighbor_weight = weights[:, 1] + weights[:, 10]
                else:
                    calf_weight = weights[:, 5] + weights[:, 8]
                    neighbor_weight = weights[:, 2] + weights[:, 11]
                calf_surface = (calf_weight > 0.03) & (calf_weight >= neighbor_weight * 0.55) & (raw_t <= 1.0)
                segment_owner = calf_surface.astype(np.float32) * axial_mask
            # Use this link's local length change only. The absolute child
            # displacement also includes every ancestor's edit (for example a
            # shortened thigh moves the knee). Applying that absolute value to
            # a calf again double-counted the thigh deformation and separated
            # the mesh from its correctly-recursed skeleton.
            delta = (joints[child] - joints[parent]) - (base_joints[child] - base_joints[parent])
            desired_translation = t[:, None] * delta
            if child in (18, 19):
                # Keep a short shoulder-side attachment band fixed so the
                # torso-to-upper-arm seam does not cave in during resizing.
                shoulder_guard = 0.12
                upper_t = np.clip((t - shoulder_guard) / (1.0 - shoulder_guard), 0.0, 1.0)
                desired_translation = upper_t[:, None] * delta
            if child in (20, 21):
                # Weld the distal forearm to the rigidly translated hand. The
                # final wrist band transitions from axial scaling to the full
                # wrist displacement, preventing a fold across shared seam
                # triangles while preserving the forearm's main length edit.
                seam = np.clip((t - 0.78) / 0.22, 0.0, 1.0)
                seam = seam * seam * (3.0 - 2.0 * seam)
                desired_translation = (t + (1.0 - t) * seam)[:, None] * delta
            subtree = descendants(child)
            downstream_segments = list((subtree - {child}) & set(body_children))
            is_branch = np.isin(nearest_segment, downstream_segments)
            is_branch |= np.isin(dominant_joint, list(subtree)) & (raw_t > 1.0)
            rigid_hand = np.zeros(len(vertices), dtype=bool)
            rigid_forearm = np.zeros(len(vertices), dtype=bool)
            if child == 18:
                rigid_forearm = (weights[:, 18] + weights[:, 20] > 0.03) & (raw_t > 0.92)
            elif child == 19:
                rigid_forearm = (weights[:, 19] + weights[:, 21] > 0.03) & (raw_t > 0.92)
            if child in (20, 21):
                rigid_hand = np.isin(dominant_joint, list(subtree))
            downstream = (is_branch & (raw_t > 1.0)) | rigid_hand | rigid_forearm
            vertices[downstream] += delta
            segment_owner[rigid_hand] = 0.0
            segment_owner[rigid_forearm] = 0.0
            vertices += desired_translation * segment_owner[:, None]

            start, end = joints[parent], joints[child]
            axis = end - start
            length = float(np.linalg.norm(axis))
            if length < 1e-6:
                continue
            axis /= length
            relative = vertices - start
            radial = relative - np.outer(relative @ axis, axis)
            radii = np.asarray(config.get("radii", [1.0, 1.0, 1.0]), dtype=np.float32)
            scale = np.interp(t, np.linspace(0.0, 1.0, len(radii)), radii).astype(np.float32)
            vertices += radial * ((scale - 1.0) * segment_owner)[:, None]

        result = (model, vertices, joints, parents)
        self.geometry_cache[gender] = result
        return result

    def _invalidate_geometry(self, gender: str):
        self.geometry_cache.pop(gender, None)

    def obj(self, gender: str) -> Path:
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        target = self.cache_dir / f"smplx_{gender}_tpose.obj"
        with self.lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            import trimesh
            model, vertices, _joints, _parents = self._geometry(gender)
            mesh = trimesh.Trimesh(vertices, model.faces, process=False)
            fd, temporary = tempfile.mkstemp(suffix=".obj", dir=self.cache_dir)
            os.close(fd)
            try:
                mesh.export(temporary)
                Path(temporary).replace(target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return target

    def skeleton(self, gender: str) -> dict:
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        model, _vertices, joints, parents = self._geometry(gender)
        # parents covers the 55 SMPL-X body and hand joints. The remaining
        # regressed landmarks are not a kinematic skeleton hierarchy.
        return {
            "joints": joints.tolist(),
            "parents": parents.tolist(),
            "names": JOINT_NAMES,
            "segments": self._segments(gender),
        }

    def update_segment(self, gender: str, child: int, length: float, radii: list[float]):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        if child < 1 or child > 21:
            raise ValueError("Invalid body segment")
        if child not in (4, 5, 7, 8, 18, 19, 20, 21) and abs(length - 1.0) > 1e-6:
            raise ValueError("Length editing is currently available only for limbs")
        if not 2 <= len(radii) <= 12 or any(not 0.3 <= value <= 2.5 for value in radii):
            raise ValueError("Radius controls must contain 2-12 values between 0.3 and 2.5")
        with self.lock:
            self._segments(gender)[str(child)] = {
                "length": float(length) if child in (4, 5, 7, 8, 18, 19, 20, 21) else 1.0,
                "radii": [float(value) for value in radii],
            }
            self._invalidate_geometry(gender)

    def reset_segment(self, gender: str, child: int):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        if child < 1 or child > 21:
            raise ValueError("Invalid body segment")
        with self.lock:
            self._segments(gender).pop(str(child), None)
            self._invalidate_geometry(gender)

    def reset(self, gender: str):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        with self.lock:
            model = self._model(gender)
            import torch
            self.poses[gender] = torch.zeros(
                (1, model.NUM_BODY_JOINTS * 3), dtype=torch.float32, device=self.device
            )
            self.global_orients.pop(gender, None)
            self.segment_configs.pop(gender, None)
            self.outputs.pop(gender, None)
            self._invalidate_geometry(gender)

    def pose_joint(self, gender: str, joint: int, target: list[float], full_body: bool,
                   iterations: int | None = None):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        if joint < 1 or joint > 21:
            raise ValueError("Only body joints 1-21 are editable in this version")
        import torch
        with self.lock:
            model = self._model(gender)
            base = self._pose(gender, model).clone()
            parents = model.parents.cpu().tolist()
            if full_body:
                chain = []
                cursor = joint
                while cursor > 0:
                    chain.append(cursor)
                    cursor = parents[cursor]
            else:
                # A joint's origin is attached to its parent in SMPL-X. To move
                # the selected point while keeping upstream points fixed, rotate
                # its direct parent; the selected point and its child subtree
                # then move together as one local branch.
                parent = parents[joint]
                if parent <= 0:
                    raise ValueError("This root-adjacent joint requires full-body IK")
                chain = [parent]
            variables = torch.nn.Parameter(base.clone())
            target_tensor = torch.tensor(
                target, dtype=torch.float32, device=self.device
            ).view(1, 3)
            # The visible skeleton is the *length-scaled* one (segment editor), so
            # the solver has to compare against that same joint, not the raw one.
            factors = np.ones(len(parents), dtype=np.float32)
            segment_config = self._segments(gender)
            for index in range(1, min(22, len(parents))):
                factors[index] = float(
                    segment_config.get(str(index), {}).get("length", 1.0)
                )
            lengths = torch.tensor(factors, dtype=torch.float32, device=self.device)
            chain_to_joint = []
            cursor = joint
            while cursor > 0:
                chain_to_joint.append(cursor)
                cursor = parents[cursor]
            chain_to_joint.reverse()

            def scaled_position(output):
                joints = output.joints[0]
                position = joints[0]
                for node in chain_to_joint:
                    parent_index = parents[node]
                    position = position + (joints[node] - joints[parent_index]) * lengths[node]
                return position

            optimizer = torch.optim.Adam([variables], lr=0.035)
            editable = torch.zeros_like(variables)
            for node in chain:
                editable[:, (node - 1) * 3:node * 3] = 1
            global_orient = self._orient(gender, model)
            for _ in range(iterations if iterations is not None else (36 if full_body else 24)):
                optimizer.zero_grad(set_to_none=True)
                pose = variables * editable + base * (1 - editable)
                output = model(body_pose=pose, global_orient=global_orient)
                loss = (scaled_position(output) - target_tensor[0]).square().sum()
                loss = loss + 0.003 * ((pose - base) * editable).square().mean()
                loss.backward()
                optimizer.step()
            self.poses[gender] = (variables.detach() * editable + base * (1 - editable))
            with torch.no_grad():
                self.outputs[gender] = model(
                    body_pose=self.poses[gender], global_orient=global_orient
                )
            self._invalidate_geometry(gender)


    # ------------------------------------------------------------------
    # Reference picture: 2D pose estimation (AnimePoseDetector)
    # ------------------------------------------------------------------

    def _detector_module(self):
        """Import AnimePoseDetector lazily (it needs OpenCV, never torch)."""
        if self._detector is None:
            if not self.detector_dir or not Path(self.detector_dir).is_dir():
                raise RuntimeError(f"AnimePoseDetector 目录不可用: {self.detector_dir!r}")
            if self.detector_dir not in sys.path:
                sys.path.insert(0, self.detector_dir)
            import anime_pose_detector as apd
            self._detector = apd
        return self._detector

    def reference_eyes(self, image_bytes: bytes) -> dict:
        """Run AnimePoseDetector on an upload and return the eye anchor.

        The detector always writes ``<stem>_result.json`` beside its input, so
        the upload is decoded inside a throw-away directory that is removed on
        the way out: the user's picture never reaches the filesystem.
        """
        apd = self._detector_module()
        with tempfile.TemporaryDirectory(prefix="smplx_ref_") as tmp:
            source = Path(tmp) / "upload.png"
            source.write_bytes(image_bytes)
            result = apd.estimate(
                str(source),
                smplx_dir=str(self.model_dir),
                output_dir=tmp,
                write_vis=False,
            )
        eyes = result.get("eyes") or {}
        midpoint = result.get("eye_midpoint")
        left = (eyes.get("left") or {}).get("center")
        right = (eyes.get("right") or {}).get("center")
        if not midpoint or not left or not right:
            raise ValueError("未检测到双眼")
        smplx_block = result.get("smplx") or {}
        fit = smplx_block.get("fit_core_torso") or smplx_block.get("image_to_model_2d") or {}
        scale = float(fit.get("scale") or 0.0)
        if not np.isfinite(scale) or scale <= 1e-6:
            raise ValueError("检测到的像素/米比例无效")
        joints = {}
        for entry in (result.get("joints") or {}).values():
            index = entry.get("smplx_index")
            xy = entry.get("xy")
            if index is None or xy is None or not (0 <= int(index) <= 22):
                continue
            joints[str(int(index))] = [float(xy[0]), float(xy[1])]
        return {
            "ok": True,
            "eye_midpoint": [float(midpoint[0]), float(midpoint[1])],
            "eye_left": [float(left[0]), float(left[1])],
            "eye_right": [float(right[0]), float(right[1])],
            "pixels_per_meter": scale,
            "image_width": int(result["image_width"]),
            "image_height": int(result["image_height"]),
            "joints": joints,
            "elapsed_ms": result.get("elapsed_ms"),
        }


    def solve_reference_lengths(self, gender: str, pixels_per_meter: float,
                                eye_midpoint, joints: dict,
                                z_weight: float = 3.0) -> dict:
        """Fit the 21 editable bone lengths so the model's joints land on the picture.

        The picture already sits on the model's eye midpoint, so every joint is
        compared *relative to the eyes*: that removes the ground-offset and the
        plane translation from the problem and makes the fit independent of how
        the picture is placed.

        With ``J_i = J_root + sum_{k in path(i)} v_k * L_k`` (``v_k`` = the posed
        bone vector, ``L_k`` the length factor -- exactly how ``_geometry`` builds
        the joints), the constraint ``J_i - J_eye = delta_i`` is **linear** in
        ``L``, so it is one least-squares solve with a light pull towards 1.
        """
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        mpp = 1.0 / float(pixels_per_meter)
        if not np.isfinite(mpp) or mpp <= 0 or not np.isfinite(float(pixels_per_meter)):
            raise ValueError("invalid pixels_per_meter")
        model, output = self._output(gender)
        base = output.joints[0, :len(model.parents)].detach().cpu().numpy().astype(np.float64)
        parents = model.parents.cpu().numpy().astype(np.int64)
        bones = [k for k in range(1, 22) if parents[k] >= 0]
        slot = {k: i for i, k in enumerate(bones)}
        vectors = {k: base[k] - base[int(parents[k])] for k in bones}

        # J_eye = J_15 + eye_offset; the 15->23/24 bones are not editable.
        eye_offset = (base[23] + base[24]) / 2.0 - base[15]

        def path(joint: int) -> list[int]:
            chain = []
            while joint > 0:
                chain.append(joint)
                joint = int(parents[joint])
            return chain

        def row_for(joint: int) -> np.ndarray:
            row = np.zeros((3, len(bones)))
            for k in path(joint):
                if k in slot:
                    row[0, slot[k]] += vectors[k][0]
                    row[1, slot[k]] += vectors[k][1]
                    row[2, slot[k]] += vectors[k][2]
            return row

        eye_row = row_for(15)
        rows, rhs = [], []
        used = 0
        for key, xy in joints.items():
            joint = int(key)
            if joint < 0 or joint > 22 or len(xy) != 2:
                continue
            target = np.array([(float(xy[0]) - eye_midpoint[0]) * mpp,
                               (eye_midpoint[1] - float(xy[1])) * mpp])
            full = row_for(joint)
            a = full[:2] - eye_row[:2]
            b = target + eye_offset[:2]
            rows.append(a[0]); rhs.append(b[0] * 1.0)
            rows.append(a[1]); rhs.append(b[1] * 1.0)
            # keep the joint's own depth: sum(v_kz * L_k) == sum(v_kz)
            rows.append(full[2] * z_weight)
            rhs.append(float((full[2]).sum()) * z_weight)
            used += 1
        if used < 4:
            raise ValueError("not enough 2D joints to fit the skeleton")

        design = np.asarray(rows)
        values = np.asarray(rhs)
        lam = 0.15
        design = np.vstack([design, lam * np.eye(len(bones))])
        values = np.concatenate([values, lam * np.ones(len(bones))])
        factors, *_ = np.linalg.lstsq(design, values, rcond=None)
        factors = np.clip(factors, 0.25, 4.0)

        with self.lock:
            config = self._segments(gender)
            for k, factor in zip(bones, factors):
                entry = dict(config.get(str(k), {}))
                entry["length"] = float(factor)
                entry.setdefault("radii", [1.0, 1.0, 1.0])
                config[str(k)] = entry
            self._invalidate_geometry(gender)
            _model, vertices, new_joints, _parents = self._geometry(gender)

        # residual measured on the geometry that will actually be rendered
        rms_px, max_px = self._reference_residual(
            gender, pixels_per_meter, eye_midpoint, joints
        )
        return {
            "ok": True,
            "lengths": {str(k): float(f) for k, f in zip(bones, factors)},
            "num_joints": int(used),
            "rms_px": rms_px,
            "max_px": max_px,
        }

    def _reference_residual(self, gender, pixels_per_meter, eye_midpoint, joints):
        """RMS / max pixel error of the joints on the geometry being rendered."""
        mpp = 1.0 / float(pixels_per_meter)
        _model, _vertices, new_joints, _parents = self._geometry(gender)
        eye = (new_joints[23] + new_joints[24]) / 2.0
        errors = []
        for key, xy in joints.items():
            joint = int(key)
            if joint < 0 or joint > 22 or len(xy) != 2:
                continue
            predicted = np.array([
                eye_midpoint[0] + (new_joints[joint][0] - eye[0]) / mpp,
                eye_midpoint[1] - (new_joints[joint][1] - eye[1]) / mpp,
            ])
            errors.append(float(np.hypot(predicted[0] - xy[0], predicted[1] - xy[1])))
        errors = np.asarray(errors) if errors else np.zeros(1)
        return float(np.sqrt((errors ** 2).mean())), float(errors.max())

    def _scaled_joints(self, model, body_pose, lengths, global_orient=None):
        """Joints of the *rendered* skeleton: segment lengths applied on top.

        Mirrors ``_geometry``: ``S[k] = S[parent] + (J[k] - J[parent]) * L[k]``.
        """
        output = model(body_pose=body_pose, global_orient=global_orient)
        joints = output.joints[0][:len(model.parents)]
        scaled = joints.clone()
        parents = model.parents
        for index in range(1, len(parents)):
            parent = int(parents[index])
            if parent < 0:
                continue
            scaled[index] = scaled[parent] + (joints[index] - joints[parent]) * lengths[index]
        return scaled

    def refine_pose_to_reference(self, gender: str, pixels_per_meter: float,
                                 eye_midpoint, joints: dict,
                                 full_body: bool | None = None,
                                 iterations: int = 160,
                                 learning_rate: float = 0.02,
                                 reference_z: dict | None = None,
                                 z_weight: float = 20.0) -> dict:
        """Second stage: let the *pose* absorb what the bone lengths cannot.

        Note on "全身 IK": running the viewer's per-joint drag solver in a sweep
        (rotate only the direct parent, checkbox off) is unusable for the torso
        -- rotating a spine joint to fix the neck swings the whole upper body and
        the error explodes (measured: 17 px -> 251 px over one sweep).  The pose
        is therefore solved as a single least-squares problem over the 21 body
        joint rotations, which is stable and never ships a worse pose.

        Each joint is pulled to its picture ``(x, y)`` **inside its own plane**:
        the target depth stays the joint's original ``z`` (``reference_z``), so
        the alignment never tilts the body through the depth axis.

        ``REFERENCE_IK_FULL_BODY`` picks how much chain participates:
        ``False`` keeps the pelvis/root fixed (only the joints rotate locally);
        ``True`` also frees the root orientation, i.e. the whole body may turn.
        """
        if full_body is None:
            full_body = REFERENCE_IK_FULL_BODY
        import torch
        model = self._model(gender)
        mpp = 1.0 / float(pixels_per_meter)
        deltas, indexes = [], []
        for key, xy in joints.items():
            index = int(key)
            if 0 <= index <= 22 and len(xy) == 2:
                deltas.append([(float(xy[0]) - eye_midpoint[0]) * mpp,
                               (eye_midpoint[1] - float(xy[1])) * mpp])
                indexes.append(index)
        if not indexes:
            raise ValueError("no 2D joints to refine the pose with")
        target = torch.tensor(deltas, dtype=torch.float32, device=self.device)
        joint_ids = torch.tensor(indexes, dtype=torch.long, device=self.device)
        if reference_z is None:
            _model0, _v0, current0, _p0 = self._geometry(gender)
            reference_z = {index: float(current0[index][2]) for index in indexes}
        target_z = torch.tensor(
            [float(reference_z[index]) for index in indexes],
            dtype=torch.float32, device=self.device,
        )

        factors = np.ones(len(model.parents), dtype=np.float32)
        config = self._segments(gender)
        for index in range(1, 22):
            factors[index] = float(config.get(str(index), {}).get("length", 1.0))
        lengths = torch.tensor(factors, dtype=torch.float32, device=self.device)

        with self.lock:
            pose0 = self._pose(gender, model).clone()
            orient0 = self._orient(gender, model).clone()
            before, _ = self._reference_residual(
                gender, pixels_per_meter, eye_midpoint, joints
            )
            variables = torch.nn.Parameter(pose0.clone())
            parameters = [variables]
            root = None
            if full_body:
                root = torch.nn.Parameter(orient0.clone())
                parameters.append(root)
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                scaled = self._scaled_joints(
                    model, variables, lengths,
                    root if root is not None else None,
                )
                eye = (scaled[23] + scaled[24]) / 2.0
                points = scaled[joint_ids]
                residual = (points[:, :2] - eye[:2]) - target
                depth = points[:, 2] - target_z          # keep each joint in its own plane
                loss = residual.square().sum() + z_weight * depth.square().sum()
                loss = loss + 0.02 * (variables - pose0).square().mean()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                self.poses[gender] = variables.detach().clone()
                if root is not None:
                    self.global_orients[gender] = root.detach().clone()
                self.outputs[gender] = model(
                    body_pose=self.poses[gender],
                    global_orient=self.global_orients.get(gender),
                )
            self._invalidate_geometry(gender)
            after, _ = self._reference_residual(
                gender, pixels_per_meter, eye_midpoint, joints
            )
            if after > before:                      # never ship a worse pose
                self.poses[gender] = pose0
                self.global_orients[gender] = orient0
                self.outputs.pop(gender, None)
                self._invalidate_geometry(gender)
                after = before

        return {
            "moved": [int(v) for v in indexes],
            "skipped": [],
            "full_body": bool(full_body),
            "rms_before_px": float(before),
            "pose_rms_px": float(after),
        }

    def align_reference(self, gender: str, pixels_per_meter: float,
                        eye_midpoint, joints: dict,
                        full_body: bool | None = None) -> dict:
        """Whole reference alignment: bone lengths first, then the pose pass.

        Each joint keeps the depth it had before the alignment (``reference_z``),
        so the fit happens inside the joint's own xy plane and the body is never
        tilted through the z axis.
        """
        model, base_output = self._output(gender)
        base_joints = base_output.joints[0].detach().cpu().numpy()
        reference_z = {index: float(base_joints[index][2]) for index in range(23)}
        fit = self.solve_reference_lengths(gender, pixels_per_meter, eye_midpoint, joints)
        lengths_fit = {"rms_px": fit["rms_px"], "max_px": fit["max_px"]}
        pose = self.refine_pose_to_reference(
            gender, pixels_per_meter, eye_midpoint, joints,
            full_body=full_body, reference_z=reference_z,
        )
        rms_px, max_px = self._reference_residual(
            gender, pixels_per_meter, eye_midpoint, joints
        )
        _model, _vertices, current, _parents = self._geometry(gender)
        z_drift = max(
            abs(float(current[index][2]) - reference_z[index]) for index in reference_z
        )
        return {
            "ok": True,
            "lengths": fit["lengths"],
            "num_joints": fit["num_joints"],
            "rms_px": rms_px,
            "max_px": max_px,
            "lengths_rms_px": lengths_fit["rms_px"],
            "lengths_max_px": lengths_fit["max_px"],
            "z_drift_m": float(z_drift),
            "pose_moved": pose["moved"],
            "pose_skipped": pose["skipped"],
            "full_body": pose["full_body"],
        }

    def owners(self, gender: str) -> bytes:
        """One byte per vertex: the body segment (1..21) that owns it.

        This is the *same* geometric partition the segment editor deforms with
        (nearest bone segment in the rest pose), so the highlighted skin is
        exactly the surface a segment edit moves.  ``argmax(lbs_weights)`` would
        be wrong here: SMPL-X binds a bone's surface to its *proximal* joint, so
        a forearm segment would only light up ~200 hand vertices.
        """
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        model = self._model(gender)
        vertices = model.v_template.detach().cpu().numpy().astype(np.float32)
        joints = (model.J_regressor @ model.v_template).detach().cpu().numpy()[:len(model.parents)]
        parents = model.parents.cpu().numpy().astype(np.int32)
        nearest_distance = np.full(len(vertices), np.inf, dtype=np.float32)
        nearest_segment = np.zeros(len(vertices), dtype=np.uint8)
        for candidate in range(1, 22):
            parent = int(parents[candidate])
            if parent < 0:
                continue
            start, end = joints[parent], joints[candidate]
            vector = end - start
            length_sq = float(vector @ vector)
            if length_sq < 1e-10:
                continue
            relative = vertices - start
            t = np.clip((relative @ vector) / length_sq, 0.0, 1.0)
            distance_sq = ((vertices - (start + t[:, None] * vector)) ** 2).sum(axis=1)
            replace = distance_sq < nearest_distance
            nearest_distance[replace] = distance_sq[replace]
            nearest_segment[replace] = candidate
        return nearest_segment.tobytes()


def handler_factory(cache: ModelCache):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def send_json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/models":
                self.send_json({"models": [{"gender": gender, "label": label, "available": cache.available(gender)} for gender, label in GENDERS.items()]})
                return
            if path.startswith("/api/model/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    model_path = cache.obj(gender)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                    return
                data = model_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path.startswith("/api/skeleton/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    self.send_json(cache.skeleton(gender))
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            if path.startswith("/api/weights/"):
                gender = path.rsplit("/", 1)[-1]
                try:
                    data = cache.owners(gender)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
                    return
                except Exception as exc:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _handle_reference(self):
            """Decode an uploaded picture with AnimePoseDetector, return its eyes."""
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if size <= 0:
                self.send_error(HTTPStatus.BAD_REQUEST, "empty upload")
                return
            if size > 32 * 1024 * 1024:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "picture larger than 32 MiB")
                return
            payload = self.rfile.read(size)
            try:
                result = cache.reference_eyes(payload)
            except Exception as exc:
                self.send_error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{type(exc).__name__}: {exc}")
                return
            self.send_json(result)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/reference":
                self._handle_reference()
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(size).decode("utf-8"))
                gender = str(data["gender"])
                if path == "/api/reset":
                    cache.reset(gender)
                    self.send_json({"ok": True})
                    return
                if path == "/api/pose":
                    target = data["target"]
                    if not isinstance(target, list) or len(target) != 3:
                        raise ValueError("target must be [x, y, z]")
                    cache.pose_joint(
                        gender,
                        int(data["joint"]),
                        [float(value) for value in target],
                        bool(data.get("full_body", False)),
                    )
                    self.send_json({"ok": True})
                    return
                if path == "/api/segment":
                    radii = data["radii"]
                    if not isinstance(radii, list):
                        raise ValueError("radii must be a list")
                    cache.update_segment(
                        gender,
                        int(data["child"]),
                        float(data["length"]),
                        [float(value) for value in radii],
                    )
                    self.send_json({"ok": True})
                    return
                if path == "/api/segment/reset":
                    cache.reset_segment(gender, int(data["child"]))
                    self.send_json({"ok": True})
                    return
                if path == "/api/reference/solve":
                    override = data.get("full_body")
                    result = cache.align_reference(
                        gender,
                        float(data["pixels_per_meter"]),
                        [float(v) for v in data["eye_midpoint"]],
                        {str(k): [float(a), float(b)] for k, (a, b) in data["joints"].items()},
                        full_body=None if override is None else bool(override),
                    )
                    self.send_json(result)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "Requested SMPL-X gender model is not installed")
            except (KeyError, TypeError, ValueError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
    return Handler


def main():
    parser = argparse.ArgumentParser(description="SMPL-X T-pose viewer")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/hgc/models/smplx"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/hgc/project/AnimeAvatarGen/outputs/smplx_tpose_viewer"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=DEVICE_CHOICES,
        help="auto (CUDA when a working GPU is detected) | cuda | cpu",
    )
    parser.add_argument(
        "--detector-dir",
        default=os.environ.get("ANIME_POSE_DETECTOR_DIR", "/home/lomerson/MyProjects/AnimePoseDetector"),
        help="directory holding anime_pose_detector.py (reference-picture eye detection)",
    )
    args = parser.parse_args()
    device, reason = resolve_device(args.device)
    if args.device != "auto" and args.device != device:
        print(f"[device] {args.device} requested but unusable -> cpu", flush=True)
    print(f"device: {device} — {reason}", flush=True)
    print(f"detector: {args.detector_dir}", flush=True)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        handler_factory(
            ModelCache(
                args.model_dir, args.cache_dir, device=device,
                detector_dir=args.detector_dir,
            )
        ),
    )
    print(f"SMPL-X viewer: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
