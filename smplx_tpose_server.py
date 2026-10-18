#!/usr/bin/env python3
"""Interactive SMPL-X T-pose viewer with automatic gender-model discovery."""

from __future__ import annotations

import argparse
import json
import os
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
  body { background:#111318; color:#edf0f5; } #app { display:grid; grid-template-columns:300px minmax(0,1fr); }
  aside { background:#1a1e25; border-right:1px solid #303642; padding:24px 18px; display:flex; flex-direction:column; gap:22px; }
  h1 { font-size:18px; margin:0; } .sub { color:#aeb7c6; font-size:13px; line-height:1.55; margin:8px 0 0; }
  .section-title { display:block; color:#f4f6fa; font-size:16px; font-weight:700; margin:0 0 10px; } select { width:100%; min-height:43px; border:1px solid #414957; border-radius:6px; background:#252b35; color:#eff2f7; padding:0 11px; font-size:14px; cursor:pointer; }
  button { border:1px solid #414957; border-radius:6px; background:#252b35; color:#eff2f7; min-height:43px; padding:0 12px; font-size:14px; text-align:left; cursor:pointer; }
  button:hover:not(:disabled) { border-color:#7da6ef; background:#293544; } button.active { color:#fff; background:#24559a; border-color:#70a5f3; }
  button:disabled { color:#697383; cursor:not-allowed; } #status { margin-top:auto; color:#aeb7c6; font-size:12px; line-height:1.55; }
  main { min-width:0; min-height:0; position:relative; } canvas { display:block; width:100%; height:100%; }
  .hud { position:absolute; top:18px; right:18px; background:#1a1e25e8; border:1px solid #343b47; border-radius:6px; padding:11px 13px; color:#b9c2d0; font-size:12px; line-height:1.55; }
  .pose-controls { display:flex; gap:8px; align-items:stretch; } .toggle { display:flex; align-items:center; gap:7px; min-height:38px; flex:1; padding:0 10px; border:1px solid #414957; border-radius:6px; background:#252b35; font-size:14px; cursor:pointer; white-space:nowrap; } .toggle input { width:16px; height:16px; accent-color:#70a5f3; } #reset { min-height:38px; padding:0 10px; font-size:13px; white-space:nowrap; }
  .segment-empty { color:#8f99a9; font-size:13px; } .segment-name { color:#d9e5f8; font-size:13px; margin:0 0 10px; } .field { display:grid; grid-template-columns:62px minmax(0,1fr) 48px; gap:7px; align-items:center; margin:8px 0; font-size:12px; color:#b9c3d0; } input[type=range] { width:100%; accent-color:#70a5f3; } .value { color:#f1f4f8; text-align:right; font-variant-numeric:tabular-nums; } .segment-tools { display:flex; gap:8px; align-items:center; margin:10px 0 4px; } .segment-tools select { flex:1; min-height:34px; font-size:13px; } .small { min-height:34px; padding:0 10px; font-size:12px; } .selected-bone { color:#fff1b3; }
  .error { color:#ffabb8 !important; } @media(max-width:700px) { #app{grid-template-columns:1fr;grid-template-rows:auto minmax(0,1fr)} aside{padding:14px; border-right:0; border-bottom:1px solid #303642; gap:12px}.gender{grid-template-columns:repeat(3,1fr)} button{font-size:12px;text-align:center} #status{margin-top:0} }
</style>
</head>
<body><div id="app"><aside>
  <div><h1>SMPL-X T-pose</h1><p class="sub">选择性别模型。鼠标左键旋转，滚轮缩放，右键平移。</p></div>
  <div><span class="section-title">人体模型</span><select id="gender" aria-label="人体模型"></select></div>
  <div><span class="section-title">调整人体姿态</span><div class="pose-controls"><label class="toggle"><input id="full-body-ik" type="checkbox"> <span>全身 IK</span></label><button id="reset" disabled>重置当前姿态</button></div><p class="sub">关闭时移动所选点及其子节点，父节点保持固定；开启后允许整条父链共同求解。</p></div>
  <div><span class="section-title">骨段编辑</span><div id="segment-editor" class="segment-empty">点击一条黄色骨骼连线以编辑该骨段。</div></div>
  <div id="status">正在读取模型信息...</div>
</aside><main><canvas id="viewer"></canvas><div class="hud">点击黄色身体关节后拖动<br>空白处拖动旋转 · 滚轮缩放</div></main></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {OBJLoader} from 'three/addons/loaders/OBJLoader.js';
const canvas=document.querySelector('#viewer'), status=document.querySelector('#status'), choices=document.querySelector('#gender');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setClearColor(0x111318);
const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(38,1,.01,100); camera.position.set(3.2,1.7,5.6);
const controls=new OrbitControls(camera,renderer.domElement); controls.target.set(0,.82,0); controls.enableDamping=true; controls.minDistance=1.2; controls.maxDistance=10;
scene.add(new THREE.HemisphereLight(0xddeaff,0x20242c,2.0)); const key=new THREE.DirectionalLight(0xffffff,2.1); key.position.set(3,5,4); scene.add(key);
const grid=new THREE.GridHelper(8,16,0x4a596f,0x27303b); scene.add(grid); const axes=new THREE.AxesHelper(.45); axes.position.set(-2.5,.01,2.5); scene.add(axes);
let meshRoot=null, skeletonRoot=null, selected=null, jointPickers=[], segmentLines=null, skeletonData=null, drag=null, selectedSegment=null;
const raycaster=new THREE.Raycaster(), pointer=new THREE.Vector2(), dragPlane=new THREE.Plane();
function resize(){const r=canvas.getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix()} addEventListener('resize',resize); resize();
function render(){controls.update();renderer.render(scene,camera);requestAnimationFrame(render)} render();
function setStatus(text,err=false){status.textContent=text;status.className=err?'error':''}
function frame(root){const box=new THREE.Box3().setFromObject(root), c=box.getCenter(new THREE.Vector3()), s=box.getSize(new THREE.Vector3()); controls.target.set(c.x,c.y,c.z); camera.position.set(c.x,c.y+s.y*.10,c.z+s.y*2.05); controls.update();}
function segmentEditor(){const host=document.querySelector('#segment-editor');if(selectedSegment===null||!skeletonData){host.className='segment-empty';host.textContent='点击一条黄色骨骼连线以编辑该骨段。';return}const child=selectedSegment,parent=skeletonData.parents[child],cfg=skeletonData.segments[String(child)]||{length:1,radii:[1,1,1]};const names=skeletonData.names;host.className='';host.innerHTML=`<p class="segment-name"><span class="selected-bone">${names[parent]} → ${names[child]}</span></p><div class="field"><span>长度</span><input id="seg-length" type="range" min="0.35" max="2.5" step="0.01" value="${cfg.length}"><span class="value" id="seg-length-value">${cfg.length.toFixed(2)}</span></div><div class="segment-tools"><span>横截面</span><select id="seg-count">${Array.from({length:11},(_,i)=>i+2).map(n=>`<option value="${n}" ${n===cfg.radii.length?'selected':''}>${n} 段</option>`).join('')}</select><button class="small" id="seg-reset">重置此段</button></div><div id="radius-fields"></div>`;const renderRadii=radii=>{const fields=document.querySelector('#radius-fields');fields.innerHTML=radii.map((v,i)=>`<div class="field"><span>${Math.round(i/(radii.length-1)*100)}%</span><input data-radius="${i}" type="range" min="0.3" max="2.5" step="0.01" value="${v}"><span class="value">${v.toFixed(2)}</span></div>`).join('');fields.querySelectorAll('input').forEach(x=>x.oninput=()=>x.nextElementSibling.textContent=Number(x.value).toFixed(2));};renderRadii(cfg.radii);const apply=()=>applySegment(child,Number(document.querySelector('#seg-length').value),[...document.querySelectorAll('[data-radius]')].map(x=>Number(x.value)));const length=document.querySelector('#seg-length');length.oninput=()=>document.querySelector('#seg-length-value').textContent=Number(length.value).toFixed(2);length.onchange=apply;document.querySelectorAll('[data-radius]').forEach(x=>x.onchange=apply);document.querySelector('#seg-count').onchange=e=>{const count=Number(e.target.value),old=[...document.querySelectorAll('[data-radius]')].map(x=>Number(x.value));const next=Array.from({length:count},(_,i)=>old[Math.min(old.length-1,Math.round(i/(count-1)*(old.length-1)))]);renderRadii(next);applySegment(child,Number(length.value),next)};document.querySelector('#seg-reset').onclick=()=>applySegment(child,1,Array(Number(document.querySelector('#seg-count').value)).fill(1));}
async function applySegment(child,length,radii){setStatus('正在更新骨段 Mesh...');try{const r=await fetch('/api/segment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gender:selected,child,length,radii})});if(!r.ok)throw Error(await r.text());await load(selected,true,true);selectedSegment=child;segmentEditor()}catch(e){setStatus('骨段更新失败：'+e.message,true)}}
function addSkeleton(data,groundOffset){if(skeletonRoot)scene.remove(skeletonRoot);skeletonData=data;skeletonRoot=new THREE.Group();skeletonRoot.position.y=groundOffset;jointPickers=[];const joints=data.joints.map(p=>new THREE.Vector3(...p));const points=[],children=[];for(let i=0;i<data.parents.length;i++){const parent=data.parents[i];if(parent>=0&&parent<joints.length){points.push(joints[parent],joints[i]);children.push(i)}}const lineGeometry=new THREE.BufferGeometry().setFromPoints(points);segmentLines=new THREE.LineSegments(lineGeometry,new THREE.LineBasicMaterial({color:0xffc64d,depthTest:false,depthWrite:false,transparent:true,opacity:1}));segmentLines.userData.children=children;segmentLines.renderOrder=2;skeletonRoot.add(segmentLines);const jointMat=new THREE.MeshBasicMaterial({color:0xfff0a6,depthTest:false,depthWrite:false});for(let i=0;i<joints.length;i++){const dot=new THREE.Mesh(new THREE.SphereGeometry(.006,6,4),jointMat);dot.position.copy(joints[i]);dot.renderOrder=3;dot.userData.joint=i;skeletonRoot.add(dot);if(i>0&&i<=21)jointPickers.push(dot);}scene.add(skeletonRoot);segmentEditor();}
async function load(gender,edited=false,force=false){if(selected===gender&&!force)return;const shouldFrame=selected!==gender||!meshRoot;selected=gender;choices.value=gender;setStatus('正在加载 '+gender+' 模型...');try{const suffix=edited?'?edited=1':'';const [obj,skeleton]=await Promise.all([new OBJLoader().loadAsync('/api/model/'+gender+suffix),fetch('/api/skeleton/'+gender+suffix).then(r=>{if(!r.ok)throw Error('骨架数据读取失败');return r.json()})]);if(meshRoot)scene.remove(meshRoot);meshRoot=obj;obj.traverse(x=>{if(x.isMesh){x.geometry.computeVertexNormals();x.material=new THREE.MeshStandardMaterial({color:0x9ab7df,roughness:.72,metalness:.03,transparent:true,opacity:.42,depthWrite:false,side:THREE.DoubleSide});}});const groundOffset=.02-new THREE.Box3().setFromObject(obj).min.y;obj.position.y=groundOffset;scene.add(obj);addSkeleton(skeleton,groundOffset);if(shouldFrame)frame(obj);document.querySelector('#reset').disabled=false;setStatus('已加载 '+gender+'，可拖动黄色身体关节');}catch(e){setStatus('加载失败：'+e.message,true)}}
function updatePointer(event){const r=canvas.getBoundingClientRect();pointer.set((event.clientX-r.left)/r.width*2-1,-((event.clientY-r.top)/r.height)*2+1);raycaster.setFromCamera(pointer,camera)}
canvas.addEventListener('pointerdown',event=>{if(!selected)return;updatePointer(event);const hit=raycaster.intersectObjects(jointPickers,false)[0];if(hit){const joint=hit.object;drag={joint:joint.userData.joint,dot:joint};const worldPosition=joint.getWorldPosition(new THREE.Vector3());dragPlane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()),worldPosition);controls.enabled=false;canvas.setPointerCapture(event.pointerId);setStatus('拖动关节 '+drag.joint+'，松开后应用 '+(document.querySelector('#full-body-ik').checked?'全身 IK':'局部旋转'));event.preventDefault();return}if(segmentLines){raycaster.params.Line.threshold=.035;const lineHit=raycaster.intersectObject(segmentLines,false)[0];if(lineHit){const child=segmentLines.userData.children[Math.floor(lineHit.index/2)];if(child<=21){selectedSegment=child;segmentEditor();setStatus('已选择骨段 '+skeletonData.names[skeletonData.parents[child]]+' → '+skeletonData.names[child]);event.preventDefault()}}}});
canvas.addEventListener('pointermove',event=>{if(!drag)return;updatePointer(event);const p=new THREE.Vector3();if(raycaster.ray.intersectPlane(dragPlane,p))drag.dot.position.copy(skeletonRoot.worldToLocal(p.clone()))});
canvas.addEventListener('pointerup',async event=>{if(!drag)return;const moved=drag;drag=null;controls.enabled=true;canvas.releasePointerCapture(event.pointerId);const fullBody=document.querySelector('#full-body-ik').checked;setStatus('正在求解 '+(fullBody?'全身 IK...':'局部旋转...'));try{const r=await fetch('/api/pose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gender:selected,joint:moved.joint,target:moved.dot.position.toArray(),full_body:fullBody})});if(!r.ok)throw Error(await r.text());await r.json();await load(selected,true,true)}catch(e){setStatus('IK 求解失败：'+e.message,true);await load(selected,true,true)}});
document.querySelector('#reset').onclick=async()=>{if(!selected)return;await fetch('/api/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gender:selected})});await load(selected,true,true)};
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


class ModelCache:
    def __init__(self, model_dir: Path, cache_dir: Path):
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.lock = threading.Lock()
        self.poses = {}
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
            self.models[gender] = smplx.create(
                str(self.source(gender)), model_type="smplx", gender=gender,
                use_pca=False, flat_hand_mean=True, batch_size=1,
            )
        return self.models[gender]

    def _pose(self, gender: str, model):
        import torch
        if gender not in self.poses:
            self.poses[gender] = torch.zeros((1, model.NUM_BODY_JOINTS * 3), dtype=torch.float32)
        return self.poses[gender]

    def _output(self, gender: str):
        import torch
        if gender in self.outputs:
            return self._model(gender), self.outputs[gender]
        model = self._model(gender)
        with torch.no_grad():
            output = model(body_pose=self._pose(gender, model))
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
        for child in range(1, len(parents)):
            parent = int(parents[child])
            if parent < 0:
                continue
            config = segments.get(str(child), {})
            scale = float(config.get("length", 1.0))
            joints[child] = joints[parent] + (base_joints[child] - base_joints[parent]) * scale
        vertices = base_vertices.copy()
        deltas = joints - base_joints
        weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
        vertices += weights[:, :len(joints)] @ deltas
        for child_text, config in segments.items():
            child = int(child_text)
            parent = int(parents[child])
            if child > 21 or parent < 0:
                continue
            start, end = joints[parent], joints[child]
            axis = end - start
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1e-6:
                continue
            axis /= axis_norm
            relative = vertices - start
            raw_t = relative @ axis / axis_norm
            # A profile belongs to a finite bone segment. Do not clamp every
            # downstream vertex to t=1: that would leak the distal radius into
            # the next limb segment. A small feather keeps the joint seam smooth.
            feather = 0.08
            t = np.clip(raw_t, 0.0, 1.0)
            axial_mask = np.clip((raw_t + feather) / feather, 0.0, 1.0)
            axial_mask *= np.clip((1.0 + feather - raw_t) / feather, 0.0, 1.0)
            radial = relative - np.outer(relative @ axis, axis)
            radii = np.asarray(config.get("radii", [1.0, 1.0, 1.0]), dtype=np.float32)
            profile_t = np.linspace(0.0, 1.0, len(radii), dtype=np.float32)
            scale = np.interp(t, profile_t, radii).astype(np.float32)
            influence = np.clip(weights[:, parent] + weights[:, child], 0.0, 1.0) * axial_mask
            vertices += radial * ((scale - 1.0) * influence)[:, None]
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
        if child < 1 or child > 21 or not 0.35 <= length <= 2.5:
            raise ValueError("Invalid body segment or length")
        if not 2 <= len(radii) <= 12 or any(not 0.3 <= value <= 2.5 for value in radii):
            raise ValueError("Radius controls must contain 2-12 values between 0.3 and 2.5")
        with self.lock:
            self._segments(gender)[str(child)] = {"length": float(length), "radii": [float(value) for value in radii]}
            self._invalidate_geometry(gender)

    def reset(self, gender: str):
        if gender not in GENDERS or not self.available(gender):
            raise FileNotFoundError(gender)
        with self.lock:
            model = self._model(gender)
            import torch
            self.poses[gender] = torch.zeros((1, model.NUM_BODY_JOINTS * 3), dtype=torch.float32)
            self.outputs.pop(gender, None)
            self._invalidate_geometry(gender)

    def pose_joint(self, gender: str, joint: int, target: list[float], full_body: bool):
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
            target_tensor = torch.tensor(target, dtype=torch.float32).view(1, 3)
            with torch.no_grad():
                initial = model(body_pose=base).joints[:, :len(parents)]
            optimizer = torch.optim.Adam([variables], lr=0.035)
            editable = torch.zeros_like(variables)
            for node in chain:
                editable[:, (node - 1) * 3:node * 3] = 1
            for _ in range(36 if full_body else 24):
                optimizer.zero_grad(set_to_none=True)
                pose = variables * editable + base * (1 - editable)
                output = model(body_pose=pose)
                if full_body:
                    loss = (output.joints[:, joint] - target_tensor).square().sum()
                else:
                    loss = (output.joints[:, joint] - target_tensor).square().sum()
                loss = loss + 0.003 * ((pose - base) * editable).square().mean()
                loss.backward()
                optimizer.step()
            self.poses[gender] = (variables.detach() * editable + base * (1 - editable))
            with torch.no_grad():
                self.outputs[gender] = model(body_pose=self.poses[gender])
            self._invalidate_geometry(gender)


def handler_factory(cache: ModelCache):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def send_json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
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
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = urlparse(self.path).path
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
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(ModelCache(args.model_dir, args.cache_dir)))
    print(f"SMPL-X viewer: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
