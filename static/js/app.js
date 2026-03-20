/* ─── App state ─────────────────────────────────────────────────────────────── */

const state = {
  uploadedModelPath: null,   // server-side temp path for the uploaded GLB
  i: { segPath: null, partsPath: null, jobId: null, splitJobId: null },
  f: { segPath: null, partsPath: null, jobId: null, splitJobId: null },
  t: { segPath: null, partsPath: null, jobId: null, splitJobId: null },
  g: { imagePath: null, jobId: null },
  samplerPresets: {},
  splitPresets:   {},
};

/* ─── BabylonJS viewers ─────────────────────────────────────────────────────── */

class Viewer3D {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.engine = new BABYLON.Engine(this.canvas, true, { preserveDrawingBuffer: true });
    this.scene  = this._buildScene();
    this.engine.runRenderLoop(() => this.scene.render());
    window.addEventListener('resize', () => this.engine.resize());
  }

  _buildScene() {
    const scene = new BABYLON.Scene(this.engine);
    scene.clearColor = new BABYLON.Color4(0.047, 0.055, 0.086, 1);

    const camera = new BABYLON.ArcRotateCamera('cam', -Math.PI / 2, Math.PI / 3, 3, BABYLON.Vector3.Zero(), scene);
    camera.attachControl(this.canvas, true);
    camera.lowerRadiusLimit = 0.1;
    camera.wheelPrecision = 50;

    const hemi = new BABYLON.HemisphericLight('hemi', new BABYLON.Vector3(0, 1, 0), scene);
    hemi.intensity = 1.2;
    const dir = new BABYLON.DirectionalLight('dir', new BABYLON.Vector3(-1, -1.5, -0.5), scene);
    dir.intensity = 0.6;

    return scene;
  }

  async loadGLB(serverFilePath) {
    // Clear previous meshes
    const toRemove = this.scene.meshes.slice();
    toRemove.forEach(m => m.dispose());

    const url = `/api/files?path=${encodeURIComponent(serverFilePath)}`;
    try {
      const result = await BABYLON.SceneLoader.ImportMeshAsync('', url, '', this.scene);
      // Frame camera on loaded model
      const allMeshes = result.meshes.filter(m => m.getTotalVertices() > 0);
      if (allMeshes.length) {
        const bb = BABYLON.Mesh.MergeMeshes(allMeshes.map(m => {
          const clone = m.clone('_temp_', null, true);
          clone.makeGeometryUnique();
          return clone;
        }), true);
        if (bb) {
          const { min, max } = bb.getBoundingInfo().boundingBox;
          const size = BABYLON.Vector3.Distance(min, max);
          this.scene.cameras[0].radius = size * 1.5;
          this.scene.cameras[0].target = BABYLON.Vector3.Center(min, max);
          bb.dispose();
        }
      }
    } catch (err) {
      console.error('GLB load error:', err);
    }
  }
}

// Instantiate viewers lazily once DOM is ready
const viewers = {};
function getViewer(id) {
  if (!viewers[id]) viewers[id] = new Viewer3D(id);
  return viewers[id];
}

/* ─── Overlay helpers ───────────────────────────────────────────────────────── */

function showOverlay(id, text = 'Running…') {
  const el = document.getElementById(id);
  if (!el) return;
  el.querySelector('span').textContent = text;
  el.classList.remove('hidden');
}
function hideOverlay(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('hidden');
}

/* ─── Tab switching ─────────────────────────────────────────────────────────── */

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    // Resize BabylonJS engines on tab switch
    setTimeout(() => Object.values(viewers).forEach(v => v.engine.resize()), 50);
  });
});

/* ─── Accordion ─────────────────────────────────────────────────────────────── */

function toggleAccordion(btn) {
  btn.classList.toggle('open');
  const body = btn.nextElementSibling;
  body.classList.toggle('open');
}

/* ─── Upload zone ───────────────────────────────────────────────────────────── */

const uploadZone = document.getElementById('uploadZone');
const modelFileInput = document.getElementById('modelFile');

uploadZone.addEventListener('click', () => modelFileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', async e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) await uploadModel(file);
});
modelFileInput.addEventListener('change', async e => {
  if (e.target.files[0]) await uploadModel(e.target.files[0]);
});

async function uploadModel(file) {
  document.getElementById('uploadFilename').textContent = '⏳ Uploading…';
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/api/upload', { method: 'POST', body: fd });
  const data = await res.json();
  state.uploadedModelPath = data.path;
  document.getElementById('uploadFilename').textContent = file.name;
}

/* ─── File picker helper ────────────────────────────────────────────────────── */

function pickFile(targetId, accept, previewId) {
  const pickerId = `${targetId}-picker`;
  let picker = document.getElementById(pickerId);
  if (!picker) {
    picker = document.createElement('input');
    picker.type = 'file';
    picker.accept = accept;
    picker.id = pickerId;
    picker.style.display = 'none';
    document.body.appendChild(picker);
  }
  picker.onchange = async () => {
    const file = picker.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    document.getElementById(targetId).value = data.path;
    if (previewId) {
      const img = document.getElementById(previewId);
      img.src = URL.createObjectURL(file);
    }
  };
  picker.click();
}

/* ─── Job polling ───────────────────────────────────────────────────────────── */

function pollJob(jobId, onDone, onError, statusId) {
  const interval = setInterval(async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    if (job.status === 'done') {
      clearInterval(interval);
      setStatus(statusId, 'done', '✓ Done');
      onDone(job.result);
    } else if (job.status === 'error') {
      clearInterval(interval);
      setStatus(statusId, 'error', `✗ ${job.error}`);
      if (onError) onError(job.error);
    }
  }, 1500);
}

function setStatus(id, cls, text) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = `job-status ${cls}`;
  el.textContent = text;
}

/* ─── Download helper ───────────────────────────────────────────────────────── */

function downloadFile(path, filename) {
  if (!path) return;
  const a = document.createElement('a');
  a.href = `/api/files?path=${encodeURIComponent(path)}`;
  a.download = filename;
  a.click();
}

/* ─── Preset loading ────────────────────────────────────────────────────────── */

async function loadPresets() {
  const [sp, spl] = await Promise.all([
    fetch('/api/presets/sampler').then(r => r.json()),
    fetch('/api/presets/split').then(r => r.json()),
  ]);
  state.samplerPresets = sp;
  state.splitPresets = spl;

  // Build sampler fields for each prefix
  ['i','f','t'].forEach(p => {
    buildSamplerFields(p, sp.balanced);
    buildSplitFields(p, spl.balanced);
  });
}

/* ─── Dynamic field builders ────────────────────────────────────────────────── */

function buildSamplerFields(prefix, defaults) {
  const container = document.getElementById(`${prefix}-sampler-fields`);
  if (!container) return;
  container.innerHTML = `
    <p class="param-section-title">Sampler</p>
    ${sliderField(`${prefix}-steps`,     'Steps',                    1, 100, 1,    defaults.steps)}
    ${sliderField(`${prefix}-rescale-t`, 'Rescale T',                0.1, 5, 0.05, defaults.rescale_t)}
    ${sliderField(`${prefix}-guidance`,  'Guidance strength (CFG)',  0,  10, 0.1,  defaults.guidance_strength)}
    ${sliderField(`${prefix}-g-rescale`, 'Guidance rescale',         0,   1, 0.05, defaults.guidance_rescale)}
    ${sliderField(`${prefix}-gi-start`,  'Guidance interval — start',0,   1, 0.01, defaults.guidance_interval_start)}
    ${sliderField(`${prefix}-gi-end`,    'Guidance interval — end',  0,   1, 0.01, defaults.guidance_interval_end)}
    <p class="param-section-title">Export</p>
    ${numberField(`${prefix}-decimation`, 'Decimation target (faces)', defaults.decimation_target)}
    ${selectField(`${prefix}-tex-size`,   'Texture size (px)', [512,1024,2048,4096], defaults.texture_size)}
    ${checkField(`${prefix}-remesh`,      'Remesh', defaults.remesh)}
    ${sliderField(`${prefix}-remesh-band`, 'Remesh band',    0, 4, 1, defaults.remesh_band)}
    ${sliderField(`${prefix}-remesh-proj`, 'Remesh project', 0, 4, 1, defaults.remesh_project)}
  `;
  wireSliders(container);
}

function buildSplitFields(prefix, defaults) {
  const container = document.getElementById(`${prefix}-split-fields`);
  if (!container) return;
  container.innerHTML = `
    <p class="param-section-title">Color palette</p>
    ${sliderField(`${prefix}-cq-step`,  'Color quant step',   1,  64, 1, defaults.color_quant_step)}
    ${numberField(`${prefix}-samp-px`,  'Palette sample pixels', defaults.palette_sample_pixels)}
    ${numberField(`${prefix}-min-px`,   'Palette min pixels',   defaults.palette_min_pixels)}
    ${numberField(`${prefix}-max-col`,  'Palette max colors',   defaults.palette_max_colors)}
    ${numberField(`${prefix}-merge-d`,  'Palette merge dist',   defaults.palette_merge_dist)}
    <p class="param-section-title">Face sampling</p>
    ${selectField(`${prefix}-spf`,   'Samples per face', [1,4], defaults.samples_per_face)}
    ${checkField(`${prefix}-flip-v`,     'Flip V (glTF convention)', defaults.flip_v)}
    ${checkField(`${prefix}-uv-wrap`,    'UV wrap repeat', defaults.uv_wrap_repeat)}
    <p class="param-section-title">Boundary refinement</p>
    ${sliderField(`${prefix}-tr-thresh`, 'Transition confidence threshold', 0.25, 1, 0.25, defaults.transition_conf_thresh)}
    ${numberField(`${prefix}-tr-iters`,  'Transition propagation iterations', defaults.transition_prop_iters)}
    ${numberField(`${prefix}-tr-min`,    'Transition neighbor minimum', defaults.transition_neighbor_min)}
    <p class="param-section-title">Small component cleanup</p>
    ${selectField(`${prefix}-sc-action`, 'Small component action', ['reassign','drop'], defaults.small_component_action)}
    ${numberField(`${prefix}-sc-min`,    'Small component min faces', defaults.small_component_min_faces)}
    ${numberField(`${prefix}-pp-iters`,  'Post-process iterations', defaults.postprocess_iters)}
    <p class="param-section-title">Output</p>
    ${numberField(`${prefix}-min-faces`, 'Min faces per part', defaults.min_faces_per_part)}
    ${checkField(`${prefix}-bake-xf`,    'Bake transforms', defaults.bake_transforms)}
  `;
  wireSliders(container);
}

/* Field template helpers */
function sliderField(id, label, min, max, step, value) {
  return `<div class="param-field">
    <label>${label}</label>
    <div class="range-wrap">
      <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}"
             oninput="document.getElementById('${id}-v').textContent=parseFloat(this.value).toFixed(step<1?2:0)">
      <span id="${id}-v">${value}</span>
    </div>
  </div>`;
}
function numberField(id, label, value) {
  return `<div class="param-field">
    <label>${label}</label>
    <input type="number" id="${id}" value="${value}">
  </div>`;
}
function selectField(id, label, options, value) {
  const opts = options.map(o => `<option value="${o}"${o==value?' selected':''}>${o}</option>`).join('');
  return `<div class="param-field"><label>${label}</label><select id="${id}">${opts}</select></div>`;
}
function checkField(id, label, checked) {
  return `<div class="param-field">
    <div class="check-row">
      <input type="checkbox" id="${id}"${checked?' checked':''}>
      <label for="${id}">${label}</label>
    </div>
  </div>`;
}
function wireSliders(container) {
  // Already wired via inline oninput — no extra work needed
}

/* ─── Preset application ────────────────────────────────────────────────────── */

function applyPreset(prefix, name) {
  const p = state.samplerPresets[name];
  if (!p) return;
  setVal(`${prefix}-steps`,      p.steps);
  setVal(`${prefix}-rescale-t`,  p.rescale_t);
  setVal(`${prefix}-guidance`,   p.guidance_strength);
  setVal(`${prefix}-g-rescale`,  p.guidance_rescale);
  setVal(`${prefix}-gi-start`,   p.guidance_interval_start);
  setVal(`${prefix}-gi-end`,     p.guidance_interval_end);
  setVal(`${prefix}-decimation`, p.decimation_target);
  setVal(`${prefix}-tex-size`,   p.texture_size);
  setVal(`${prefix}-remesh`,     p.remesh);
  setVal(`${prefix}-remesh-band`,p.remesh_band);
  setVal(`${prefix}-remesh-proj`,p.remesh_project);
}

function applySplitPreset(prefix, name) {
  const p = state.splitPresets[name];
  if (!p) return;
  setVal(`${prefix}-cq-step`,   p.color_quant_step);
  setVal(`${prefix}-samp-px`,   p.palette_sample_pixels);
  setVal(`${prefix}-min-px`,    p.palette_min_pixels);
  setVal(`${prefix}-max-col`,   p.palette_max_colors);
  setVal(`${prefix}-merge-d`,   p.palette_merge_dist);
  setVal(`${prefix}-spf`,       p.samples_per_face);
  setVal(`${prefix}-flip-v`,    p.flip_v);
  setVal(`${prefix}-uv-wrap`,   p.uv_wrap_repeat);
  setVal(`${prefix}-tr-thresh`, p.transition_conf_thresh);
  setVal(`${prefix}-tr-iters`,  p.transition_prop_iters);
  setVal(`${prefix}-tr-min`,    p.transition_neighbor_min);
  setVal(`${prefix}-sc-action`, p.small_component_action);
  setVal(`${prefix}-sc-min`,    p.small_component_min_faces);
  setVal(`${prefix}-pp-iters`,  p.postprocess_iters);
  setVal(`${prefix}-min-faces`, p.min_faces_per_part);
  setVal(`${prefix}-bake-xf`,   p.bake_transforms);
}

function setVal(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  if (el.type === 'checkbox') {
    el.checked = !!value;
  } else {
    el.value = value;
    // Update range display label
    const label = document.getElementById(`${id}-v`);
    if (label) label.textContent = typeof value === 'number'
      ? (Number.isInteger(value) ? value : value.toFixed(2))
      : value;
  }
}

/* ─── Param readers ─────────────────────────────────────────────────────────── */

function getSamplerParams(prefix) {
  return {
    steps:                   parseInt(getVal(`${prefix}-steps`)),
    rescale_t:               parseFloat(getVal(`${prefix}-rescale-t`)),
    guidance_strength:       parseFloat(getVal(`${prefix}-guidance`)),
    guidance_rescale:        parseFloat(getVal(`${prefix}-g-rescale`)),
    guidance_interval_start: parseFloat(getVal(`${prefix}-gi-start`)),
    guidance_interval_end:   parseFloat(getVal(`${prefix}-gi-end`)),
    decimation_target:       parseInt(getVal(`${prefix}-decimation`)),
    texture_size:            parseInt(getVal(`${prefix}-tex-size`)),
    remesh:                  getChecked(`${prefix}-remesh`),
    remesh_band:             parseInt(getVal(`${prefix}-remesh-band`)),
    remesh_project:          parseInt(getVal(`${prefix}-remesh-proj`)),
  };
}

function getSplitParams(prefix, segPath) {
  return {
    seg_glb_path:              segPath,
    color_quant_step:          parseInt(getVal(`${prefix}-cq-step`)),
    palette_sample_pixels:     parseInt(getVal(`${prefix}-samp-px`)),
    palette_min_pixels:        parseInt(getVal(`${prefix}-min-px`)),
    palette_max_colors:        parseInt(getVal(`${prefix}-max-col`)),
    palette_merge_dist:        parseInt(getVal(`${prefix}-merge-d`)),
    samples_per_face:          parseInt(getVal(`${prefix}-spf`)),
    flip_v:                    getChecked(`${prefix}-flip-v`),
    uv_wrap_repeat:            getChecked(`${prefix}-uv-wrap`),
    transition_conf_thresh:    parseFloat(getVal(`${prefix}-tr-thresh`)),
    transition_prop_iters:     parseInt(getVal(`${prefix}-tr-iters`)),
    transition_neighbor_min:   parseInt(getVal(`${prefix}-tr-min`)),
    small_component_action:    getVal(`${prefix}-sc-action`),
    small_component_min_faces: parseInt(getVal(`${prefix}-sc-min`)),
    postprocess_iters:         parseInt(getVal(`${prefix}-pp-iters`)),
    min_faces_per_part:        parseInt(getVal(`${prefix}-min-faces`)),
    bake_transforms:           getChecked(`${prefix}-bake-xf`),
  };
}

function getVal(id) {
  const el = document.getElementById(id);
  return el ? el.value : null;
}
function getChecked(id) {
  const el = document.getElementById(id);
  return el ? el.checked : false;
}

/* ─── Run: Interactive ──────────────────────────────────────────────────────── */

async function runInteractive() {
  const glbPath = state.uploadedModelPath;
  if (!glbPath) return alert('Upload a model first.');
  const ckpt = document.getElementById('i-ckpt').value.trim();
  if (!ckpt) return alert('Enter a checkpoint path.');

  setStatus('i-status', 'running', '⏳ Segmenting…');
  showOverlay('i-overlay-seg', 'Segmenting…');
  document.getElementById('i-run-btn').disabled = true;

  const params = {
    glb_path: glbPath,
    ckpt_path: ckpt,
    transforms_path: document.getElementById('i-transforms').value,
    rendered_img: document.getElementById('i-rendered-img').value || null,
    points_str: document.getElementById('i-points').value,
    ...getSamplerParams('i'),
  };
  const res = await fetch('/api/jobs/interactive', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params) });
  const { job_id } = await res.json();
  state.i.jobId = job_id;

  pollJob(job_id, async (result) => {
    state.i.segPath = result;
    document.getElementById('i-run-btn').disabled = false;
    hideOverlay('i-overlay-seg');
    showOverlay('i-overlay-seg', 'Loading 3D…');
    await getViewer('i-canvas-seg').loadGLB(result);
    hideOverlay('i-overlay-seg');
  }, () => {
    document.getElementById('i-run-btn').disabled = false;
    hideOverlay('i-overlay-seg');
  }, 'i-status');
}

/* ─── Run: Full ─────────────────────────────────────────────────────────────── */

async function runFull() {
  const glbPath = state.uploadedModelPath;
  if (!glbPath) return alert('Upload a model first.');
  const ckpt = document.getElementById('f-ckpt').value.trim();
  if (!ckpt) return alert('Enter a checkpoint path.');

  setStatus('f-status', 'running', '⏳ Segmenting…');
  showOverlay('f-overlay-seg', 'Segmenting…');
  document.getElementById('f-run-btn').disabled = true;

  const params = {
    glb_path: glbPath,
    ckpt_path: ckpt,
    transforms_path: document.getElementById('f-transforms').value,
    rendered_img: document.getElementById('f-rendered-img').value || null,
    ...getSamplerParams('f'),
  };
  const res = await fetch('/api/jobs/full', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params) });
  const { job_id } = await res.json();
  state.f.jobId = job_id;

  pollJob(job_id, async (result) => {
    state.f.segPath = result;
    document.getElementById('f-run-btn').disabled = false;
    hideOverlay('f-overlay-seg');
    showOverlay('f-overlay-seg', 'Loading 3D…');
    await getViewer('f-canvas-seg').loadGLB(result);
    hideOverlay('f-overlay-seg');
  }, () => {
    document.getElementById('f-run-btn').disabled = false;
    hideOverlay('f-overlay-seg');
  }, 'f-status');
}

/* ─── Run: Full 2D ──────────────────────────────────────────────────────────── */

async function runFull2D() {
  const glbPath = state.uploadedModelPath;
  if (!glbPath) return alert('Upload a model first.');
  const ckpt = document.getElementById('t-ckpt').value.trim();
  if (!ckpt) return alert('Enter a checkpoint path.');
  const guidanceImg = document.getElementById('t-guidance-img').value.trim();
  if (!guidanceImg) return alert('Select a 2D guidance map image.');

  setStatus('t-status', 'running', '⏳ Segmenting…');
  showOverlay('t-overlay-seg', 'Segmenting…');
  document.getElementById('t-run-btn').disabled = true;

  const params = {
    glb_path: glbPath,
    ckpt_path: ckpt,
    guidance_img: guidanceImg,
    ...getSamplerParams('t'),
  };
  const res = await fetch('/api/jobs/full_2d', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params) });
  const { job_id } = await res.json();
  state.t.jobId = job_id;

  pollJob(job_id, async (result) => {
    state.t.segPath = result;
    document.getElementById('t-run-btn').disabled = false;
    hideOverlay('t-overlay-seg');
    showOverlay('t-overlay-seg', 'Loading 3D…');
    await getViewer('t-canvas-seg').loadGLB(result);
    hideOverlay('t-overlay-seg');
  }, () => {
    document.getElementById('t-run-btn').disabled = false;
    hideOverlay('t-overlay-seg');
  }, 't-status');
}

/* ─── Run: Split ────────────────────────────────────────────────────────────── */

async function runSplit(prefix) {
  const segPath = state[prefix].segPath;
  if (!segPath) return alert('Run segmentation first.');

  setStatus(`${prefix}-status`, 'running', '⏳ Splitting…');
  showOverlay(`${prefix}-overlay-parts`, 'Splitting…');

  const params = getSplitParams(prefix, segPath);
  const res = await fetch('/api/jobs/split', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params) });
  const { job_id } = await res.json();
  state[prefix].splitJobId = job_id;

  pollJob(job_id, async (result) => {
    state[prefix].partsPath = result;
    setStatus(`${prefix}-status`, 'done', '✓ Split done');
    hideOverlay(`${prefix}-overlay-parts`);
    showOverlay(`${prefix}-overlay-parts`, 'Loading 3D…');
    await getViewer(`${prefix}-canvas-parts`).loadGLB(result);
    hideOverlay(`${prefix}-overlay-parts`);
  }, () => {
    hideOverlay(`${prefix}-overlay-parts`);
  }, `${prefix}-status`);
}

/* ─── Run: Guidance ─────────────────────────────────────────────────────────── */

async function runGuidance() {
  const glbPath = document.getElementById('g-glb').value.trim() || state.uploadedModelPath;
  if (!glbPath) return alert('Upload a model or enter a GLB path.');
  const apiKey = document.getElementById('g-api-key').value.trim();
  if (!apiKey) return alert('Enter a Gemini API key.');

  const mode = document.querySelector('input[name="g-mode"]:checked').value;
  const gridViews = ['front','back','left','right','top','bottom'].filter(v =>
    document.getElementById(`g-${v}`)?.checked
  );

  setStatus('g-status', 'running', '⏳ Generating…');
  document.getElementById('g-run-btn').disabled = true;

  const params = {
    glb_path: glbPath,
    transforms_path: document.getElementById('g-transforms').value,
    gemini_api_key: apiKey,
    analyze_model: document.getElementById('g-analyze-model').value,
    generate_model: document.getElementById('g-generate-model').value,
    resolution: parseInt(document.getElementById('g-resolution').value),
    mode: mode,
    grid_views: mode === 'grid' ? gridViews : [],
    grid_cols: parseInt(document.getElementById('g-grid-cols').value),
  };
  const res = await fetch('/api/jobs/guidance', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params) });
  const { job_id } = await res.json();
  state.g.jobId = job_id;

  pollJob(job_id, (result) => {
    state.g.imagePath = result.image_path;
    document.getElementById('g-run-btn').disabled = false;
    setStatus('g-status', 'done', '✓ Done');
    // Show output
    const outRow = document.getElementById('g-output-row');
    outRow.style.display = '';
    document.getElementById('g-output-img').src = `/api/files?path=${encodeURIComponent(result.image_path)}`;
    document.getElementById('g-output-json').textContent = JSON.stringify(result.description, null, 2);
  }, () => {
    document.getElementById('g-run-btn').disabled = false;
  }, 'g-status');
}

/* ─── Use guidance map in Tab 3 ─────────────────────────────────────────────── */

function useAsGuidance() {
  if (!state.g.imagePath) return;
  document.getElementById('t-guidance-img').value = state.g.imagePath;
  // Show preview
  const img = document.getElementById('t-guidance-preview');
  img.src = `/api/files?path=${encodeURIComponent(state.g.imagePath)}`;
  // Switch to Full 2D tab
  document.querySelector('[data-tab="full2d"]').click();
}

/* ─── Grid controls toggle ──────────────────────────────────────────────────── */

function toggleGridControls(mode) {
  const el = document.getElementById('g-grid-controls');
  el.style.display = mode === 'grid' ? '' : 'none';
}

/* ─── Init ──────────────────────────────────────────────────────────────────── */

// Pre-initialize viewers so they're ready
window.addEventListener('DOMContentLoaded', async () => {
  // Eagerly init the first tab's viewers
  ['i-canvas-seg','i-canvas-parts'].forEach(id => getViewer(id));

  // Load presets and build parameter UI
  await loadPresets();
});
