import { DyNCA, ImagePreprocessor } from './dynca.js'

function isInViewport(element) {
  var rect = element.getBoundingClientRect();
  var html = document.documentElement;
  var w = window.innerWidth || html.clientWidth;
  var h = window.innerHeight || html.clientHeight;
  return rect.top < h && rect.left < w && rect.bottom > 0 && rect.right > 0;
}

export function createDemo(divId, imageCapture) {
  const root = document.getElementById(divId);
  const $ = q => root.querySelector(q);
  const $$ = q => root.querySelectorAll(q);



  // const W = 256, H = 256;
  const resolutions = [64, 96, 128, 192, 256];
  var W = 128, H = 128;
  var last_resolution_idx = 2;
  let ca = null;
  let paused = false;

  const canvas = $('#demo-canvas');
  canvas.width = W * 6; //so we can render hexells
  canvas.height = H * 6;
  let gl = canvas.getContext("webgl2");


  // const test_canvas = $('#image-canvas');
  // test_canvas.width = W * 4; //so we can render hexells
  // test_canvas.height = H * 4;
  // let test_gl = test_canvas.getContext("webgl2");
  //
  // let imageProcessor = new ImagePreprocessor(test_gl, [W * 6, H * 6]);

  if (!gl) {
    console.log('your browser/OS/drivers do not support WebGL2');
    console.log('Switching to WebGL1');
    const gl = canvas.getContext("webgl2");
    const ext1 = gl.getExtension('OES_texture_float');
    if (!ext1) {
      console.log("Sorry, your browser does not support OES_texture_float. Use a different browser");
      // return;
    }

  } else {
    console.log('webgl2 works!');
    const ext2 = gl.getExtension('EXT_color_buffer_float');
    if (!ext2) {
      console.log("Sorry, your browser does not support  EXT_color_buffer_float. Use a different browser");
      // return;
    }
  }

  gl.disable(gl.DITHER);
  // test_gl.disable(test_gl.DITHER);


  twgl.addExtensionsToContext(gl);
  // twgl.addExtensionsToContext(test_gl);

  const maxZoom = 32.0;

  const params = {
    // modelSet: 'demo/models.json',
    // modelSet: 'demo/test2.json',
    // modelSet: 'demo/test_pos.json',
    // modelSet: 'data/test3.json',
    metadataJson: 'data/metadata.json',
    metadata: null,
    models: null,
    model_type: "large",

    brushSize: 16,
    autoFill: true,
    debug: false,
    our_version: true,
    zoom: 1.0,
    alignment: 0,
    rotationAngle: 0,
    rate: 1.0,

    texture_name: "flames",
    motion_name: "0",
    video_name: "water_3",


    texture_img: null,
    motion_img: null,
    video_gif: null,

    texture_idx: 0,
    motion_idx: 1,
  };

  let metadata = null;
  let exp_type = "VectorFieldMotion";
  // let exp_type = "VideoMotion"

  let gui = null;
  let currentTexture = null;
  let currentMotion = null;
  // const initTexture = "interlaced_0172";

  const initTexture = "flames";
  const initVideo = "water_3";
  const initMotion = "up";


  initMetaData();

  async function initMetaData(load_meta_data = true) {
    if (load_meta_data) {
      const r = await fetch(params.metadataJson);
      metadata = await r.json();
      params.metadata = metadata;
    } else {
      metadata = params.metadata
    }


    let texture_names = metadata['texture_names'];
    // let texture_images = metadata['texture_images'];

    let motion_names = metadata['motion_names'];
    let motion_images = metadata['motion_images'];

    // let vec_field_model_files = metadata['vec_field_model_files'];
    // let video_model_files = metadata['video_model_files'];

    let video_names = metadata['video_names'];
    // let video_appearance_images = metadata['video_appearance'];
    // let video_gifs = metadata['video_gifs'];

    async function setTextureModel(idx) {
      if (exp_type == "VectorFieldMotion") {
        params.texture_name = texture_names[idx];
        params.texture_img = "images/texture/" + texture_names[idx] + ".jpg"
        params.modelSet = "data/vec_field_models/large/" + texture_names[idx] + ".json"
        // params.modelSet = vec_field_model_files[params.model_type][idx]
        // params.texture_img = texture_images[idx];
      }
      params.texture_idx = idx;
      updateUI();
      updateCA();

    }

    let len = (exp_type == "VectorFieldMotion") ? texture_names.length : video_names.length;
    for (let idx = 0; idx < len; idx++) {
      let media_path = "";
      let texture_name = "";
      if (exp_type == "VectorFieldMotion") {
        texture_name = texture_names[idx];
        media_path = params.texture_img = "images/texture/" + texture_name + ".jpg"
      } else {
        texture_name = video_names[idx];
        media_path = "images/picked_video_frames/" + texture_name + ".png";
      }

      // let media_path = (exp_type == "VectorFieldMotion") ? texture_images[idx] : video_appearance_images[idx];
      // let media_path = texture_images[idx];

      // let texture_name = (exp_type == "VectorFieldMotion") ? texture_names[idx] : video_names[idx];
      const texture = document.createElement('div');
      texture.style.background = "url('" + media_path + "')";
      texture.style.backgroundSize = "100%100%";
      // texture.style.backgroundSize = "100px100px";
      texture.id = name; //html5 support arbitrary id:s
      texture.className = 'texture-square';
      texture.onclick = () => {
        // removeOverlayIcon();
        if (currentTexture){
          currentTexture.style.borderColor = "white";
        }
        currentTexture = texture;
        texture.style.borderColor = "rgb(245 140 44)";
        if (!window.matchMedia('(min-width: 500px)').matches && navigator.userAgent.includes("Chrome")) {
          texture.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" })
        }
        setTextureModel(idx);
      };
      let gridBox = $('#texture');

      if (exp_type == "VectorFieldMotion") {
        if (texture_name == initTexture) {
          currentTexture = texture;
          texture.style.borderColor = "rgb(245 140 44)";
          gridBox.prepend(texture);

        } else {
          gridBox.insertBefore(texture, gridBox.lastElementChild);
        }
      } else {
        if (texture_name == initVideo) {
          currentTexture = texture;
          texture.style.borderColor = "rgb(245 140 44)";
          gridBox.prepend(texture);

        } else {
          gridBox.insertBefore(texture, gridBox.lastElementChild);
        }
      }


    }
    setTextureModel(0);

    function setMotionModel(idx) {
      params.motion_idx = idx;
      params.motion_name = motion_names[idx];
      params.motion_img = motion_images[idx];
      updateUI();
      if (ca != null) {
        ca.clearCircle(0, 0, 1000);
        ca.paint(0, 0, 10000, params.motion_idx, [0, 0]);
      }

      // updateUI();
    }

    if (exp_type == "VectorFieldMotion") {
      for (let idx = 0; idx < motion_names.length; idx++) {
        let motion_name = motion_names[idx];
        const motion = document.createElement('div');
        motion.style.background = "url('" + motion_images[idx] + "')";
        motion.style.backgroundSize = "100%100%";
        motion.id = name; //html5 support arbitrary id:s
        motion.className = 'texture-square';
        motion.onclick = () => {
          // removeOverlayIcon();
          currentMotion.style.borderColor = "white";
          currentMotion = motion;
          motion.style.borderColor = "rgb(245 140 44)";
          if (!window.matchMedia('(min-width: 500px)').matches && navigator.userAgent.includes("Chrome")) {
            motion.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" })
          }
          setMotionModel(idx);
        };


      }
      setMotionModel(params.motion_idx);
    }


  }


  function removeOverlayIcon() {
    $$(".overlayicon").forEach(sel2 => {
      sel2.style.opacity = 0.0; //"rgba(255, 255, 255, 0.0)";
    });
  }

  function createGUI(models) {
    if (gui != null) {
      gui.destroy();
    }
    gui = new dat.GUI();
    if (!params.debug) {
      dat.GUI.toggleHide();
    }
    const brush2idx = Object.fromEntries(models.model_names.map((s, i) => [s, i]));
    params.modelname = models.model_names[params.model];
    gui.add(params, 'brushSize').min(1).max(32).step(1);
    gui.add(params, 'zoom').min(1).max(20);

  }

  function canvasToGrid(x, y) {
    const [w, h] = ca.gridSize;
    const gridX = x / canvas.clientWidth * w;
    const gridY = y / canvas.clientHeight * h;
    return [gridX, gridY];
  }

  function getMousePos(e) {
    return canvasToGrid(e.offsetX, e.offsetY);
  }

  function createCA() {
    ca = new DyNCA(gl, params.models, [W, H], gui, params.our_version);
    if (exp_type == "VectorFieldMotion") {
      ca.paint(0, 0, 10000, params.motion_idx, [0.5, 0.5]);
    } else {
      // ca.paint(0, 0, 10000, params.texture_idx, [0.5, 0.5]);
      ca.paint(0, 0, 10000, 0, [0.5, 0.5]);
    }

    ca.clearCircle(0, 0, 1000);
    ca.alignment = params.alignment;
    ca.rotationAngle = params.rotationAngle

  }


  function getTouchPos(touch) {
    const rect = canvas.getBoundingClientRect();
    return canvasToGrid(touch.clientX - rect.left, touch.clientY - rect.top);
  }

  let prevPos = [0, 0]

  function click(pos) {
    const [x, y] = pos;
    const [px, py] = prevPos;
    let brushSize = params.brushSize * W / 128.0
    ca.clearCircle(x, y, brushSize, null, params.zoom);
    // ca.paint(x, y, params.brushSize, params.model, [x - px, y - py]);
    prevPos = pos;
  }


  function updateUI() {
    $('#play').style.display = paused ? "inline" : "none";
    $('#pause').style.display = !paused ? "inline" : "none";

    const speed = parseInt($('#speed').value);
    $('#speedLabel').innerHTML = ['1/60 x', '1/30', '1/10 x', '1/2 x', '1x', '2x', '4x', '6x', '<b>max</b>'][speed + 4];

    const resolution_idx = parseInt($('#resolution').value);
    $('#resolutionLabel').innerHTML = ['64x64', '96x96', '128x128', '192x192', '256x256'][resolution_idx + 2];
    W = resolutions[resolution_idx + 2]
    H = resolutions[resolution_idx + 2]
    canvas.width = W * 6;
    canvas.height = H * 6;

    if (resolution_idx != last_resolution_idx) {
      createCA();
      last_resolution_idx = resolution_idx;
    }
    // ca = new CA(gl, models, [W, H], gui, params.our_version);
    // ca.paint(0, 0, 10000, params.model, [0.5, 0.5]);


    params.rotationAngle = parseInt($('#rotation').value);
    $('#rotationLabel').innerHTML = params.rotationAngle + " deg";

    // params.rate = parseFloat($('#rate').value);
    // $('#rateLabel').innerHTML = params.rate.toFixed(2);


    $("#origtex").style.background = "url('" + params.texture_img + "')";
    $("#origtex").style.backgroundSize = "100%100%";
    let dtd = document.createElement('p')
    dtd.innerHTML = "<b>Current Style: </b><em>" + params.texture_name + "</em>"
    // dtd.href = "https://www.robots.ox.ac.uk/~vgg/data/dtd/"
    $("#texhinttext").innerHTML = '';
    $("#texhinttext").appendChild(dtd);



    $('#zoomOut').classList.toggle('disabled', params.zoom <= 1.0);
    $('#zoomIn').classList.toggle('disabled', params.zoom >= maxZoom);
  }

  function clearPalette() {
    let textureGridBox = $('#texture');
    textureGridBox.innerHTML = '<div class="whitespace"></div>';
    let motionGridBox = $('#motion');
    motionGridBox.innerHTML = '<div class="whitespace"></div>';

    $("#origtex").style.background = '';
    $("#texhinttext").innerHTML = '';

  }

  function initUI() {

    $('#play-pause').onclick = () => {
      paused = !paused;
      updateUI();
    };
    $('#reset').onclick = () => {

      ca.paint(0, 0, 10000, 0, [0.5, 0.5]);

      ca.clearCircle(0, 0, 1000);

      // ca.clearCircle(0, 0, 1000);
      // ca.paint(0, 0, 10000, params.model, [0, 0]);
    };
    $('#benchmark').onclick = () => {
      ca.benchmark();
    };

    $$('#alignSelect input').forEach((sel, i) => {
      sel.onchange = () => {
        params.alignment = i
      }
    });

    $$('#brushSelect input').forEach((sel, i) => {
      sel.onchange = () => {
        if (i == 0) {
          params.brushSize = 4;
        } else {
          if (i == 1) {
            params.brushSize = 8;
          } else {
            params.brushSize = 16;
          }
        }
      }
    });
    $('#speed').onchange = updateUI;
    $('#speed').oninput = updateUI;
    $('#rotation').onchange = updateUI;
    $('#rotation').oninput = updateUI;
    $('#resolution').onchange = updateUI;
    $('#resolution').oninput = updateUI;

    // $('#rate').onchange = updateUI;
    // $('#rate').oninput = updateUI;


    $('#zoomIn').onclick = () => {
      if (params.zoom < maxZoom) {
        params.zoom *= 2.0;
      }
      updateUI();
    };
    $('#zoomOut').onclick = () => {
      if (params.zoom > 1.0) {
        params.zoom /= 2.0;
      }
      updateUI();
    };


    canvas.onmousedown = e => {
      e.preventDefault();
      if (e.buttons == 1) {
        click(getMousePos(e));
      }
    }
    canvas.onmousemove = e => {
      e.preventDefault();
      if (e.buttons == 1) {
        click(getMousePos(e));
      }
    }
    canvas.addEventListener("touchstart", e => {
      e.preventDefault();
      click(getTouchPos(e.changedTouches[0]));
    });
    canvas.addEventListener("touchmove", e => {
      e.preventDefault();
      for (const t of e.touches) {
        click(getTouchPos(t));
      }
    });
    updateUI();
  }

  async function updateCA() {
    // Fetch models from json file
    const firstTime = ca == null;

    const r = await fetch(params.modelSet);
    const models = await r.json();
    params.models = models;
    createCA();

    window.ca = ca;
    if (firstTime) {
      createGUI(models);
      initUI();
      requestAnimationFrame(render);
    }
    updateUI();
  }

  // updateCA();

  let lastDrawTime = 0;
  let stepsPerFrame = 1;
  let frameCount = 0;

  let first = true;

  function render(time) {
    // imageProcessor.render();
    if (!isInViewport(canvas)) {
      requestAnimationFrame(render);
      return;
    }

    if (first) {
      first = false;
      requestAnimationFrame(render);
      return;
    }

    ca.rotationAngle = params.rotationAngle;
    ca.alignment = params.alignment;
    ca.rate = params.rate;
    // ca.hexGrid = params.hexGrid;

    if (!paused) {
      const speed = parseInt($("#speed").value);
      if (speed <= 0) {  // slow down by skipping steps
        const skip = [1, 2, 10, 30, 60][-speed];
        // alert(skip)
        stepsPerFrame = (frameCount % skip) ? 0 : 1;
        // alert(stepsPerFrame)
        frameCount += 1;
      } else if (speed > 0) { // speed up by making more steps per frame
        const interval = time - lastDrawTime;
        stepsPerFrame += interval < 20.0 ? 1 : -1;
        stepsPerFrame = Math.max(1, stepsPerFrame);
        stepsPerFrame = Math.min(stepsPerFrame, [1, 2, 4, 6, Infinity][speed])
        // stepsPerFrame = 600;
      }
      for (let i = 0; i < stepsPerFrame; ++i) {
        // ca.step("preprocess_image");
        // ca.step("display_image");
        ca.step();
      }
      // $("#stepCount").innerText = ca.getStepCount();
      // $("#ips").innerText = ca.fps();
    }
    lastDrawTime = time;

    twgl.bindFramebufferInfo(gl);
    ca.draw(params.zoom);
    requestAnimationFrame(render);
  }
}
