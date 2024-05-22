/*
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
Usage:
  const gui = new dat.GUI();
  const ca = new CA(gl, models_json, [W, H], gui); // gui is optional
  ca.step();
  
  ca.paint(x, y, radius, modelIndex);
  ca.clearCircle(x, y, radius;

  const stats = ca.benchmark();
  ca.draw();
  ca.draw(zoom);
*/


// Maps the position which is in [-1.0, 1.0]
// to uv which is in [0, 1]
const vs_code = `
    attribute highp vec4 position;
    varying highp vec2 uv;
    void main() {
        uv = position.xy*0.5 + 0.5;
        gl_Position = position;
    }
`

const fs_code = `
    precision highp float;
    varying highp vec2 uv;
    uniform sampler2D u_tex;

    void main() {
        // invert the y axis
        vec2 uv_ = vec2(uv.x, 1.0 - uv.y);

        // read the texture
        vec4 color = texture2D(u_tex, uv_);

        // convert to greyscale
        float grey = dot(color.rgb, vec3(0.299, 0.587, 0.114));

        gl_FragColor = vec4(grey, grey, grey, 1.0);
    }

    `

function defInput(name) {
    return `
        uniform Tensor ${name};
        uniform sampler2D ${name}_tex;

        vec4 ${name}_read(vec2 pos, float ch) {return _read(${name}, ${name}_tex, pos, ch);}
        vec4 ${name}_read01(vec2 pos, float ch) {return _read01(${name}, ${name}_tex, pos, ch);}
        vec4 ${name}_readUV(vec2 uv) {return _readUV(${name}, ${name}_tex, uv);}
    `
}

const PREFIX = `
    // #ifdef GL_FRAGMENT_PRECISION_HIGH
    //     precision highp float;
    //     precision highp sampler2D;
    //     precision highp int;
    // #else
    //     precision mediump float;
    //     precision mediump sampler2D;
    //     precision mediump int;
    // #endif
    
    precision highp float;
    precision highp sampler2D;
    precision highp int;
    
    

    // "Hash without Sine" by David Hoskins (https://www.shadertoy.com/view/4djSRW)
    float hash13(vec3 p3) {
      p3  = fract(p3 * .1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
    }
    vec2 hash23(vec3 p3)
    {
        p3 = fract(p3 * vec3(.1031, .1030, .0973));
        p3 += dot(p3, p3.yzx+33.33);
        return fract((p3.xx+p3.yz)*p3.zy);
    }

    struct Tensor {
        vec2 size;
        vec2 gridSize;
        float depth, depth4;
        vec2 packScaleZero;
        
        // tensor.gridSize is the size of the rectangle 
        // containing the channel information of the tensor
        // tensor.size is the spatial size of the original tensor
        // depth: Number of channels
        // depth4: Number of channels // 4
    };
    uniform Tensor u_output;

    vec4 _readUV(Tensor tensor, sampler2D tex, vec2 uv) {
        highp vec4 v = texture2D(tex, uv);
        highp vec2 p = tensor.packScaleZero;
        // p.y is the bias
        // p.x is the scaling factor
        // the sampled texture values is between 0.0 and 1.0 (?) 
        v = (v-p.y)*p.x;
        return v;
    }
    vec2 _getUV(Tensor tensor, vec2 pos, float ch) {
        // pos is the absolute coordinate
        // uv is the texture coordinate
        
        ch += 0.5;
        // [tx, ty] the offset to move to get the desired channel
        float tx = floor(mod(ch, tensor.gridSize.x));
        float ty = floor(ch / tensor.gridSize.x);
        
        #ifdef OURS
            // vec2 p = clamp(pos / tensor.size, 0.0, 1.0 - 1.0 / tensor.size.y); // replicate padding
            highp vec2 p = clamp(pos, vec2(0.0, 0.0), tensor.size - 0.5); // replicate padding
            p = p / tensor.size;
            // vec2 p = clamp(pos / tensor.size, 0.0, 1.0 - 0.0 / tensor.size.y); // replicate padding
        #else
            highp vec2 p = fract(pos/tensor.size); // circular padding
        #endif 
        
         
        p += vec2(tx, ty); 
        
        p /= tensor.gridSize;
        
        // the output p is in range [0.0, 1.0] 
        
        return p;
    }
    vec4 _read01(Tensor tensor, sampler2D tex, vec2 pos, float ch) {
        // Returns the scaled value of the tensor (between 0.0 and 1.0)
        return texture2D(tex, _getUV(tensor, pos, ch));
    }
    vec4 _read(Tensor tensor, sampler2D tex, vec2 pos, float ch) {
        // Returns the correct value of the tensor
        highp vec2 p = _getUV(tensor, pos, ch);
        return _readUV(tensor, tex, p);
    }
    vec2 getOutputXY() {
        // gl_FragCoord is the coordinate in the texture
        // which contains the tensor information
        // If the original tensor is 3x3 and has 32 channels then
        // The first channel of the texture would look like this
        // 0 0 0 1 1 1 2 2 2 3 3 3
        // 0 0 0 1 1 1 2 2 2 3 3 3
        // 0 0 0 1 1 1 2 2 2 3 3 3
        // 4 4 4 5 5 5 6 6 6 7 7 7 
        // 4 4 4 5 5 5 6 6 6 7 7 7
        // 4 4 4 5 5 5 6 6 6 7 7 7
        
        // Taking the mode with respect to the output size
        // will give us the spatial index in the original tensor

        highp vec2 xy = mod(gl_FragCoord.xy, u_output.size);  
        
        return xy;
        
    }
    float getOutputChannel() {
        highp vec2 xy = floor(gl_FragCoord.xy/u_output.size);
        return xy.y*u_output.gridSize.x+xy.x;
    }

    void setOutput(vec4 v) {
        highp vec2 p = u_output.packScaleZero;
        v = v/p.x + p.y;
        
        #ifndef OURS
            v = clamp(v, -2.0, 2.0);
        #else    
            // v = clamp(v, -6.0, 6.0);
        #endif
        gl_FragColor = v;
    }

    #ifdef SPARSE_UPDATE
        uniform highp sampler2D u_shuffleTex, u_unshuffleTex;
        uniform highp vec2 u_shuffleOfs;
    #endif

    ${defInput('u_input')}

    uniform float u_angle, u_alignment;
    uniform float u_hexGrid;
    uniform highp vec2 HW;
    
    mat2 rotate(float ang) {
        float s = sin(ang), c = cos(ang);
        return mat2(c, s, -s, c);
    }

    vec2 getCellDirection(vec2 xy) {
        vec2 dir = vec2(0.0, 1.0);
        if (u_alignment == 1.0) {
            dir = normalize(xy-0.5*u_input.size);
        } else if (u_alignment == 2.0) {
            vec2 v1 = xy-0.25*u_input.size;
            vec2 v2 = 0.75*u_input.size-xy;
            dir = normalize(v1/pow(length(v1), 3.0) +  v2/pow(length(v2), 3.0));
        }
        dir = rotate(u_angle) * dir;
        return dir;
    }
    // https://www.shadertoy.com/view/Xljczw
    // https://www.shadertoy.com/view/MlXyDl
    // returns xy - in cell pos, zw - skewed cell id
    vec4 getHex(vec2 u) {
        vec2 s = vec2(1., mix(2.0, 1.732, u_hexGrid));
        vec2 p = vec2(0.5*u_hexGrid, 0.5);
        vec2 a = mod(u    ,s)*2.-s;
        vec2 b = mod(u+s*p,s)*2.-s;
        vec2 ai = floor(u/s);
        vec2 bi = floor(u/s+p);
        // skewed coords
        ai = vec2(ai.x-ai.y*u_hexGrid, ai.y*2.0+1.0);
        bi = vec2(bi.x-bi.y*u_hexGrid, bi.y*2.0);
        return dot(a,a)<dot(b,b) ? vec4(a, ai) : vec4(b, bi);    
    }

    vec2 hex2screen(vec2 u) {
        return vec2(u.x + u.y/2.0, u.y*1.732/2.0); 
    }

    const highp mat3 sobelX = mat3(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0)/8.0;
    const highp mat3 sobelY = mat3(-1.0,-2.0,-1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0)/8.0;
    const highp mat3 gauss = mat3(1.0, 2.0, 1.0, 2.0, 4.0-16.0, 2.0, 1.0, 2.0, 1.0)/8.0;
    const highp mat3 sobelXhex = mat3( 0.0,    -1.0, 1.0, 
                                       -2.0, 0.0, 2.0, 
                                         -1.0, 1.0,        0.0)/8.0;

    const highp mat3 sobelYhex = mat3( 0.0,    -2.0,-2.0, 
                                        0.0, 0.0, 0.0, 
                                          2.0, 2.0,        0.0)/8.0;

    const highp mat3 gaussHex = mat3(0.0,       2.0, 2.0, 
                                       2.0, 4.0-16.0, 2.0, 
                                          2.0, 2.0,        0.0)/8.0;

    vec4 conv3x3(vec2 xy, float inputCh, mat3 filter) {
        highp vec4 a = vec4(0.0);
        for (int y=0; y<3; ++y)
        for (int x=0; x<3; ++x) {
          highp vec2 p = xy+vec2(float(x-1), float(y-1));
          a += filter[y][x] * u_input_read(p, inputCh);
        }
        return a;
    }

    // TANH Function (Hyperbolic Tangent)
    float tanh(float val)
    {

        float tmp = exp(val);
        float tanH = (tmp - 1.0 / tmp) / (tmp + 1.0 / tmp);
        return tanH;
        

    }
`;

const PROGRAMS = {
    // preprocess_image: `
    // // Fragment shader to make image grey scale

    // `,
    bilinear_downsample: `
    void main() {
        vec2 xy = floor(getOutputXY()) * 2.0 + 0.5;  // taking floor is necessary
        float ch = getOutputChannel();
        vec4 state1 = u_input_read(xy, ch);
        vec4 state2 = u_input_read(xy + vec2(1.0, 0.0), ch);
        vec4 state3 = u_input_read(xy + vec2(0.0, 1.0), ch);
        vec4 state4 = u_input_read(xy + vec2(1.0, 1.0), ch);
        vec4 state = 0.25 * (state1 + state2 + state3 + state4);
        setOutput(state);
        
    }
    `,
    bilinear_upsample_add: `
    ${defInput('u_perception0')}
    uniform bool scale_zero;
    
    void main() {
        vec2 xy = getOutputXY();
        float ch = getOutputChannel();
        
        vec2 realXY = xy;
        #ifdef SPARSE_UPDATE
            if (scale_zero) {
            // realXY = texture2D(u_shuffleTex, xy/u_output.size).xy*255.0 + 0.5 + u_shuffleOfs;
            // realXY = texture2D(u_shuffleTex, xy/u_output.size).xy*(HW.x - 1.0) + 0.5 + u_shuffleOfs;
            realXY = texture2D(u_shuffleTex, xy/u_output.size).xy + 0.5 + u_shuffleOfs;
            realXY = mod(realXY, HW);
            } 
        #endif
        
        vec4 p_current = u_perception0_read(xy, ch);
        vec2 center = floor(realXY) + 0.5;
        
        vec2 p = floor((center + 0.5) / 2.0);
        // vec2 p = floor(realXY / 2.0);

        vec2 p1 = p + 0.5;
        vec2 p2 = p - vec2(1.0, 0.0) + 0.5;
        vec2 p3 = p - vec2(0.0, 1.0) + 0.5;
        vec2 p4 = p - vec2(1.0, 1.0) + 0.5;
        
        // vec2 p1 = clamp(p, vec2(0.0, 0.0), u_input.size - 1.0) + 0.5;
        // vec2 p2 = clamp(p - vec2(1.0, 0.0), vec2(0.0, 0.0), u_input.size - 1.0) + 0.5;
        // vec2 p3 = clamp(p - vec2(0.0, 1.0), vec2(0.0, 0.0), u_input.size - 1.0) + 0.5;
        // vec2 p4 = clamp(p - vec2(1.0, 1.0), vec2(0.0, 0.0), u_input.size - 1.0) + 0.5;
        
        vec4 state1 = u_input_read(p1, ch); 
        vec4 state2 = u_input_read(p2, ch); 
        vec4 state3 = u_input_read(p3, ch); 
        vec4 state4 = u_input_read(p4, ch); 
        
        // p = p + 0.5;        
        float w1 = center.x + 1.0 - 2.0 * p.x;
        float w2 = center.y + 1.0 - 2.0 * p.y;
        float w3 = 2.0 * p.x + 1.0 - center.x;
        float w4 = 2.0 * p.y + 1.0 - center.y;

        vec4 state = 0.25 * (state1 * w1 * w2
                           + state2 * w3 * w2
                           + state3 * w1 * w4
                           + state4 * w3 * w4);
        // setOutput(state4);
        setOutput(0.5 * (state + p_current));
        // setOutput(p_current);
        
    }
    `,
    paint: `
    uniform highp vec2 u_pos;
    uniform float u_r;
    uniform highp vec4 u_brush;
    uniform float u_zoom;

    void main() {

        vec2 xy = u_pos;
        xy = (xy + u_output.size*(0.5)*(u_zoom-1.0))/u_zoom;
        vec2 xy_out = getOutputXY();
        if (u_hexGrid > 0.0) {
            // vec4 r = getHex(u_pos - u_output.size*0.5);
            // xy = r.zw + u_output.size*0.5;
            xy_out = hex2screen(xy_out - u_output.size*0.5);
            xy_out = xy_out + u_output.size*0.5;
        }
        vec2 diff = abs(xy_out-xy);
        // diff = min(diff, u_output.size-diff); // circular padding for the brush
        if (length(diff)*u_zoom>=u_r) 
          discard;
        setOutput(u_brush);

    }`,
    perception: `

    uniform float u_seed, u_updateProbability;
    uniform bool scale_zero;
    
    void main() {
        vec2 xy = getOutputXY();

        #ifndef SPARSE_UPDATE
          if (hash13(vec3(xy, u_seed)) > u_updateProbability) {
            setOutput(vec4(0.0, 0.0, 0.0, 0.0));
            return;
          }
        #endif
        
        #ifdef SPARSE_UPDATE
            if (scale_zero) {
            // xy = texture2D(u_shuffleTex, xy/u_output.size).xy*255.0+0.5 + u_shuffleOfs;
            xy = texture2D(u_shuffleTex, xy/u_output.size).xy+0.5 + u_shuffleOfs;
            // xy = texture2D(u_shuffleTex, xy/u_output.size).xy * (HW.x - 1.0) + 0.5 + u_shuffleOfs;
            xy = mod(xy, u_input.size);
            } else {
                // xy = xy;
                xy = floor(xy) + 0.5;
            }
        #endif
        float ch = getOutputChannel();
        if (ch >= u_output.depth4)
            return;
            
        

        float filterBand = floor((ch+0.5)/u_input.depth4);
        // inputCh: this is the channel idx in the original tensor
        float inputCh = ch-filterBand*u_input.depth4; 
        if (filterBand < 0.5) {
            setOutput(u_input_read(xy, inputCh));
        } else if (filterBand < 2.5) {
            highp vec4 dx = conv3x3(xy, inputCh, sobelX*(1.0-u_hexGrid) + sobelXhex*u_hexGrid);
            highp vec4 dy = conv3x3(xy, inputCh, sobelY*(1.0-u_hexGrid) + sobelYhex*u_hexGrid);
            highp vec2 dir = getCellDirection(xy);
            float s = dir.x, c = dir.y;
            highp vec4 res = filterBand < 1.5 ? dx*c-dy*s : dx*s+dy*c;
            #ifdef OURS
                res = res * 8.0; // We didn't normalize the kernels
            #endif
            setOutput(res);
        
        
        } else {
            highp vec4 res = conv3x3(xy, inputCh, gauss*(1.0-u_hexGrid) + gaussHex*u_hexGrid);
            #ifdef OURS
                res = res * 8.0;  // We didn't normalize the kernels
            #endif
            setOutput(res);
        }
    }`,
    greyscale: `
    void main() {
        vec2 xy = getOutputXY();
        
        vec4 v = texture2D(u_input_tex, xy/u_output.size);
        float grey = dot(v.rgb, vec3(0.33, 0.33, 0.33))* 2.0 - 1.0;

        // if (xy.x > 60.)
        //     grey = -0.1;
        // else
        //     grey = tanh(-0.1);


        gl_FragColor = vec4(grey, grey, grey, 1.0);
    }
    `,
    preprocess_image: `
    void main() {
        vec2 xy = getOutputXY();
        
        
        float ch = getOutputChannel();
        if (ch >= u_output.depth4)
            return;

        // inputCh: this is the channel idx in the original tensor
        // float inputCh = ch-filterBand*u_input.depth4; 
        float inputCh = 0.; 
        
        highp vec4 dx = conv3x3(xy, inputCh, sobelX*(1.0-u_hexGrid) + sobelXhex*u_hexGrid);
        highp vec4 dy = conv3x3(xy, inputCh, sobelY*(1.0-u_hexGrid) + sobelYhex*u_hexGrid);
        highp vec2 dir = getCellDirection(xy);
        float s = dir.x, c = dir.y;
        highp vec4 sob_x = dx*c-dy*s;
        highp vec4 sob_y = dx*s+dy*c;

        #ifdef OURS
            sob_x = sob_x * 8.0; // We didn't normalize the kernels
            sob_y = sob_y * 8.0; // We didn't normalize the kernels
        #endif



        highp vec4 lap = conv3x3(xy, inputCh, gauss*(1.0-u_hexGrid) + gaussHex*u_hexGrid);
        #ifdef OURS
            lap = lap * 8.0;  // We didn't normalize the kernels
        #endif

        vec4 res = vec4(sob_x.r, sob_y.r, lap.r, 1.0);
        if (xy.x < 60.0)
            res = vec4(tanh(sob_x.r), tanh(sob_y.r), tanh(lap.r), 1.0);


        setOutput(res);
        
    }
    `,
    dense: `
    ${defInput('u_control')}
    ${defInput('u_edgemap')}
    //u_weightTex contains the layer weights
    uniform sampler2D u_weightTex;
    uniform float u_seed, u_fuzz, u_updateProbability;
    uniform highp vec2 u_weightCoefs; // scale, center
    uniform highp vec2 u_layout;
    uniform highp vec2 grid_size;
    uniform bool bias, pos_emb, relu, edge_conditioning;
    
    const float MAX_PACKED_DEPTH = 50.0;
    
    vec4 readWeightUnscaled(vec2 p) {
        highp vec4 w = texture2D(u_weightTex, p);
        return w-u_weightCoefs.y; // centerize
    }
    
    void main() {
      vec2 xy = getOutputXY();
    
    
      #ifndef SPARSE_UPDATE
      if (hash13(vec3(xy, u_seed)) > u_updateProbability) {
        setOutput(vec4(0.0, 0.0, 0.0, 0.0));
        return;
      }
      #endif
      
      
      float ch = getOutputChannel();
      if (ch >= u_output.depth4)
          return;


      float d = u_input.depth + 1.0;
      if (pos_emb) {
        d = d + 2.0;
      }
      float dy = 1.0 / (d) / u_layout.y;
      // float dy = 1.0/(d)/u_layout.y;
      // float dy = 1.0/(u_input.depth+1.0)/u_layout.y;
      vec2 p = vec2((ch+0.5)/u_output.depth4, dy*0.5);
      vec2 fuzz = (hash23(vec3(xy, u_seed+ch))-0.5)*u_fuzz;
      // vec2 fuzz = vec2(0.0, 0.0);

      vec2 realXY = xy;
      #ifdef SPARSE_UPDATE
        // realXY = texture2D(u_shuffleTex, xy/u_output.size).xy * 255.0 +0.5 + u_shuffleOfs;
        realXY = texture2D(u_shuffleTex, xy/u_output.size).xy + 0.5 + u_shuffleOfs;
        realXY = mod(realXY, HW);
        // realXY = texture2D(u_shuffleTex, xy/u_output.size).xy * (HW.x - 1.0) +0.5 + u_shuffleOfs;
      #endif
      
      
      
    //   float modelIdx = u_control_read(realXY+fuzz, 0.0).x+0.5;
      float modelIdx = 0.5;

      p.x += floor(mod(modelIdx, u_layout.x));
      p.y += floor(modelIdx/u_layout.x);
      p /= u_layout;
      highp vec4 result = vec4(0.0);
      for (float i=0.0; i < MAX_PACKED_DEPTH; i+=1.0) {
          highp vec4 inVec = u_input_read(xy, i);
          result += inVec.x * readWeightUnscaled(p); p.y += dy;
          result += inVec.y * readWeightUnscaled(p); p.y += dy;
          result += inVec.z * readWeightUnscaled(p); p.y += dy;
          result += inVec.w * readWeightUnscaled(p); p.y += dy;
          if (i+1.5>u_input.depth4) {
              break;
          }
      }
      if (pos_emb) {
        
        highp vec2 pos = floor(realXY);
        highp vec2 delta = vec2(0.5, 0.5) / HW;
        highp vec2 pemb = pos / HW;
        pemb = 2.0 * (pemb - 0.5 + delta);
        pemb = rotate(-u_angle) * pemb;
        result += pemb.y * readWeightUnscaled(p); p.y += dy;
        result += pemb.x * readWeightUnscaled(p); p.y += dy;
      
      };
      if (edge_conditioning) {
        
        highp vec2 pos = floor(realXY);
        highp vec2 delta = vec2(0.5, 0.5) / HW;
        highp vec3 edges = u_edgemap_read(pos, 0.0).rgb;
        // edges = vec3(0.);
        result += edges.x * readWeightUnscaled(p); p.y += dy;
        result += edges.y * readWeightUnscaled(p); p.y += dy;
        result += edges.z * readWeightUnscaled(p); p.y += dy;

      };
      if (bias) {
        result += readWeightUnscaled(p);  // bias
        // p.y += dy; 
      };
      
      result = result*u_weightCoefs.x;
      if (relu) {
        result = max(result, 0.0);
      
      }
      setOutput(result);
    }`,
    update: `
    ${defInput('u_update')}
    uniform float u_seed, u_updateProbability;
    uniform float u_rate;
    varying vec2 uv;

    void main() {
      vec2 xy = getOutputXY();
    //   if (xy.y>100.0 && xy.y < 150.0) {
    //       xy.x -= 1.0;
    //   }
      float ch = getOutputChannel();
      highp vec4 state = u_input_read(xy, ch); //u_input_readUV(uv);
      highp vec4 update = vec4(0.0);
      #ifdef SPARSE_UPDATE
        vec4 shuffleInfo = texture2D(u_unshuffleTex, fract((xy-u_shuffleOfs)/u_output.size));
        if (shuffleInfo.z > 0.5) {
            // update = u_update_read(shuffleInfo.xy*255.0+0.5, getOutputChannel());
            update = u_update_read(shuffleInfo.xy + 0.5, getOutputChannel());
            // update = u_update_read(shuffleInfo.xy*(HW.x - 1.0)+0.5, getOutputChannel());
        }
      #else
        if (hash13(vec3(xy, u_seed)) <= u_updateProbability) {
            update = u_update_readUV(uv);    
        }
      #endif
      setOutput(state + update * u_rate);
    }`,
    vis: `
    uniform float u_raw;
    uniform float u_zoom;
    uniform float u_perceptionCircle, u_arrows;
    varying vec2 uv;

    float clip01(float x) {
        return min(max(x, 0.0), 1.0);
    }

    const float PI = 3.141592653;

    float peak(float x, float r) {
        float y = x/r;
        return exp(-y*y);
    }

    float getElement(vec4 v, float i) {
        if (i<1.0) return v.x;
        if (i<2.0) return v.y;
        if (i<3.0) return v.z;
        return v.w;
    }

    vec3 onehot3(float i) {
        if (i<1.0) return vec3(1.0, 0.0, 0.0);
        if (i<2.0) return vec3(0.0, 1.0, 0.0);
        return vec3(0.0, 0.0, 1.0);
    }

    float sdTriangleIsosceles( in vec2 p, in vec2 q ) {
        p.x = abs(p.x);
        vec2 a = p - q*clamp( dot(p,q)/dot(q,q), 0.0, 1.0 );
        vec2 b = p - q*vec2( clamp( p.x/q.x, 0.0, 1.0 ), 1.0 );
        float s = -sign( q.y );
        vec2 d = min( vec2( dot(a,a), s*(p.x*q.y-p.y*q.x) ),
                      vec2( dot(b,b), s*(p.y-q.y)  ));
        return -sqrt(d.x)*sign(d.y);
    }
    

    void main() {
        vec2 xy = vec2(uv.x, 1.0-uv.y);
        if (u_raw > 0.5) {
            // Show groups of 4 channels
            if (u_input.depth == 12.0) {
                vec3 rgb = vec3(0.0, 0.0, 0.0);
                if (xy.x <= 0.5 || xy.y <= 0.5) {
                    rgb = texture2D(u_input_tex, xy).rgb;
                    

                } else {
                    
                    float r = texture2D(u_input_tex, xy - vec2(0.5, 0.5)).a;
                    float g = texture2D(u_input_tex, xy - vec2(0.5, 0.0)).a;
                    float b = texture2D(u_input_tex, xy - vec2(0.0, 0.5)).a;
                    rgb = vec3(r, g, b);

                }
                
                

                #ifdef OURS                                    
                    rgb = clamp(rgb + 0.5, 0.0, 1.0);
                #else
                    rgb = rgb / 2.0 + 0.5;
                #endif
                
                gl_FragColor = vec4(rgb, 1.0);
 
            } else {
                gl_FragColor = texture2D(u_input_tex, xy);
                #ifdef OURS                    
                    gl_FragColor = clamp(gl_FragColor + 0.5, 0.0, 1.0);
                #else
                    gl_FragColor = gl_FragColor / 2.0 + 0.5;
                #endif
                gl_FragColor.a = 1.0;
            } 
        } else {
            xy = (xy + vec2(0.5)*(u_zoom-1.0))/u_zoom;
            xy *= u_input.size;
            vec2 fp = 2.0 * fract(xy)-1.0;

            if (u_hexGrid > 0.0) {
                vec4 r = getHex(xy-u_input.size*0.5);
                xy = r.zw+u_input.size*0.5;
                fp = r.xy;
            }

            #ifdef OURS
                vec3 cellRGB = clamp(u_input_read(xy, 0.0).rgb + 0.5, 0.0, 1.0);
            #else
                vec3 cellRGB = u_input_read(xy, 0.0).rgb/2.0+0.5;
            #endif
            
            vec3 rgb = cellRGB;
            if (3.0 < u_zoom) {
                vec2 dir = getCellDirection(floor(xy)+0.5);
                float s = dir.x, c = dir.y;
                fp = mat2(c, s, -s, c) * fp;    
                float r = length(fp);
                float fade = clip01((u_zoom-3.0)/3.0);
                float m = 1.0;//1.0-min(r*r*r, 1.0)*fade;
                rgb *= m;
                if (12.0 < u_zoom) {
                    float ang = atan(-fp.x, fp.y)/(2.0*PI)+0.5;
                    float ch = mod(ang*u_input.depth+1.5, u_input.depth);
                    float barLengh = 0.0;
                    vec3 barColor = vec3(0.5);
                    if (ch < 3.0) {
                        vec3 i3 = onehot3(ch);
                        barColor = i3;
                        barLengh = dot(cellRGB, i3);
                    } else {
                        vec4 v4 = u_input_read01(xy, floor(ch/4.0));
                        barLengh = getElement(v4, mod(ch, 4.0));
                    }

                    float c = mod(ch, 1.0);
                    c = peak(c-0.5, 0.2);
                    if (r>barLengh)
                      c = 0.0;
                    float fade = clip01((u_zoom-12.0)/8.0);
                    c *= fade;
                    rgb += barColor*c;

                    float arrow = sdTriangleIsosceles((fp+vec2(0.0, 0.95))*vec2(4.0, 4.0), vec2(1.0, 1.0));
                    arrow = clip01(1.0-abs(arrow)*u_zoom/4.0);
                    rgb += arrow*fade*u_arrows;

                    float cr = length(u_input.size/2.0-0.5-xy);
                    rgb += peak(cr-1.5, 0.5/u_zoom)*fade*u_perceptionCircle;
                }
            } 

            gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
        }
    }`
}

function createPrograms(gl, defines) {
    defines = defines || '';
    const res = {};
    for (const name in PROGRAMS) {
        const fs_code = defines + PREFIX + PROGRAMS[name];
        // vs : vertex shader
        // fs: fragment shader
        const progInfo = twgl.createProgramInfo(gl, [vs_code, fs_code]);
        progInfo.name = name;
        res[name] = progInfo;
    }
    // res is a dictionary of shader programs
    return res;
}

function createTensor(gl, w, h, depth, packScaleZero, is_float = false) {
    // Pack the depth dimension into the spatial dimension
    const depth4 = Math.ceil(depth / 4);
    const gridW = Math.ceil(Math.sqrt(depth4));
    const gridH = Math.floor((depth4 + gridW - 1) / gridW);
    const texW = w * gridW, texH = h * gridH;

    // const ext = gl.getExtension('EXT_color_buffer_float');


    const attachments = [{
        minMag: gl.NEAREST,
        format: gl.RGBA,
        internalFormat: gl.RGBA32F
    }];
    if (is_float) {
        attachments[0].type = gl.FLOAT;
    }

    const fbi = twgl.createFramebufferInfo(gl, attachments, texW, texH);
    const tex = fbi.attachments[0];
    return {
        _type: 'tensor',
        fbi, w, h, depth, gridW, gridH, depth4, tex, packScaleZero,
    };
}

function setTensorUniforms(uniforms, name, tensor) {
    uniforms[name + '.size'] = [tensor.w, tensor.h];
    uniforms[name + '.gridSize'] = [tensor.gridW, tensor.gridH];
    uniforms[name + '.depth'] = tensor.depth;
    uniforms[name + '.depth4'] = tensor.depth4;
    uniforms[name + '.packScaleZero'] = tensor.packScaleZero;
    if (name != 'u_output') {
        uniforms[name + '_tex'] = tensor.tex;
    }
}

function createDenseInfo(gl, params) {
    // params is basically one of layers from the json file

    const center = "center" in params ? params.center : 127.0 / 255.0;

    const [in_n, out_n] = params.shape;
    const info = {
        layout: params.layout, out_n,
        // quantScaleZero: params.quant_scale_zero,
        ready: false
    };

    info.pos_emb = params.pos_emb ? "pos_emb" in params : false;
    info.bias = params.bias ? "bias" in params : true;
    info.edge_conditioning = params.edge_conditioning ? "edge_conditioning" in params : false;
    var ch_in = in_n;
    ch_in = info.pos_emb ? ch_in - 2 : ch_in;
    ch_in = info.bias ? ch_in - 1 : ch_in;
    info.in_n = ch_in;
    info.coefs = [params.scale, center];

    if ("data_flatten" in params) {
        let width = params.data_shape[1];
        let height = params.data_shape[0];
        info.tex = twgl.createTexture(gl, {
            minMag: gl.NEAREST, src: params.data_flatten, flipY: false, premultiplyAlpha: false,
            width: width, height: height,
            internalFormat: gl.RGBA32F,
            format: gl.RGBA,
            type: gl.FLOAT,
        })
        info.ready = true;

    } else {
        info.tex = twgl.createTexture(gl, {
            minMag: gl.NEAREST, src: params.data, flipY: false, premultiplyAlpha: false,
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
        }, () => {
            info.ready = true;
        });
    }
    return info;
}

export class DyNCA {
    constructor(gl, models, gridSize, gui, our_version = true, use_webcam = true) {
        // models is basically the json file
        this.use_webcam = use_webcam;
        self = this;
        this.gl = gl;

        this.n_perception_scales = "n_perception_scales" in models ? models.n_perception_scales : 1;
        // alert(this.n_perception_scales)

        this.gridSize = gridSize || [96, 96];

        this.updateProbability = 0.5;
        this.shuffledMode = true; // changed
        // alert(this.shuffledMode)

        this.rotationAngle = 0.0;
        this.rate = 1.0;
        this.alignment = 0;
        this.fuzz = 8.0;
        this.perceptionCircle = 0.0;
        this.arrowsCoef = 0.0;
        this.visMode = 'color';
        this.hexGrid = false;

        this.our_version = our_version;

        this.layers = [];
        this.setWeights(models);

        const defs = (this.our_version ? '#define OURS\n' : '') + (this.shuffledMode ? '#define SPARSE_UPDATE\n' : '');

        this.progs = createPrograms(gl, defs);

        // representing vertices of a square with two triangles
        this.quad = twgl.createBufferInfoFromArrays(gl, {
            position: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
        });

        this.setupBuffers();
        const visNames = Object.getOwnPropertyNames(this.buf);
        visNames.push('color');

        if (gui) {
            gui.add(this, 'rotationAngle').min(0.0).max(360.0);
            gui.add(this, 'alignment', { cartesian: 0, polar: 1, bipolar: 2 }).listen();
            //gui.add(this, 'fuzz').min(0.0).max(128.0);
            //gui.add(this, 'perceptionCircle').min(0.0).max(1.0);
            //gui.add(this, 'visMode', visNames);
            gui.add(this, 'hexGrid');

            gui.add(this, 'benchmark');

            // this.benchmark = ()=>{
            //   document.getElementById('log').insertAdjacentHTML('afterbegin', this.benchmark());
            // }


        }

        this.streamReady = false;

        if (this.use_webcam) {
            this.setupWebcam();
        }


        this.clearCircle(0, 0, 10000);
    }

    setupWebcam() {

        navigator.getUserMedia = (navigator.getUserMedia ||
            navigator.webkitGetUserMedia ||
            navigator.mozGetUserMedia ||
            navigator.msGetUserMedia);


        this.videoElement = document.getElementById('webcamVideo');

        navigator.mediaDevices.getUserMedia({
            video: {
                frameRate: { ideal: 20, max: 30 },
                width: this.gridSize[0], height: this.gridSize[1]
            }
        }).then((stream) => {
            this.videoElement.srcObject = stream;
            this.streamReady = true;
        });

    }

    setupBuffers() {
        const gl = this.gl;
        const [gridW, gridH] = this.gridSize;
        const shuffleH = Math.ceil(gridH * this.updateProbability);
        const shuffleCellN = shuffleH * gridW;
        const totalCellN = gridW * gridH;
        // const shuffleBuf = new Uint8Array(shuffleCellN * 4); // Indices of the cells to be updated
        // const unshuffleBuf = new Uint8Array(totalCellN * 4);

        const shuffleBuf = new Float32Array(shuffleCellN * 4); // Indices of the cells to be updated
        const unshuffleBuf = new Float32Array(totalCellN * 4);

        let k = 0;


        for (let i = 0; i < totalCellN; ++i) {
            // This exactly updates shuffleCellN of cells.
            if (Math.random() < (shuffleCellN - k) / (totalCellN - i)) {
                shuffleBuf[k * 4 + 0] = i % gridW;
                shuffleBuf[k * 4 + 1] = Math.floor(i / gridW);
                unshuffleBuf[i * 4 + 0] = k % gridW;
                unshuffleBuf[i * 4 + 1] = Math.floor(k / gridW);
                unshuffleBuf[i * 4 + 2] = 255;
                k += 1;
            }
        }
        this.shuffleTex = twgl.createTexture(gl, {
            minMag: gl.NEAREST, width: gridW, height: shuffleH, src: shuffleBuf,
            flipY: false, premultiplyAlpha: false,
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
        });
        this.unshuffleTex = twgl.createTexture(gl, {
            minMag: gl.NEAREST,
            width: gridW,
            height: gridH,
            src: unshuffleBuf,
            flipY: false, premultiplyAlpha: false,
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
        });
        this.shuffleOfs = [0, 0];

        const updateH = this.shuffledMode ? shuffleH : gridH;
        const perception_n = this.layers[0].in_n;
        const lastLayer = this.layers[this.layers.length - 1];
        const channel_n = lastLayer.out_n;
        const stateQuantization = lastLayer.quantScaleZero;
        const no_quantization = [1.0, 0.0];
        this.buf = {
            control: createTensor(gl, gridW, gridH, 4, [255.0, 0.0], false),
            state: createTensor(gl, gridW, gridH, channel_n, no_quantization, true),
            newState: createTensor(gl, gridW, gridH, channel_n, no_quantization, true),
            perception0: createTensor(gl, gridW, updateH, perception_n, no_quantization, true),
            frame: createTensor(gl, gridW, gridH, 4, no_quantization, true),
            greyscale: createTensor(gl, gridW, gridH, 4, no_quantization, true),
            edgemap: createTensor(gl, gridW, gridH, 4, no_quantization, true),

        };

        // For now we only support multi-scale perception with 2 scales
        if (this.n_perception_scales > 1) {
            this.buf.state_down = createTensor(gl, Math.floor(gridW / 2 + 0.5), Math.floor(gridH / 2 + 0.5), channel_n, no_quantization, true);
            this.buf.state_down_up = createTensor(gl, gridW, gridH, channel_n, no_quantization, true);
            this.buf.perception1 = createTensor(gl, Math.floor(gridW / 2 + 0.5), Math.floor(gridH / 2 + 0.5), perception_n, no_quantization, true);
            this.buf.perception1_up = createTensor(gl, gridW, gridH, perception_n, no_quantization, true);
            this.buf.perception = createTensor(gl, gridW, updateH, perception_n, no_quantization, true);

        }

        for (let i = 0; i < this.layers.length; ++i) {
            const layer = this.layers[i];
            // this.buf[`layer${i}`] = createTensor(gl, gridW, updateH, layer.out_n, layer.quantScaleZero, false);
            this.buf[`layer${i}`] = createTensor(gl, gridW, updateH, layer.out_n, no_quantization, true);
        }
    }
    // TODO: Implement this function
    updateImage() {
        if (this.use_webcam && this.streamReady) {
            const gl = this.gl;

            gl.bindTexture(gl.TEXTURE_2D, this.buf.frame.tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.videoElement);
        }

    }

    step(stage) {

        this.updateImage();


        stage = stage || 'all';
        if (!this.layers.every(l => l.ready))
            return;

        const seed = Math.random() * 1000;

        if (stage == 'all') {
            const [gridW, gridH] = this.gridSize;
            // Random point on the grid
            this.shuffleOfs = [Math.floor(Math.random() * gridW), Math.floor(Math.random() * gridH)];
        }

        if (stage == 'all' || stage == 'Perception' || stage == 'Multi-Scale Perception') {
            this.runLayer(self.progs.perception, this.buf.perception0, {
                u_input: this.buf.state, u_angle: this.rotationAngle / 180.0 * Math.PI,
                u_alignment: this.alignment, u_hexGrid: this.hexGrid,
                u_seed: seed, u_updateProbability: this.updateProbability, scale_zero: true,
            });
        }

        if (this.n_perception_scales > 1) {
            this.runLayer(self.progs.bilinear_downsample, this.buf.state_down, {
                u_input: this.buf.state,
            });
            this.runLayer(self.progs.perception, this.buf.perception1, {
                u_input: this.buf.state_down, u_angle: this.rotationAngle / 180.0 * Math.PI,
                u_alignment: this.alignment, u_hexGrid: this.hexGrid,
                u_seed: seed, u_updateProbability: this.updateProbability, scale_zero: false,
            });

            this.runLayer(self.progs.bilinear_upsample_add, this.buf.perception, {
                u_input: this.buf.perception1, u_perception0: this.buf.perception0, scale_zero: true,
            });
            [this.buf.perception0, this.buf.perception] = [this.buf.perception, this.buf.perception0];
        }

        if (stage == 'all' || stage == 'preprocess_image') {
            this.runLayer(self.progs.greyscale, this.buf.greyscale, {
                u_input: this.buf.frame, u_angle: this.rotationAngle / 180.0 * Math.PI,
                u_alignment: this.alignment, u_hexGrid: this.hexGrid,
                u_seed: seed, u_updateProbability: this.updateProbability, scale_zero: true,
            });

            this.runLayer(self.progs.preprocess_image, this.buf.edgemap, {
                u_input: this.buf.greyscale, u_angle: this.rotationAngle / 180.0 * Math.PI,
                u_alignment: this.alignment, u_hexGrid: this.hexGrid,
                u_seed: seed, u_updateProbability: this.updateProbability, scale_zero: true,
            });
        }

        let inputBuf = this.buf.perception0;

        for (let i = 0; i < this.layers.length; ++i) {
            if (stage == 'all' || stage == `FC Layer${i + 1}`)
                var relu = i == 0 ? true : false;
            var rate = i == 0 ? 1.0 : this.rate;
            this.runDense(this.buf[`layer${i}`], inputBuf, this.layers[i], relu, seed, rate);
            inputBuf = this.buf[`layer${i}`];
        }
        if (stage == 'all' || stage == 'Stochastic Update') {
            this.runLayer(this.progs.update, this.buf.newState, {
                u_input: this.buf.state, u_update: inputBuf,
                u_unshuffleTex: this.unshuffleTex, u_rate: this.rate,
                u_seed: seed, u_updateProbability: this.updateProbability
            });
        }

        if (stage == 'all') {
            [this.buf.state, this.buf.newState] = [this.buf.newState, this.buf.state];
        }
    }

    benchmark() {
        const gl = this.gl;
        // const flushBuf = new Uint8Array(4);
        const flushBuf = new Float32Array(4);
        const flush = buf => {
            buf = buf || this.buf.state;
            // gl.flush/finish don't seem to do anything, so reading a single
            // pixel from the state buffer to flush the GPU command pipeline
            twgl.bindFramebufferInfo(gl, buf.fbi);
            gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, flushBuf);
        }

        // const flushBuf = new Uint8Array(4);
        // const flush = buf=>{
        //     buf = buf || this.buf.state;
        //     // gl.flush/finish don't seem to do anything, so reading a single
        //     // pixel from the state buffer to flush the GPU command pipeline
        //     twgl.bindFramebufferInfo(gl, buf.fbi);
        //     gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, flushBuf);
        // }

        flush();
        const stepN = 500;
        const start = Date.now();
        for (let i = 0; i < stepN; ++i)
            this.step();
        flush();
        const total = (Date.now() - start) / stepN;

        const ops = [];
        if (this.n_perception_scales > 1) {
            ops.push('Multi-Scale Perception');
        } else {
            ops.push('Perception')
        }


        for (let i = 0; i < this.layers.length; ++i)
            ops.push(`FC Layer${i + 1}`);
        ops.push('Stochastic Update');
        let perOpTotal = 0.0;
        const perOp = [];
        for (const op of ops) {
            const start = Date.now();
            for (let i = 0; i < stepN; ++i) {
                this.step(op);
            }
            flush(this.buf[op]);
            const dt = (Date.now() - start) / stepN;
            perOpTotal += dt
            perOp.push([op, dt]);
        }
        const perOpStr = perOp.map((p) => {
            const [programName, dt] = p;
            const percent = 100.0 * dt / perOpTotal;
            return `${programName}: ${percent.toFixed(1)}%`;
        }).join('\n');
        const T = this.n_perception_scales > 1 ? 64.0 : 24.0;
        var result = `FPS: ${(1000.0 / (total * T)).toFixed(2)}, ${(total).toFixed(2)} ms/step, ${(1000.0 / total).toFixed(2)} steps/sec\n` + perOpStr + '\n\n'
        // document.getElementById('log').innerHTML = `${(total).toFixed(2)} ms/step, ${(1000.0 / total).toFixed(2)} step/sec\n` + perOpStr + '\n\n'
        alert(result);
    }

    paint(x, y, r, brush) {
        // the model idx is passed as the brush
        this.runLayer(this.progs.paint, this.buf.control, {
            u_pos: [x, y], u_r: r, u_brush: [brush, 0, 0, 0], u_hexGrid: this.hexGrid, u_zoom: 1.0
        });
    }

    clearCircle(x, y, r, brush, zoom = 1.0) {
        self.runLayer(self.progs.paint, this.buf.state, {
            u_pos: [x, y], u_r: r, u_brush: [0.0, 0.0, 0.0, 0.0], u_hexGrid: this.hexGrid, u_zoom: zoom
        });
    }

    setWeights(models) {
        const gl = this.gl;
        this.layers.forEach(layer => gl.deleteTexture(layer));
        this.layers = models.layers.map(layer => createDenseInfo(gl, layer));
    }

    runLayer(program, output, inputs) {
        const gl = this.gl;
        inputs = inputs || {};
        const uniforms = {};
        for (const name in inputs) {
            const val = inputs[name];
            if (val._type == 'tensor') {
                setTensorUniforms(uniforms, name, val);
            } else {
                uniforms[name] = val;
            }
        }
        uniforms['u_shuffleTex'] = this.shuffleTex;
        uniforms['u_shuffleOfs'] = this.shuffleOfs;
        uniforms['HW'] = this.gridSize;
        setTensorUniforms(uniforms, 'u_output', output);

        twgl.bindFramebufferInfo(gl, output.fbi);
        gl.useProgram(program.program);
        twgl.setBuffersAndAttributes(gl, program, this.quad);
        twgl.setUniforms(program, uniforms);
        twgl.drawBufferInfo(gl, this.quad);
        return { programName: program.name, output }
    }

    runDense(output, input, layer, relu = false, seed = 0) {
        return this.runLayer(this.progs.dense, output, {
            u_input: input, u_control: this.buf.control,
            u_weightTex: layer.tex, u_weightCoefs: layer.coefs, u_layout: layer.layout,
            u_seed: seed, u_fuzz: this.fuzz, u_updateProbability: this.updateProbability,
            bias: layer.bias, pos_emb: layer.pos_emb, relu: relu, u_edgemap: this.buf.edgemap,
            edge_conditioning: layer.edge_conditioning,
            grid_size: this.gridSize, u_angle: this.rotationAngle / 180.0 * Math.PI,

        });
    }

    draw(zoom) {
        const gl = this.gl;
        zoom = zoom || 1.0;

        gl.useProgram(this.progs.vis.program);
        twgl.setBuffersAndAttributes(gl, this.progs.vis, this.quad);
        const uniforms = {
            u_raw: 0.0, u_zoom: zoom,
            u_angle: this.rotationAngle / 180.0 * Math.PI,
            u_alignment: this.alignment,
            u_perceptionCircle: this.perceptionCircle,
            u_arrows: this.arrowsCoef,
            u_hexGrid: this.hexGrid,
        };
        let inputBuf = this.buf.state;
        if (this.visMode != 'color') {
            inputBuf = this.buf[this.visMode];
            uniforms.u_raw = 0.0;
        }
        inputBuf = this.buf.state;
        // inputBuf = this.buf.edgemap;
        // uniforms.u_raw = 1.5;
        setTensorUniforms(uniforms, 'u_input', inputBuf);
        twgl.setUniforms(this.progs.vis, uniforms);
        twgl.drawBufferInfo(gl, this.quad);
    }
}

export class ImagePreprocessor {
    constructor(gl, gridSize) {
        // models is basically the json file

        self = this;
        this.gl = gl;


        this.gridSize = gridSize || [96, 96];

        this.rotationAngle = 0.0;
        this.rate = 1.0;
        this.alignment = 0;
        this.fuzz = 8.0;
        this.perceptionCircle = 0.0;
        this.arrowsCoef = 0.0;
        this.visMode = 'color';
        this.hexGrid = false;

        this.program = twgl.createProgramInfo(this.gl, [vs_code, fs_code]);

        const defs = (this.our_version ? '#define OURS\n' : '') + (this.shuffledMode ? '#define SPARSE_UPDATE\n' : '');

        // this.progs = createPrograms(gl, defs);

        // representing vertices of a square with two triangles
        this.quad = twgl.createBufferInfoFromArrays(gl, {
            position: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
        });
        this.streamReady = false;

        navigator.getUserMedia = (navigator.getUserMedia ||
            navigator.webkitGetUserMedia ||
            navigator.mozGetUserMedia ||
            navigator.msGetUserMedia);


        this.videoElement = document.getElementById('webcamVideo');

        navigator.mediaDevices.getUserMedia({
            video: {
                frameRate: { ideal: 20, max: 30 },
                width: this.gridSize[0], height: this.gridSize[1]
            }
        }).then((stream) => {
            this.videoElement.srcObject = stream;
            this.streamReady = true;
        });

        this.image_texture = twgl.createTexture(gl, {
            src: [0, 0, 255],
            format: gl.RGB,
            min: gl.LINEAR,
            wrap: gl.CLAMP_TO_EDGE,
        });

    }


    render() {
        const gl = this.gl;
        if (this.streamReady) {
            gl.bindTexture(gl.TEXTURE_2D, this.image_texture)
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, this.videoElement);
        }


        twgl.resizeCanvasToDisplaySize(gl.canvas);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.program.program);

        twgl.setBuffersAndAttributes(gl, this.program, this.quad);
        twgl.setUniforms(this.program, {
            u_tex: this.image_texture,
        });
        twgl.drawBufferInfo(gl, this.quad);
    }


}
