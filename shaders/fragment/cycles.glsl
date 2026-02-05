#version 330 core
out vec4 FragColor;

uniform float time;
uniform vec2 resolution;
uniform vec2 viewportOffset;
uniform float camYaw;
uniform float camPitch;
uniform float radius = 5.0;
uniform vec3 CamOrbit = vec3(0.0);
uniform int frameIndex; // Essential for noise decorrelation
uniform sampler2D accumulationTexture;
uniform int useAccumulation; // 0 = no accumulation, 1 = with accumulation
uniform vec3 SkyColorTop;
uniform vec3 SkyColorBottom;
uniform vec3 LightColor = vec3(0.7);
uniform vec3 LightDir = vec3(0.5, 0.5, -1.0);

uniform vec3 MovePos;
uniform vec3 MoveRot;

uniform int MaxFrames = 0;

{SDF_LIBRARY}

// --- RANDOM GENERATOR (Hash without Sine) ---
float hash12(vec2 p) {
    vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

vec2 stratifiedSample(vec2 seed, int sampleIndex, int totalSamples) {
    int gridSize = int(sqrt(float(totalSamples)));
    vec2 gridPos = vec2(mod(sampleIndex, gridSize), int(sampleIndex / gridSize)) / float(gridSize);
    vec2 offset = vec2(hash12(seed), hash12(seed + 0.1234)) / float(gridSize);
    return gridPos + offset;
}

vec3 getCosHemisphereSample(vec3 normal, vec2 seed, int sampleIndex, int totalSamples) {
    vec2 sample = stratifiedSample(seed, sampleIndex, totalSamples);
    float u = sample.x;
    float v = sample.y;
    float r = sqrt(u);
    float phi = 6.28318530718 * v;
    vec3 dir = vec3(r * cos(phi), r * sin(phi), sqrt(1.0 - u));

    vec3 up = abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    return tangent * dir.x + bitangent * dir.y + normal * dir.z;
}


vec3 mixColorSmooth(vec3 colA, vec3 colB, float dA, float dB, float k) {
    k *= 4.0;
    float h = max(k - abs(dA - dB), 0.0) / k;
    float t = clamp(0.5 + 0.5 * (dB - dA) / k, 0.0, 1.0);
    vec3 blended = mix(colA, colB, t);
    vec3 closer = (dA < dB) ? colA : colB;
    return mix(closer, blended, h);
}

vec4 getSceneDist(vec3 p) {
    {SCENE_CODE}
}

vec4 map(vec3 p) {
    return getSceneDist(p);
}

float FOV_ANGLE = {FOV_ANGLE_VAL}; 

vec3 calcNormal(vec3 p) {
    float h = 0.001;
    vec2 k = vec2(1, -1);
    return normalize(k.xyy * map(p + k.xyy * h).w + 
                     k.yxy * map(p + k.yxy * h).w +
                     k.yyx * map(p + k.yyx * h).w +
                     k.xxx * map(p + k.xxx * h).w);
}

// --- PATH TRACING CORE ---
float traceShadowRay(vec3 origin, vec3 dir, float maxDist) {
    float dO = 0.015; // Start slightly away from the surface to avoid self-intersection
    
    for(int i = 0; i < 64; i++) {
        vec3 p = origin + dir * dO;
        vec4 res = map(p);
        
        // If we hit any geometry (res.w > 0.001) before maxDist, it's occluded.
        if(res.w < 0.001 && dO < maxDist) {
            return 0.0; // Shadowed
        }
        
        dO += res.w;
        if(dO > maxDist) break;
    }
    
    return 1.0; // Not shadowed (visible to the light source)
}


vec3 tracePath(vec3 ro, vec3 rd, vec2 seed) {
    vec3 throughput = vec3(1.0);
    vec3 totalLight = vec3(0.0);
    
    for(int bounce = 0; bounce < 3; bounce++) {
        float dO = 0.0;
        vec4 res;
        bool hit = false;
        
        // --- Scene Intersection ---
        for(int i = 0; i < 80; i++) {
            vec3 p = ro + rd * dO;
            res = map(p);
            if(res.w < 0.001) { hit = true; break; }
            dO += res.w;
            if(dO > 50.0) break;
        }

        if(hit) {
            vec3 p = ro + rd * dO;
            vec3 n = calcNormal(p);
            vec3 albedo = res.xyz;
            
            // 1. DIRECT SUN ILLUMINATION CALCULATION
            float maxDist = 100.0; // Assuming the sun is far away
            float visibility = traceShadowRay(p, normalize(LightDir), maxDist);
            
            if (visibility > 0.001) {
                // Calculate diffuse contribution (Lambertian scattering)
                float NdotL = max(0.0, dot(n, normalize(LightDir)));
                
                vec3 sunLight = LightColor * NdotL * visibility;
                
                // Add direct light contribution, scaled by throughput (what reached this point)
                totalLight += throughput * albedo * sunLight;
            }
            
            // 2. INDIRECT LIGHT SAMPLING (Bounce)
            ro = p + n * 0.001;
            // Note: The seed for hemisphere sampling should ideally use a different mechanism 
            // or careful modification to ensure correct uncorrelated sampling across bounces.
            rd = getCosHemisphereSample(n, seed + float(bounce) + float(frameIndex), frameIndex, MaxFrames);
            
            throughput *= albedo;
        } else {
            // --- Sky/Atmosphere (Ambient Light) ---
            float t = 0.5 * (rd.y + 1.0);
            t = smoothstep(0.35, 0.5, t);
            vec3 skyColor = mix(SkyColorBottom, SkyColorTop, t);
            
            // Add sky contribution (which is multiplied by the current throughput)
            totalLight += throughput * skyColor;
            break;
        }
        
        // Russian Roulette (Path Termination)
        float p = max(throughput.r, max(throughput.g, throughput.b));
        if (hash12(seed + float(bounce)) > p) break;
        throughput /= p;
    }
    return totalLight;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Convert gl_FragCoord (window coords) into viewport-local coords
    vec2 localFrag = fragCoord - viewportOffset;
    // jitter / seed must use the local viewport coordinate (not the full-window coords)
    vec2 jitter = hash33(vec3(localFrag, frameIndex)).xy - 0.5;
    vec2 uv = (localFrag + jitter - 0.5 * resolution.xy) / resolution.y;

    vec3 ta = CamOrbit; 
    float pitch = clamp(camPitch, -1.55, 1.55); 
    float x = radius * cos(pitch) * cos(camYaw);
    float y = radius * sin(pitch);
    float z = radius * cos(pitch) * sin(camYaw);
    vec3 ro = vec3(x, y, z) + ta; 

    vec3 ww = normalize(ta - ro);     
    vec3 uu = normalize(cross(vec3(0,1,0), ww)); 
    vec3 vv = cross(ww, uu);          
    float fovFactor = 1.0 / tan(FOV_ANGLE * 0.5);
    vec3 rd = normalize(uu * uv.x * fovFactor + vv * uv.y * fovFactor + ww);

    vec2 seed = localFrag + float(frameIndex) * 13.41;
    vec3 col = tracePath(ro, rd, seed);

    // Temporal accumulation blending
    if (useAccumulation == 1) {
        vec2 texCoord = fragCoord / resolution;
        vec4 accumulated = texture(accumulationTexture, texCoord);
        
        // Blend:  weight new frame less heavily as we accumulate more samples
        float sampleCount = accumulated.w;
        float blendFactor = 1.0 / (sampleCount + 1.0);
        
        // IMPORTANT: accumulation texture stores LINEAR radiance (no tonemap)
        col = mix(accumulated.rgb, col, blendFactor);
        fragColor = vec4(col, sampleCount + 1.0);
    } else {
        // No accumulation -> write single-sample linear radiance; alpha=1.0
        fragColor = vec4(col, 1.0);
    }
    
    // NOTE: Do NOT apply tonemapping when we are writing into the accumulation buffer.
    // Apply tonemapping on display instead to avoid double / repeated tonemapping.
    if (useAccumulation == 0) {
        // Tonemapping & Gamma only when NOT accumulating (direct output)
        vec3 toneMapped = pow(fragColor.rgb, vec3(0.4545));
        fragColor = vec4(toneMapped, fragColor.w);
    }
}

void main() {
    mainImage(FragColor, gl_FragCoord.xy);
}