#version 330 core
out vec4 FragColor;

uniform float time;
uniform vec2 resolution;
uniform float camYaw;     // Horizontal rotation (around Y axis)
uniform float camPitch;   // Vertical rotation (around X axis)
uniform float radius = 5.0; // Camera orbit distance
uniform vec3 CamOrbit = vec3(0.0); // Center of orbit

// Assuming these are replaced by your external code
{SDF_LIBRARY}

// Color mixing function for smooth operations
vec3 mixColorSmooth(vec3 colA, vec3 colB, float dA, float dB, float k) {
    k *= 4.0;
    float h = max(k - abs(dA - dB), 0.0) / k;
    float t = clamp(0.5 + 0.5 * (dB - dA) / k, 0.0, 1.0);
    vec3 blended = mix(colA, colB, t);
    vec3 closer = (dA < dB) ? colA : colB;
    return mix(closer, blended, h);
}

// --- SCENE WRAPPER ---
// We wrap your external scene code here so we can mix it with the floor later
vec4 getSceneDist(vec3 p)
{
    {SCENE_CODE} // Your python code injects the object logic here
}

// --- MAIN MAP FUNCTION ---
// Combines your scene with the new floor
vec4 map(vec3 p)
{
    // 1. Get distance and color from your injected scene
    vec4 sceneRes = getSceneDist(p);
    float dScene = sceneRes.w;
    vec3 colScene = sceneRes.xyz;

    // 2. Define the Floor (Plane at y = -1.0)
    float floorHeight = -1.0;
    float dFloor = abs(p.y - floorHeight);

    // 3. Floor Color (Line pattern)
    float lineWidth = 0.9; // Adjust this value to change the thickness of the lines
    float lineX = step(1.0, mod(p.x, 1.0) / lineWidth); // Vertical lines
    float lineZ = step(1.0, mod(p.z, 1.0) / lineWidth); // Horizontal lines
    float linePattern = max(lineX, lineZ); // Combine lines
    vec3 colFloor = vec3(0.3) + 0.1 * linePattern; // Alternates between 0.3 and 0.4 gray

    // 4. Union (Return the closer object)
    if (dFloor < dScene) {
        return vec4(colFloor, dFloor);
    } else {
        return sceneRes;
    }
}

float FOV_ANGLE = {FOV_ANGLE_VAL}; 

// Raymarching function - returns distance
float rayMarch(vec3 ro, vec3 rd) {
    float dO = 0.0;
    for(int i = 0; i < 128; i++) {
        vec3 p = ro + rd * dO;
        vec4 res = map(p);
        float dS = res.w; 
        dO += dS;
        if(dS < 0.001 || dO > 100.0) break;
    }
    return dO;
}

// Raymarching function that also returns color
vec4 rayMarchWithColor(vec3 ro, vec3 rd) {
    float dO = 0.0;
    vec3 finalColor = vec3(0.0);
    for(int i = 0; i < 128; i++) {
        vec3 p = ro + rd * dO;
        vec4 res = map(p);
        float dS = res.w; 
        vec3 col = res.xyz; 
        dO += dS;
        if(dS < 0.001 || dO > 100.0) {
            finalColor = col;
            break;
        }
    }
    return vec4(finalColor, dO);
}

// Calculate normal using central differences
vec3 calcNormal(vec3 p) {
    float h = 0.001;
    vec2 k = vec2(1, -1);
    return normalize(k.xyy * map(p + k.xyy * h).w + 
                     k.yxy * map(p + k.yxy * h).w +
                     k.yyx * map(p + k.yyx * h).w +
                     k.xxx * map(p + k.xxx * h).w);
}

// Main image function
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Calculate normalized UV coordinates
    vec2 uv = (fragCoord - 0.825 * resolution.xy) / resolution.y;

    // --- Camera Setup ---
    vec3 ta = CamOrbit; 
    
    float pitch = clamp(camPitch, -1.55, 1.55); 
    float yaw = camYaw; 

    float x = radius * cos(pitch) * cos(yaw);
    float y = radius * sin(pitch);
    float z = radius * cos(pitch) * sin(yaw);

    vec3 ro = vec3(x, y, z) + ta; 

    vec3 ww = normalize(ta - ro);     
    vec3 up = vec3(0.0, 1.0, 0.0);    
    vec3 uu = normalize(cross(up, ww)); 
    vec3 vv = cross(ww, uu);          

    float fovFactor = 1.0 / tan(FOV_ANGLE * 0.5);
    vec3 rd = normalize(uu * uv.x * fovFactor + vv * uv.y * fovFactor + ww);

    // --- Raymarching ---
    vec4 rayResult = rayMarchWithColor(ro, rd);
    float d = rayResult.w;
    vec3 surfaceColor = rayResult.xyz;

    // --- Shading ---
    vec3 col = vec3(0.0);
    
    if(d < 100.0) {
        // Hit object or floor
        vec3 p = ro + rd * d;
        vec3 n = calcNormal(p);

        // Simple directional light
        vec3 lightDir = normalize(vec3(0.5, 0.8, 0.5));
        float diff = max(dot(n, lightDir), 0.0);

        // Ambient + Diffuse
        col = surfaceColor * diff + surfaceColor * vec3(0.15);  
        
        // Optional: Simple distance fog to blend floor into background
        float fog = 1.0 - exp(-d * 0.02);
        vec3 skyColor = vec3(0.1, 0.15, 0.25); // Match the background color below
        col = mix(col, skyColor, fog);
        
    } else {
        // Background (Sky) - Replaces the flat gray
        // Simple vertical gradient
        float t = 0.5 * (rd.y + 1.0);
        col = mix(vec3(0.1, 0.15, 0.25), vec3(0.05, 0.05, 0.1), t);
    }

    fragColor = vec4(col, 1.0);
}

void main() {
    mainImage(FragColor, gl_FragCoord.xy);
}
