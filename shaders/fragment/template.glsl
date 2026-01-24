#version 330 core
out vec4 FragColor;

uniform float time;
uniform vec2 resolution;
uniform vec2 viewportOffset;
uniform float camYaw;
uniform float camPitch;
uniform float radius = 5.0;
uniform vec3 CamOrbit = vec3(0.0);
uniform vec3 SkyColorTop = vec3(0.1, 0.15, 0.25);
uniform vec3 SkyColorBottom =  vec3(0.05, 0.05, 0.1);
uniform bool GridEnabled = true;
uniform vec3 MovePos;

// BVH uniforms
uniform samplerBuffer bvhNodeBuffer;
uniform int bvhNodeCount;
uniform int bvhRootIdx;

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


struct BVHNode {
    vec3 aabb_min;
    float left_idx;          // -1 for leaf
    vec3 aabb_max;
    float right_or_prim_idx; // right child idx for internal, prim_start for leaf
    float prim_count;        // 0 for internal, >0 for leaf
};

BVHNode getBVHNode(int idx) {
    int offset = idx * 9;
    BVHNode node;
    node.aabb_min = vec3(
        texelFetch(bvhNodeBuffer, offset).x,
        texelFetch(bvhNodeBuffer, offset + 1).x,
        texelFetch(bvhNodeBuffer, offset + 2).x
    );
    node.left_idx = texelFetch(bvhNodeBuffer, offset + 3).x;
    node.aabb_max = vec3(
        texelFetch(bvhNodeBuffer, offset + 4).x,
        texelFetch(bvhNodeBuffer, offset + 5).x,
        texelFetch(bvhNodeBuffer, offset + 6).x
    );
    node.right_or_prim_idx = texelFetch(bvhNodeBuffer, offset + 7).x;
    node.prim_count = texelFetch(bvhNodeBuffer, offset + 8).x;
    return node;
}

bool rayIntersectsAABB(vec3 rayOrigin, vec3 rayDir, vec3 aabbMin, vec3 aabbMax) {
    vec3 invDir = 1.0 / (abs(rayDir) + vec3(0.0001));
    invDir = mix(invDir, -invDir, lessThan(rayDir, vec3(0.0)));
    
    vec3 t1 = (aabbMin - rayOrigin) * invDir;
    vec3 t2 = (aabbMax - rayOrigin) * invDir;
    
    vec3 tMin = min(t1, t2);
    vec3 tMax = max(t1, t2);
    
    float tNear = max(max(tMin.x, tMin.y), tMin.z);
    float tFar = min(min(tMax.x, tMax.y), tMax.z);
    
    return tNear <= tFar && tFar >= 0.0;
}

// --- SCENE WRAPPER ---
vec4 getSceneDist(vec3 p)
{
    {SCENE_CODE}
}

// --- MAIN MAP FUNCTION ---
vec4 map(vec3 p) {
    vec4 sceneRes = getSceneDist(p);
    return sceneRes;
}

float FOV_ANGLE = {FOV_ANGLE_VAL}; 

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

// Grid
bool intersectPlane(vec3 ro, vec3 rd, out float t) {
    t = (-ro.y) / rd.y;
    return t > 0.0;
}

float gridFloor(vec3 worldPos, float cellSize, float lineThickness) {
    vec2 uv = worldPos.xz;
    vec2 aaWidth = fwidth(uv) * 0.5;
    vec2 grid = abs(fract(uv / cellSize - 0.5) - 0.5) * cellSize;
    grid = 1.0 - smoothstep(lineThickness - aaWidth, lineThickness + aaWidth, grid);
    return max(grid.x, grid.y);
}

vec3 add_grid(vec3 color, vec3 ro, vec3 rd, float depth) {
    float planeDepth = 0.0;
    bool hitPlane = intersectPlane(ro, rd, planeDepth);

    if (hitPlane && planeDepth < depth) {
        vec3 hitPoint = ro + rd * planeDepth;
        float gridIntensity =
            gridFloor(hitPoint, 4.0, 0.05) * 0.6 +
            gridFloor(hitPoint, 2.0, 0.03) * 0.8 +
            gridFloor(hitPoint, 0.5, 0.02) * 1.0;
        gridIntensity *= 0.45;
        float depthFade = 1.0-smoothstep(3.0, 30.0, planeDepth);
        if (gridIntensity > 0.0) {
            color = mix(color, vec3(1.0), gridIntensity*depthFade);
        }
    }
    return color;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 localFrag = fragCoord - viewportOffset;
    vec2 uv = (localFrag - 0.5 * resolution.xy) / resolution.y;

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

    vec4 rayResult = rayMarchWithColor(ro, rd);
    float d = rayResult.w;
    vec3 surfaceColor = rayResult.xyz;

    vec3 col = vec3(0.0);
    
    if(d < 100.0) {
        vec3 p = ro + rd * d;
        vec3 n = calcNormal(p);

        vec3 lightDir = normalize(vec3(0.5, 0.8, 0.5));
        float diff = max(dot(n, lightDir), 0.0);

        col = surfaceColor * diff + surfaceColor * vec3(0.15);  
        
        float fog = 1.0 - exp(-d * 0.02);
        vec3 skyColor = vec3(0.1, 0.15, 0.25);
        col = mix(col, skyColor, fog);
        
    } else {
        float t = 0.5 * (rd.y + 1.0);
        t = smoothstep(0.35, 0.5, t);
        col = mix(SkyColorBottom, SkyColorTop, t);
    }

    if(GridEnabled == true) col = add_grid(col,ro,rd,d);

    fragColor = vec4(col, 1.0);
}

void main() {
    mainImage(FragColor, gl_FragCoord.xy);
}