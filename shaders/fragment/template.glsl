#version 330 core
out vec4 FragColor;

uniform float time;
uniform vec2 resolution;
uniform vec2 viewportOffset;
uniform float camYaw;     // Horizontal rotation (around Y axis)
uniform float camPitch;   // Vertical rotation (around X axis)
uniform float radius = 5.0; // Camera orbit distance
uniform vec3 CamOrbit = vec3(0.0); // Center of orbit
uniform vec3 SkyColorTop = vec3(0.1, 0.15, 0.25);
uniform vec3 SkyColorBottom =  vec3(0.05, 0.05, 0.1);
uniform bool GridEnabled = true;

uniform vec3 LightDir = vec3(0.5, 0.5, -1.0);
uniform vec3 MovePos;
uniform vec3 MoveRot;

{ADDITIONAL_UNIFORMS}


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
// Improved grid floor pattern with anti-aliasing
float gridFloor(vec3 worldPos, float cellSize, float lineThickness) {
    // Use world XZ coordinates for grid
    vec2 uv = worldPos.xz;

    // Calculate anti-aliasing width based on screen derivatives
    vec2 aaWidth = fwidth(uv) * 0.5;

    // Create grid lines with smooth edges
    vec2 grid = abs(fract(uv / cellSize - 0.5) - 0.5) * cellSize;
    grid = 1.0 - smoothstep(lineThickness - aaWidth, lineThickness + aaWidth, grid);

    // Combine X and Z lines
    return max(grid.x, grid.y);
}

// Improved grid rendering with better blending and depth handling
vec3 add_grid(vec3 color, vec3 ro, vec3 rd, float depth) {
    // Plane intersection
    float planeDepth = 0.0;
    bool hitPlane = intersectPlane(ro, rd, planeDepth);

    if (hitPlane && planeDepth < depth) {
        // Get intersection point
        vec3 hitPoint = ro + rd * planeDepth;

        // Calculate grid intensity with multiple scales
        float gridIntensity =
            gridFloor(hitPoint, 4.0, 0.05) * 0.6 +
            gridFloor(hitPoint, 2.0, 0.03) * 0.8 +
            gridFloor(hitPoint, 0.5, 0.02) * 1.0;
        gridIntensity *= 0.45;

        // Apply depth-based fading
        float depthFade = 1.0-smoothstep(3.0, 30.0, planeDepth);

        // Blend grid with scene color
        if (gridIntensity > 0.0) {
            color = mix(color, vec3(1.0), gridIntensity*depthFade);
        }
    }

    return color;
}



// Struct to hold the results of the intersection test
struct IntersectionResult {
    bool intersects;
    vec2 uv;
    float t; // Distance along the ray
};

// Helper function to build an orthonormal basis for a given normal vector
void buildOrthonormalBasis(vec3 N, out vec3 U, out vec3 V) {
    // Robust method to find a vector not parallel to N
    vec3 T = abs(N.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    
    // U is perpendicular to N
    U = normalize(cross(T, N));
    
    // V is perpendicular to both N and U
    V = cross(N, U);
    // Since N and U are orthonormal, V is already normalized.
}


IntersectionResult intersectRayPlane(
    vec3 rayOrigin, 
    vec3 rayDirection, 
    vec3 planePoint, 
    vec3 planeNormal, 
    float planeWidth,
    float planeHeight
) {


    IntersectionResult result = IntersectionResult(false, vec2(0.0), 0.0);

    // 1. Calculate denominator (N dot D)
    float N_dot_D = dot(planeNormal, rayDirection);

    // Check for parallelism (near zero)
    if (abs(N_dot_D) < 1e-6) {
        return result; // Parallel or coplanar
    }

    // 2. Calculate t (distance along ray)
    vec3 P0_minus_O = planePoint - rayOrigin;
    float numerator = dot(planeNormal, P0_minus_O);
    
    float t = numerator / N_dot_D;

    // 3. Check if intersection is in front of the ray origin (t >= 0)
    if (t < 0.0) {
        return result; // Intersection is behind the ray origin
    }
    
    result.t = t;
    
    // 4. Calculate Intersection Point I
    vec3 I = rayOrigin + t * rayDirection;
    
    // 5. Calculate UV Mapping
    
    // Vector from the plane's reference point (P0) to the intersection point (I)
    vec3 V_rel = I - planePoint;
    
    vec3 U, V;
    buildOrthonormalBasis(planeNormal, U, V);

    // Project V_rel onto the basis vectors to get UV coordinates
    float u_raw = dot(V_rel, U);
    float v_raw = dot(V_rel, V);
    
    // Normalize UV by the size parameters to get texture coordinates [0, 1]
    float u_norm = u_raw / planeWidth;
    float v_norm = v_raw / planeHeight;

    // 6. Check Size Constraints (Is the intersection within the defined quad?)
    if (u_norm >= 0.0 && u_norm <= 1.0 && 
        v_norm >= 0.0 && v_norm <= 1.0) 
    {
        result.intersects = true;
        result.uv = vec2(u_norm, v_norm);
    }

    return result;
}




vec3 Sprite(
    // Plane
    vec3 rayOrigin, 
    vec3 rayDirection, 
    vec3 planePoint, 
    vec3 planeNormal, 
    float planeWidth,
    float planeHeight,

    // Scene
    vec3 SceneColor,
    float SceneDepth,

    // Texture
    sampler2D SprTexture,
    vec2 uvSize,
    float Alpha,
    float LOD
){

    vec3 col = SceneColor;

    IntersectionResult plane = intersectRayPlane(
        rayOrigin, rayDirection,
        planePoint, planeNormal,
        planeWidth, planeHeight
    );

    if( plane.intersects ){
        if(plane.t < SceneDepth){
            vec2 uv = 1.0-plane.uv*uvSize;
            vec4 spr_col = texture2DLod(SprTexture, uv, LOD);
            col = mix(col, spr_col.rgb, spr_col.a*Alpha);
        }
    }

    return col;
}






// Main image function
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // convert fragCoord to viewport-local coordinates (fragCoord is window coords)
    vec2 localFrag = fragCoord - viewportOffset;

    // use localFrag when building uv
    vec2 uv = (localFrag - 0.5 * resolution.xy) / resolution.y;


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
        float diff = max(dot(n, normalize(LightDir)), 0.0);

        // Ambient + Diffuse
        col = surfaceColor * diff + surfaceColor * vec3(0.15);  
        
        // Optional: Simple distance fog to blend floor into background
        float fog = 1.0 - exp(-d * 0.02);
        vec3 skyColor = vec3(0.1, 0.15, 0.25);
        col = mix(col, skyColor, fog);
        
    } else {
        float t = 0.5 * (rd.y + 1.0);
        t = smoothstep(0.35, 0.5, t);
        col = mix(SkyColorBottom, SkyColorTop, t);
    }

    if(GridEnabled == true) col = add_grid(col,ro,rd,d);

    {POSTPROC}

    fragColor = vec4(col, 1.0);
}

void main() {
    mainImage(FragColor, gl_FragCoord.xy);
}