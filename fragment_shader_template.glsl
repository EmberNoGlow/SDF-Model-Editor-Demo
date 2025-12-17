#version 330 core
out vec4 FragColor;

uniform float time;
uniform vec2 resolution;
uniform float camYaw;     // Horizontal rotation (around Y axis)
uniform float camPitch;   // Vertical rotation (around X axis), assuming this is the elevation angle
uniform float radius = 5.0; // Camera orbit distance
uniform vec3 CamOrbit = vec3(0.0); // Center of orbit

// Assuming these are replaced by your external code
{SDF_LIBRARY}

// Color mixing function for smooth operations
// Blends colors smoothly when surfaces are close together
vec3 mixColorSmooth(vec3 colA, vec3 colB, float dA, float dB, float k) {
    k *= 4.0;
    float h = max(k - abs(dA - dB), 0.0) / k;
    // Calculate blend factor based on which surface is closer
    // When dA < dB, we're closer to surface A, so use more of colA
    float t = clamp(0.5 + 0.5 * (dB - dA) / k, 0.0, 1.0);
    // Blend: when h is high (surfaces close), blend colors; when h is low, use closer surface color
    vec3 blended = mix(colA, colB, t);
    vec3 closer = (dA < dB) ? colA : colB;
    return mix(closer, blended, h);
}

vec4 map(vec3 p)
{
    {SCENE_CODE} // For setup from python code, dont worry
}
float FOV_ANGLE = {FOV_ANGLE_VAL}; // Placeholder for your actual FOV value (in radians if possible)

// Raymarching function - returns distance
float rayMarch(vec3 ro, vec3 rd) {
    float dO = 0.0;
    // Increased iteration count for better scene penetration
    for(int i = 0; i < 128; i++) {
        vec3 p = ro + rd * dO;
        vec4 res = map(p);
        float dS = res.w; // Distance is in w component
        dO += dS;
        // Improved hit check (lower tolerance)
        if(dS < 0.001 || dO > 100.0) break;
    }
    return dO;
}

// Raymarching function that also returns color
vec4 rayMarchWithColor(vec3 ro, vec3 rd) {
    float dO = 0.0;
    vec3 finalColor = vec3(0.0);
    // Increased iteration count for better scene penetration
    for(int i = 0; i < 128; i++) {
        vec3 p = ro + rd * dO;
        vec4 res = map(p);
        float dS = res.w; // Distance is in w component
        vec3 col = res.xyz; // Color is in xyz components
        dO += dS;
        // Improved hit check (lower tolerance)
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
    // It's correct, not error, its working good.
    vec2 uv = (fragCoord - 0.825 * resolution.xy) / resolution.y;

    // --- Camera Setup: Spherical Orbit ---
    
    // Target (center of orbit)
    vec3 ta = CamOrbit; //vec3(0.0, 0.0, 0.0);
    
    // Convert pitch and yaw into a standard spherical coordinate position (ro)
    // Ensure camPitch is clamped to avoid flipping (e.g., -PI/2 to PI/2, or -1.57 to 1.57 radians)
    float pitch = clamp(camPitch, -1.55, 1.55); // Clamp near 90 degrees
    float yaw = camYaw; // Full 360 rotation

    // Standard Spherical Coordinates:
    float x = radius * cos(pitch) * cos(yaw);
    float y = radius * sin(pitch);
    float z = radius * cos(pitch) * sin(yaw);

    // Camera Origin (Position)
    vec3 ro = vec3(x, y, z) + ta; // Add target/center offset if ta isn't (0,0,0)

    // Calculate camera vectors
    vec3 ww = normalize(ta - ro);     // Forward vector (Z-axis of camera space)
    vec3 up = vec3(0.0, 1.0, 0.0);    // World Up (used to define horizontal orientation)
    vec3 uu = normalize(cross(up, ww)); // Right vector (X-axis of camera space)
    vec3 vv = cross(ww, uu);          // Up vector (Y-axis of camera space, guaranteed orthogonal)

    // --- Ray Direction Setup: Corrected Perspective Projection ---

    // Assuming FOV_ANGLE is in radians
    float fovFactor = 1.0 / tan(FOV_ANGLE * 0.5);
    
    // Calculate Ray Direction (rd)
    // The projection plane is at distance 1/fovFactor = tan(half_fov) away from the camera.
    // The components uv.x and uv.y are multiplied by the fovFactor to achieve the correct angular spread.
    vec3 rd = normalize(uu * uv.x * fovFactor + vv * uv.y * fovFactor + ww);

    // --- End Camera Setup ---

    // Raymarching with color
    vec4 rayResult = rayMarchWithColor(ro, rd);
    float d = rayResult.w;
    vec3 surfaceColor = rayResult.xyz;

    // Shading
    vec3 col = vec3(0.0);
    if(d < 100.0) {
        vec3 p = ro + rd * d;
        vec3 n = calcNormal(p);

        // Lighting (Diffuse)
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
        float diff = max(dot(n, lightDir), 0.0);

        // Apply lighting and a small ambient term
        col = surfaceColor * diff + surfaceColor * vec3(0.1);  
    } else {
        // Background color (e.g., sky)
        col = vec3(0.3); 
    }

    fragColor = vec4(col, 1.0);
}

void main() {
    mainImage(FragColor, gl_FragCoord.xy);
}