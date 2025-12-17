// It's 3D grid. Useless. Idk.....................
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates
    vec2 uv = fragCoord.xy / iResolution.xy;

    // Camera parameters
    float cameraHeight = 1.0;
    vec3 cameraPosition = vec3(0.0, cameraHeight, 0.0);
    vec3 cameraTarget = vec3(0.0, 0.2, 1.0);
    vec3 cameraUp = vec3(0.0, 1.0, 0.0);

    // Calculate camera direction (forward vector)
    vec3 cameraForward = normalize(cameraTarget - cameraPosition);

    // Calculate camera right and up vectors in world space
    vec3 cameraRight = normalize(cross(cameraForward, cameraUp));
    vec3 cameraWorldUp = normalize(cross(cameraRight, cameraForward));

    // Calculate the direction for the current pixel
    vec3 pixelDirection = normalize(
        cameraForward +
        (uv.x - 0.5) * 2.0 * (iResolution.x / iResolution.y) * cameraRight +
        (uv.y - 0.5) * 2.0 * cameraWorldUp
    );


    float fogDensity = 0.4; // Fog density
    // Intersection with the floor (plane y = 0)
    float t = -cameraPosition.y / pixelDirection.y;
    
    vec3 background = vec3(0.3);
    vec3 color = vec3(0.0);

    // Check if intersection is valid (in front of camera)
    if (t > 0.0)
    {
        // Calculate the intersection point
        vec3 intersectionPoint = cameraPosition + t * pixelDirection;

        // Calculate UV coordinates for the floor
        vec2 floorUV = intersectionPoint.xz * 4.0; // Scale for grid size

        // Create grid pattern
        float line_mul = 24.0;
        float gridLineThickness = -.1;
        float pattern = smoothstep(gridLineThickness, gridLineThickness + 0.005*line_mul, min(fract(floorUV.x), fract(floorUV.y))) -
                        smoothstep(gridLineThickness + 0.005*line_mul, gridLineThickness + 0.01*line_mul, min(fract(floorUV.x), fract(floorUV.y)));
        vec3 surfaceColor = mix(background, vec3(1.0), pattern);

        // Fog
        float distance = t;
        float fogFactor = exp(-fogDensity * distance);
        color = mix(surfaceColor, background, 1.0 - fogFactor);
    } else {
        // No intersection or behind camera: draw background
        color = background;
    }
    fragColor = vec4(color,1.0);
}
