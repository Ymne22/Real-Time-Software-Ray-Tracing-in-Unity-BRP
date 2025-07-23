//RaytracingController.cs

#if UNITY_EDITOR
using UnityEditor;
#endif

using UnityEngine;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;

//Enable bottom line to run in the scene view and allow image effects
[ExecuteInEditMode]
[ImageEffectAllowedInSceneView]

[RequireComponent(typeof(Camera))]
public class RayTracingController : MonoBehaviour
{
    public enum ResolutionScalePreset { Full = 1, Half = 2, Quarter = 4 }

    [Header("Performance")]
    [Tooltip("Render at a lower resolution to improve performance.")]
    public ResolutionScalePreset resolutionScale = ResolutionScalePreset.Half;
    [Tooltip("Maximum number of bounces for ray tracing, higher values increase quality but reduce performance.")]
    [Range(1, 32)]
    public int MaxBounces = 3;
    [Tooltip("Number of samples per pixel, higher values increase quality but reduce performance.")]
    [Range(1, 64)]
    public int SamplesPerPixel = 1;

    [Header("Temporal Capture (Only static scenes, Only on camera)")]
    [Tooltip("Use temporal capture for the scenes, this will accumulate samples over time for a smoother result.")]
    public bool useTemporalCapture = true;

    [Header("Shader and Lighting")]
    [Tooltip("Ray tracing compute shader to use for rendering.")]
    public ComputeShader RayTracingShader;

    [Header("Procedural Sky")]
    [Tooltip("The colors used for the procedural skybox.")]
    public Color DayZenithColor = new Color(0.2f, 0.5f, 0.9f);
    public Color DayHorizonColor = new Color(0.5f, 0.8f, 1.0f);
    public Color NightZenithColor = new Color(0.01f, 0.02f, 0.05f);
    public Color NightHorizonColor = new Color(0.05f, 0.1f, 0.25f);

    [Header("Settings")]
    [Tooltip("Use smooth shading for the scene")]
    public bool UseSmoothShading = true;
    [Tooltip("Use textures for the scene")]
    public bool UseTextures = true;
    [Tooltip("Size of the texture array for materials")]
    [Range(1, 4096)]
    public int TextureSize = 1024;

    [Header("Automatic Scene Refresh")]
    [Tooltip("Automatically refresh the scene every few seconds to capture changes in the scene.")]
    public bool autoRefreshScene = true;
    [Tooltip("Interval in seconds to check for scene changes and refresh if necessary.")]
    public float refreshCheckInterval = 1.0f;
    [Tooltip("Last time the scene was checked for changes.")]
    private float _lastCheckTime = 0f;

    [Header("Rasterization Fallback")]
    [Tooltip("Combines the ray-traced output with the rasterized fallback. If disabled, shows only the pure ray-traced result.")]
    public bool CombineWithRasterize = true;

    [Header("Feature Toggles")]
    [Tooltip("Enable reflection, requires indirect lighting calculations to enable this.")]
    public bool EnableReflection = true;
    [Tooltip("Enable shadow calculations for directional lights only other than that using rasterize, stack the rasterized scene with the ray traced scene for shadows.")]
    public bool EnableRayTraceDirectLight = true;
    [Tooltip("Enable indirect lighting calculations to enable this.")]
    public bool EnableRayTraceIndirectLight = true;

    [Header("Feature Toggles (Experimental)")]
    [Tooltip("Enable ambient occlusion, requires indirect lighting calculations to enable this. really expensive, use with caution.")]
    public bool EnableAmbientOcclusion = false;

    [Header("BVH Debug Visualization")]
    [Tooltip("Show BVH gizmos in the scene view for debugging.")]
    public bool showBVHGizmos = false;
    [Tooltip("Highlight leaf nodes in the BVH with a different color.")]
    public bool highlightLeafNodes = true;
    [Tooltip("Width of the BVH gizmo lines in the scene view.")]
    [Range(0.01f, 0.1f)]
    public float gizmoLineWidth = 0.05f;
    [Tooltip("Color of the BVH gizmos in the scene view.")]
    public Color gizmoColor = Color.green;
    [Tooltip("Color for leaf nodes in the BVH gizmos.")]
    public Color leafNodeColor = Color.red;

    // Internal State
    private Camera _camera;
    private RenderTexture _target;
    private RenderTexture _rasterizedScene;
    private uint _currentSample = 0;
    private Light _sunLight;


    // Data Lists
    private List<MeshObject> _meshObjects = new List<MeshObject>();
    private List<Vector3> _vertices = new List<Vector3>();
    private List<Vector3> _normals = new List<Vector3>();
    private List<Vector4> _tangents = new List<Vector4>();
    private List<Vector2> _uvs = new List<Vector2>();
    private List<Vector3Int> _indices = new List<Vector3Int>();
    private List<MaterialData> _materials = new List<MaterialData>();
    private List<Texture2D> _textures = new List<Texture2D>();
    private List<BVHNode> _bvhNodes = new List<BVHNode>();
    private List<MeshRenderer> _sceneRenderers = new List<MeshRenderer>();
    private List<LightData> _lights = new List<LightData>();

    // Compute Buffers
    private ComputeBuffer _meshObjectBuffer;
    private ComputeBuffer _vertexBuffer;
    private ComputeBuffer _normalBuffer;
    private ComputeBuffer _tangentBuffer;
    private ComputeBuffer _uvBuffer;
    private ComputeBuffer _indexBuffer;
    private ComputeBuffer _materialBuffer;
    private ComputeBuffer _bvhBuffer;
    private ComputeBuffer _lightBuffer;
    private Texture2DArray _textureArray;

    // Struct Definitions
    [StructLayout(LayoutKind.Sequential)]
    struct MeshObject
    {
        public Matrix4x4 localToWorldMatrix, worldToLocalMatrix;
        public int indices_offset, indices_count, materialIndex; }


    [StructLayout(LayoutKind.Sequential)]
    struct MaterialData
    {
        public Vector3 albedo, specular, emission;
        public float transparency;
        public Vector2 textureScale, textureOffset;
        public int textureIndex, metallicIndex, normalIndex; }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    struct BVHNode
    {
        public Vector3 aabb_min;
        public Vector3 aabb_max;
        public int first_child_or_object_index;
        public int object_count;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct LightData
    {
        public Vector3 position;
        public Vector3 direction;
        public Vector3 color;
        public float intensity;
        public float range;
        public float spotAngle;
        public int type;
        private float _padding;
    }


    void Start()
    {
        _camera = GetComponent<Camera>();
        RefreshScene();
    }

    void Update()
    {
        if ((_sunLight != null && _sunLight.transform.hasChanged) || _camera.transform.hasChanged)
        {
            _currentSample = 0;
            if (_sunLight != null) _sunLight.transform.hasChanged = false;
            _camera.transform.hasChanged = false;
        }

        if (autoRefreshScene && !useTemporalCapture && Time.realtimeSinceStartup - _lastCheckTime >= refreshCheckInterval)
        {
            _lastCheckTime = Time.realtimeSinceStartup;
            RefreshScene();
        }
    }

    void InitRasterizedSceneTexture()
    {
        int scale = (int)resolutionScale;
        int scaledWidth = Screen.width / scale;
        int scaledHeight = Screen.height / scale;

        if (_rasterizedScene == null || _rasterizedScene.width != scaledWidth || _rasterizedScene.height != scaledHeight)
        {
            if (_rasterizedScene != null) _rasterizedScene.Release();
            _rasterizedScene = new RenderTexture(scaledWidth, scaledHeight, 0, RenderTextureFormat.ARGB32);
            _rasterizedScene.Create();
        }
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        InitRenderTexture();
        InitRasterizedSceneTexture();
        Graphics.Blit(source, _rasterizedScene);

        SetShaderParameters();
        RayTracingShader.SetTexture(0, "_RasterizedScene", _rasterizedScene);

        int scale = (int)resolutionScale;
        int threadGroupsX = Mathf.CeilToInt(Screen.width / (scale * 8.0f));
        int threadGroupsY = Mathf.CeilToInt(Screen.height / (scale * 8.0f));
        RayTracingShader.Dispatch(0, threadGroupsX, threadGroupsY, 1);

        Graphics.Blit(_target, destination);
        if (useTemporalCapture) _currentSample++; else _currentSample = 0;

#if UNITY_EDITOR
        bool isSceneView = !Application.isPlaying && Camera.current != null && Camera.current.cameraType == CameraType.SceneView;
        if (isSceneView)
            _currentSample = 0;
        else if (useTemporalCapture)
            _currentSample++;
#else
            if (useTemporalCapture)
                _currentSample++;
#endif
    }

    void OnEnable()
    {
        TryFindSunLight();
        RefreshScene();
    }

    private void TryFindSunLight()
    {
        _sunLight = null;
        Light[] lights = FindObjectsOfType<Light>();
        foreach (Light light in lights)
        {
            if (light.type == LightType.Directional && light.enabled)
            {
                _sunLight = light;
                Debug.Log("Ray Tracing: Found directional sun light '" + light.name + "'.");
                return;
            }
        }
        if (_sunLight == null)
        {
            Debug.LogWarning("Ray Tracing: No enabled directional light found in scene to use as sun.");
        }
    }

    void OnDisable()
    {
        ReleaseBuffers();
        if (_target != null) _target.Release();
    }


    void OnDrawGizmos()
    {
        if (!showBVHGizmos || _bvhNodes == null || _bvhNodes.Count == 0) return;

        Matrix4x4 originalMatrix = Gizmos.matrix;
        Color originalColor = Gizmos.color;

        for (int i = 0; i < _bvhNodes.Count; i++)
        {
            BVHNode node = _bvhNodes[i];
            Vector3 center = (node.aabb_min + node.aabb_max) * 0.5f;
            Vector3 size = node.aabb_max - node.aabb_min;

            if (size.x <= 0 || size.y <= 0 || size.z <= 0) continue;

            if (node.object_count > 0 && highlightLeafNodes)
            {
                Gizmos.color = leafNodeColor;
                DrawWireCubeWithWidth(center, size, gizmoLineWidth * 1.5f);
            }
            else
            {
                Gizmos.color = gizmoColor;
                DrawWireCubeWithWidth(center, size, gizmoLineWidth);
            }
        }

        Gizmos.color = originalColor;
        Gizmos.matrix = originalMatrix;
    }

    void DrawWireCubeWithWidth(Vector3 center, Vector3 size, float width)
    {
        Vector3 halfSize = size * 0.5f;

        DrawThickLine(center + new Vector3(-halfSize.x, -halfSize.y, -halfSize.z), center + new Vector3(halfSize.x, -halfSize.y, -halfSize.z), width);
        DrawThickLine(center + new Vector3(halfSize.x, -halfSize.y, -halfSize.z), center + new Vector3(halfSize.x, -halfSize.y, halfSize.z), width);
        DrawThickLine(center + new Vector3(halfSize.x, -halfSize.y, halfSize.z), center + new Vector3(-halfSize.x, -halfSize.y, halfSize.z), width);
        DrawThickLine(center + new Vector3(-halfSize.x, -halfSize.y, halfSize.z), center + new Vector3(-halfSize.x, -halfSize.y, -halfSize.z), width);

        DrawThickLine(center + new Vector3(-halfSize.x, halfSize.y, -halfSize.z), center + new Vector3(halfSize.x, halfSize.y, -halfSize.z), width);
        DrawThickLine(center + new Vector3(halfSize.x, halfSize.y, -halfSize.z), center + new Vector3(halfSize.x, halfSize.y, halfSize.z), width);
        DrawThickLine(center + new Vector3(halfSize.x, halfSize.y, halfSize.z), center + new Vector3(-halfSize.x, halfSize.y, halfSize.z), width);
        DrawThickLine(center + new Vector3(-halfSize.x, halfSize.y, halfSize.z), center + new Vector3(-halfSize.x, halfSize.y, -halfSize.z), width);

        DrawThickLine(center + new Vector3(-halfSize.x, -halfSize.y, -halfSize.z), center + new Vector3(-halfSize.x, halfSize.y, -halfSize.z), width);
        DrawThickLine(center + new Vector3(halfSize.x, -halfSize.y, -halfSize.z), center + new Vector3(halfSize.x, halfSize.y, -halfSize.z), width);
        DrawThickLine(center + new Vector3(halfSize.x, -halfSize.y, halfSize.z), center + new Vector3(halfSize.x, halfSize.y, halfSize.z), width);
        DrawThickLine(center + new Vector3(-halfSize.x, -halfSize.y, halfSize.z), center + new Vector3(-halfSize.x, halfSize.y, halfSize.z), width);
    }

    void DrawThickLine(Vector3 start, Vector3 end, float width)
    {
        Vector3 direction = (end - start).normalized;
        Vector3 perpendicular = Vector3.Cross(direction, Camera.current != null ?
            Camera.current.transform.forward : Vector3.forward).normalized * width;

        Gizmos.DrawLine(start - perpendicular, end - perpendicular);
        Gizmos.DrawLine(start + perpendicular, end + perpendicular);
        Gizmos.DrawLine(start - perpendicular, start + perpendicular);
        Gizmos.DrawLine(end - perpendicular, end + perpendicular);
    }

    public void RefreshScene()
    {
        _currentSample = 0;
        ReleaseBuffers();
        Debug.Log("Building Ray Tracing Scene...");
        ConvertSceneToData();
        CollectLights();
        BuildBVH();
        InitComputeBuffers();
    }

    void CollectLights()
    {
        _lights.Clear();
        Light[] sceneLights = FindObjectsOfType<Light>();

        foreach (Light light in sceneLights)
        {
            if (!light.enabled) continue;

            LightData data = new LightData
            {
                position = light.transform.position,
                direction = light.transform.forward,
                color = new Vector3(light.color.linear.r, light.color.linear.g, light.color.linear.b),
                intensity = light.intensity,
                range = light.range,
            };

            switch (light.type)
            {
                case LightType.Directional:
                    data.type = 0;
                    break;
                case LightType.Point:
                    data.type = 1;
                    break;
                case LightType.Spot:
                    data.type = 2;
                    data.spotAngle = Mathf.Cos(Mathf.Deg2Rad * light.spotAngle * 0.5f);
                    break;
                default:
                    continue;
            }
            _lights.Add(data);
        }
    }

    void ConvertSceneToData()
    {
        _meshObjects.Clear(); _vertices.Clear(); _normals.Clear(); _uvs.Clear(); _indices.Clear();
        _materials.Clear(); _textures.Clear(); _sceneRenderers.Clear();

        var materialMap = new Dictionary<Material, int>();
        _materials.Add(new MaterialData() { albedo = Vector3.one * 0.8f, specular = Vector3.one * 0.2f });

        MeshRenderer[] renderers = FindObjectsOfType<MeshRenderer>();
        foreach (var renderer in renderers)
        {
            if (!renderer.enabled || renderer.sharedMaterial == null) continue;
            MeshFilter meshFilter = renderer.GetComponent<MeshFilter>();
            if (meshFilter == null || meshFilter.sharedMesh == null) continue;

            Mesh mesh = meshFilter.sharedMesh;

            if (!materialMap.TryGetValue(renderer.sharedMaterial, out int materialIndex))
            {
                materialIndex = _materials.Count;
                materialMap[renderer.sharedMaterial] = materialIndex;
                _materials.Add(CreateMaterialData(renderer.sharedMaterial));
            }

            int firstVertex = _vertices.Count;
            _vertices.AddRange(mesh.vertices);
            _normals.AddRange(mesh.normals);
            _tangents.AddRange(mesh.tangents);
            var uvs = new List<Vector2>(); mesh.GetUVs(0, uvs);
            if (uvs.Count < mesh.vertexCount) uvs.AddRange(Enumerable.Repeat(Vector2.zero, mesh.vertexCount - uvs.Count));
            _uvs.AddRange(uvs);

            int firstIndex = _indices.Count;
            var tris = mesh.triangles;
            for (int i = 0; i < tris.Length; i += 3)
                _indices.Add(new Vector3Int(tris[i] + firstVertex, tris[i + 1] + firstVertex, tris[i + 2] + firstVertex));

            _meshObjects.Add(new MeshObject { materialIndex = materialIndex, indices_offset = firstIndex, indices_count = tris.Length / 3 });
            _sceneRenderers.Add(renderer);
        }
        CreateTextureArray();
    }

    void BuildBVH()
    {
        // Clear previous BVH data
        _bvhNodes.Clear();

        // If no mesh objects, skip BVH building
        if (_meshObjects.Count == 0) return;

        // Pre-compute all bounds once
        var objectBounds = new List<Bounds>(_meshObjects.Count);
        for (int i = 0; i < _meshObjects.Count; i++)
        {
            objectBounds.Add(_sceneRenderers[i].bounds);
        }

        // Create object indices array
        var objectIndices = new List<int>(_meshObjects.Count);
        for (int i = 0; i < _meshObjects.Count; i++)
        {
            objectIndices.Add(i);
        }

        // Build recursively
        Subdivide(objectIndices, 0, objectIndices.Count, objectBounds);

        // Reorder objects to match BVH leaves (better cache coherence)
        var reorderedObjects = new List<MeshObject>(_meshObjects.Count);
        var reorderedRenderers = new List<MeshRenderer>(_sceneRenderers.Count);

        ReorderObjectsDFS(0, ref reorderedObjects, ref reorderedRenderers, objectIndices);

        _meshObjects = reorderedObjects;
        _sceneRenderers = reorderedRenderers;
    }

    void ReorderObjectsDFS(int nodeIndex, ref List<MeshObject> reorderedObjects, ref List<MeshRenderer> reorderedRenderers, List<int> originalIndices)
    {
        BVHNode node = _bvhNodes[nodeIndex];

        if (node.object_count > 0)
        {
            for (int i = 0; i < node.object_count; i++)
            {
                int originalIdx = originalIndices[node.first_child_or_object_index + i];
                reorderedObjects.Add(_meshObjects[originalIdx]);
                reorderedRenderers.Add(_sceneRenderers[originalIdx]);
            }
        }
        else
        {
            ReorderObjectsDFS(node.first_child_or_object_index, ref reorderedObjects, ref reorderedRenderers, originalIndices);
            ReorderObjectsDFS(node.first_child_or_object_index + 1, ref reorderedObjects, ref reorderedRenderers, originalIndices);
        }
    }

    void Subdivide(List<int> objectIndices, int first, int count, List<Bounds> objectBounds)
    {
        if (count == 0) return;

        var queue = new Queue<(int nodeIndex, int first, int count)>();
        queue.Enqueue((_bvhNodes.Count, first, count));
        _bvhNodes.Add(new BVHNode());

        while (queue.Count > 0)
        {
            var (nodeIndex, f, c) = queue.Dequeue();

            Bounds nodeBounds = objectBounds[objectIndices[f]];
            for (int i = 1; i < c; i++) nodeBounds.Encapsulate(objectBounds[objectIndices[f + i]]);

            var node = new BVHNode { aabb_min = nodeBounds.min, aabb_max = nodeBounds.max };

            if (c <= 2)
            {
                node.first_child_or_object_index = f;
                node.object_count = c;
                _bvhNodes[nodeIndex] = node;
                continue;
            }

            // SAH-based split
            int bestAxis = -1;
            int bestSplit = -1;
            float bestCost = float.MaxValue;

            for (int axis = 0; axis < 3; axis++)
            {
                objectIndices.Sort(f, c, Comparer<int>.Create((a, b) =>
                    objectBounds[a].center[axis].CompareTo(objectBounds[b].center[axis])));

                for (int split = 1; split < c; split++)
                {
                    Bounds leftBounds = objectBounds[objectIndices[f]];
                    Bounds rightBounds = objectBounds[objectIndices[f + split]];

                    for (int i = 1; i < split; i++) leftBounds.Encapsulate(objectBounds[objectIndices[f + i]]);
                    for (int i = split; i < c; i++) rightBounds.Encapsulate(objectBounds[objectIndices[f + i]]);

                    float leftArea = 2 * (leftBounds.size.x * leftBounds.size.y + leftBounds.size.y * leftBounds.size.z + leftBounds.size.z * leftBounds.size.x);
                    float rightArea = 2 * (rightBounds.size.x * rightBounds.size.y + rightBounds.size.y * rightBounds.size.z + rightBounds.size.z * rightBounds.size.x);
                    float cost = leftArea * split + rightArea * (c - split);

                    if (cost < bestCost)
                    {
                        bestAxis = axis;
                        bestSplit = split;
                        bestCost = cost;
                    }
                }
            }

            objectIndices.Sort(f, c, Comparer<int>.Create((a, b) =>
                objectBounds[a].center[bestAxis].CompareTo(objectBounds[b].center[bestAxis])));

            int mid = f + bestSplit;

            int leftChildIndex = _bvhNodes.Count;
            _bvhNodes.Add(new BVHNode());
            int rightChildIndex = _bvhNodes.Count;
            _bvhNodes.Add(new BVHNode());

            node.object_count = 0;
            node.first_child_or_object_index = leftChildIndex;
            _bvhNodes[nodeIndex] = node;

            queue.Enqueue((leftChildIndex, f, mid - f));
            queue.Enqueue((rightChildIndex, mid, c - (mid - f)));
        }
    }

    void InitComputeBuffers()
    {
        _meshObjectBuffer = new ComputeBuffer(_meshObjects.Count > 0 ? _meshObjects.Count : 1, Marshal.SizeOf(typeof(MeshObject)));
        _vertexBuffer = new ComputeBuffer(_vertices.Count > 0 ? _vertices.Count : 1, Marshal.SizeOf(typeof(Vector3)));
        _normalBuffer = new ComputeBuffer(_normals.Count > 0 ? _normals.Count : 1, Marshal.SizeOf(typeof(Vector3)));
        _tangentBuffer = new ComputeBuffer(_tangents.Count > 0 ? _tangents.Count : 1, Marshal.SizeOf(typeof(Vector4)));
        _uvBuffer = new ComputeBuffer(_uvs.Count > 0 ? _uvs.Count : 1, Marshal.SizeOf(typeof(Vector2)));
        _indexBuffer = new ComputeBuffer(_indices.Count > 0 ? _indices.Count : 1, Marshal.SizeOf(typeof(Vector3Int)));
        _materialBuffer = new ComputeBuffer(_materials.Count > 0 ? _materials.Count : 1, Marshal.SizeOf(typeof(MaterialData)));
        _bvhBuffer = new ComputeBuffer(_bvhNodes.Count > 0 ? _bvhNodes.Count : 1, Marshal.SizeOf(typeof(BVHNode)));
        _lightBuffer = new ComputeBuffer(_lights.Count > 0 ? _lights.Count : 1, Marshal.SizeOf(typeof(LightData)));

        if (_meshObjects.Count > 0) _meshObjectBuffer.SetData(_meshObjects);
        if (_vertices.Count > 0) _vertexBuffer.SetData(_vertices);
        if (_normals.Count > 0) _normalBuffer.SetData(_normals);
        if (_uvs.Count > 0) _uvBuffer.SetData(_uvs);
        if (_indices.Count > 0) _indexBuffer.SetData(_indices);
        if (_materials.Count > 0) _materialBuffer.SetData(_materials);
        if (_bvhNodes.Count > 0) _bvhBuffer.SetData(_bvhNodes);
        if (_lights.Count > 0) _lightBuffer.SetData(_lights);
    }

    void SetShaderParameters()
    {
        for (int i = 0; i < _meshObjects.Count; i++)
        {
            var obj = _meshObjects[i];
            obj.localToWorldMatrix = _sceneRenderers[i].localToWorldMatrix;
            obj.worldToLocalMatrix = _sceneRenderers[i].worldToLocalMatrix;
            _meshObjects[i] = obj;
        }
        if (_meshObjects.Count > 0) _meshObjectBuffer.SetData(_meshObjects);

        if (!useTemporalCapture)
        {
            BuildBVH();
            if (_bvhNodes.Count > 0) _bvhBuffer.SetData(_bvhNodes);
        }

        CollectLights();
        if (_lightBuffer == null || _lightBuffer.count != (_lights.Count > 0 ? _lights.Count : 1))
        {
            _lightBuffer?.Release();
            _lightBuffer = new ComputeBuffer(_lights.Count > 0 ? _lights.Count : 1, Marshal.SizeOf(typeof(LightData)));
        }
        if (_lights.Count > 0) _lightBuffer.SetData(_lights);


        RayTracingShader.SetMatrix("_CameraToWorld", _camera.cameraToWorldMatrix);
        RayTracingShader.SetMatrix("_CameraInverseProjection", _camera.projectionMatrix.inverse);
        RayTracingShader.SetInt("_LightCount", _lights.Count);
        RayTracingShader.SetInt("_ObjectCount", _meshObjects.Count);
        RayTracingShader.SetInt("_MaterialCount", _materials.Count);
        RayTracingShader.SetInt("_TextureCount", _textures.Count);
        RayTracingShader.SetBuffer(0, "_Lights", _lightBuffer);
        RayTracingShader.SetBuffer(0, "_MeshObjects", _meshObjectBuffer);
        RayTracingShader.SetBuffer(0, "_Vertices", _vertexBuffer);
        RayTracingShader.SetBuffer(0, "_Normals", _normalBuffer);
        RayTracingShader.SetBuffer(0, "_Tangents", _tangentBuffer);
        RayTracingShader.SetBuffer(0, "_UVs", _uvBuffer);
        RayTracingShader.SetBuffer(0, "_Indices", _indexBuffer);
        RayTracingShader.SetBuffer(0, "_Materials", _materialBuffer);
        RayTracingShader.SetBuffer(0, "_BVH", _bvhBuffer);
        RayTracingShader.SetTexture(0, "_TextureArray", _textureArray);
        RayTracingShader.SetBool("_UseTemporalCapture", useTemporalCapture);
        RayTracingShader.SetBool("_UseSmoothShading", UseSmoothShading);
        RayTracingShader.SetBool("_UseTextures", UseTextures);
        RayTracingShader.SetInt("_FrameIndex", (int)_currentSample);
        RayTracingShader.SetInt("_MaxBounces", MaxBounces);
        RayTracingShader.SetBool("_CombineWithRasterize", CombineWithRasterize);
        RayTracingShader.SetBool("_EnableReflection", EnableReflection);
        RayTracingShader.SetBool("_EnableRayTraceDirectLight", EnableRayTraceDirectLight);
        RayTracingShader.SetBool("_EnableRayTraceIndirectLight", EnableRayTraceIndirectLight);
        RayTracingShader.SetBool("_EnableAmbientOcclusion", EnableAmbientOcclusion);
        RayTracingShader.SetInt("_SamplesPerPixel", SamplesPerPixel);

        // Set Skybox parameters
        RayTracingShader.SetVector("_DayZenithColor", DayZenithColor.linear);
        RayTracingShader.SetVector("_DayHorizonColor", DayHorizonColor.linear);
        RayTracingShader.SetVector("_NightZenithColor", NightZenithColor.linear);
        RayTracingShader.SetVector("_NightHorizonColor", NightHorizonColor.linear);
        if (_sunLight != null)
        {
            RayTracingShader.SetVector("_SunLightDirection", _sunLight.transform.forward);
        }
        else
        {
            RayTracingShader.SetVector("_SunLightDirection", Vector3.down);
        }
    }

    void InitRenderTexture()
    {
        int scale = (int)resolutionScale;
        int scaledWidth = Screen.width / scale;
        int scaledHeight = Screen.height / scale;

        if (_target == null || _target.width != scaledWidth || _target.height != scaledHeight)
        {
            if (_target != null) _target.Release();
            _target = new RenderTexture(scaledWidth, scaledHeight, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            _target.enableRandomWrite = true;
            _target.Create();
            _currentSample = 0;
        }
        RayTracingShader.SetTexture(0, "Result", _target);
    }

    MaterialData CreateMaterialData(Material mat)
    {
        MaterialData data = new MaterialData();

        if (mat.HasProperty("_Color"))
        {
            data.albedo = new Vector3(mat.color.r, mat.color.g, mat.color.b);
            if (mat.IsKeywordEnabled("_ALPHABLEND_ON") || mat.renderQueue >= 3000)
            {
                data.transparency = 1.0f - mat.color.a;
            }
            else
            {
                data.transparency = 0.0f;
            }
        }
        else
        {
            data.albedo = Vector3.one;
            data.transparency = 0.0f;
        }

        data.specular = mat.HasProperty("_Metallic") ? Vector3.one * mat.GetFloat("_Metallic") : Vector3.one * 0.1f;
        data.emission = mat.HasProperty("_EmissionColor") ? new Vector3(mat.GetColor("_EmissionColor").r, mat.GetColor("_EmissionColor").g, mat.GetColor("_EmissionColor").b) : Vector3.zero;
        float smoothness = mat.HasProperty("_Glossiness") ? mat.GetFloat("_Glossiness") : 0.5f;
        data.specular.y = 1.0f - smoothness;

        if (mat.HasProperty("_MainTex_ST"))
        {
            Vector4 st = mat.GetVector("_MainTex_ST");
            data.textureScale = new Vector2(st.x, st.y);
            data.textureOffset = new Vector2(st.z, st.w);
        }
        else
        {
            data.textureScale = Vector2.one;
            data.textureOffset = Vector2.zero;
        }
        data.textureIndex = GetTextureIndex(mat, "_MainTex");
        data.metallicIndex = GetTextureIndex(mat, "_MetallicGlossMap");
        data.normalIndex = GetTextureIndex(mat, "_BumpMap");
        return data;
    }

    int GetTextureIndex(Material mat, string propertyName)
    {
        if (!UseTextures || !mat.HasProperty(propertyName)) return -1;
        Texture tex = mat.GetTexture(propertyName);
        if (tex == null) return -1;
        int index = _textures.IndexOf(tex as Texture2D);
        if (index == -1)
        {
            index = _textures.Count;
            _textures.Add(tex as Texture2D);
        }
        return index;
    }

    void CreateTextureArray()
    {
        if (_textures.Count == 0)
        {
            _textureArray = new Texture2DArray(1, 1, 1, TextureFormat.RGBA32, false);
            return;
        }
        _textureArray = new Texture2DArray(TextureSize, TextureSize, _textures.Count, TextureFormat.RGBA32, true);
        RenderTexture rt = new RenderTexture(TextureSize, TextureSize, 0, RenderTextureFormat.ARGB32);
        for (int i = 0; i < _textures.Count; i++)
        {
            Graphics.Blit(_textures[i], rt);
            Graphics.CopyTexture(rt, 0, 0, _textureArray, i, 0);
        }
        RenderTexture.active = null;
        rt.Release();
    }

    void ReleaseBuffers()
    {
        _meshObjectBuffer?.Release(); _vertexBuffer?.Release(); _normalBuffer?.Release();
        _tangentBuffer?.Release();
        _uvBuffer?.Release(); _indexBuffer?.Release(); _materialBuffer?.Release();
        _bvhBuffer?.Release(); _lightBuffer?.Release();
    }
}
