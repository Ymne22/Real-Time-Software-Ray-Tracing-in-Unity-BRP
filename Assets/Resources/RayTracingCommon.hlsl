// RaytracingCommon.hlsl

#ifndef RAY_TRACING_COMMON_HLSL
#define RAY_TRACING_COMMON_HLSL

struct Ray
{
    float3 origin;
    float3 direction;
};

struct HitInfo
{
    float distance;
    float3 position;
    float3 normal;
    float2 uv;
    int materialIndex;
    float4 tangent;
    float3 bitangent;
};

struct Triangle
{
    float3 v0, v1, v2;
};

struct MaterialData
{
    float3 albedo;
    float3 specular;
    float3 emission;
    float transparency;
    float2 textureScale;
    float2 textureOffset;
    int textureIndex;
    int metallicIndex;
    int normalIndex;
};

struct MeshObject
{
    float4x4 localToWorldMatrix;
    float4x4 worldToLocalMatrix;
    uint indices_offset;
    uint indices_count;
    uint materialIndex;
};

struct BVHNode
{
    float3 aabb_min;
    float3 aabb_max;
    int first_child_or_object_index;
    int object_count;
};

struct Light
{
    float3 position;
    float3 direction;
    float3 color;
    float intensity;
    float range;
    float spotAngle;
    int type;
};

float maxComponent(float3 v)
{
    return max(max(v.x, v.y), v.z);
}

float3 reflect(float3 incident, float3 normal)
{
    return incident - 2.0 * dot(incident, normal) * normal;
}

#endif
