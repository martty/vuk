struct input_from_vertex
{
    float3 color : COLOR;
};

float4 main(in input_from_vertex IN) : SV_TARGET
{
    return float4(IN.color, 1.0);
};