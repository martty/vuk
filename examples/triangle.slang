struct VSOutput {
    float4 position : SV_Position;
    float3 color : Color;
};

[shader("vertex")]
VSOutput VSmain(uint vertexID: SV_VertexID) {
  VSOutput output;

  if (vertexID == 0) {
      output.position = float4(0.0, -0.3, 0.0, 1.0);
      output.color = float3(0, 1, 0);
  } else if (vertexID == 1) {
      output.position = float4(-0.3, 0.3, 0.0, 1.0);
      output.color = float3(1, 0, 0);
  } else {
      output.position = float4(0.3, 0.3, 0.0, 1.0);
      output.color = float3(0, 0, 1);
  }

  return output;
}

[shader("fragment")]
float4 PSmain(VSOutput input) : SV_Target {
  return float4(input.color, 1.0);
}
