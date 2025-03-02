#pragma once

#include <imgui.h>

#include <vuk/Types.hpp>
#include <vuk/Value.hpp>

namespace vuk::extra {
	struct ImGuiData {
		Unique<Image> font_image;
		Unique<ImageView> font_image_view;
		SamplerCreateInfo font_sci;
		ImageAttachment font_ia;
		std::vector<Value<SampledImage>> sampled_images;

		/// @brief Add a sampled image that can be used in ImGui (use vuk::combine_image_sampler to create)
		/// @param image SampledImage to be added
		/// @return Value to be passed to ImGui::Image
		ImTextureID add_sampled_image(Value<SampledImage> sampled_image);

		/// @brief Add an image that can be used in ImGui, with the default sampler
		/// @param image Image to be added
		/// @return Value to be passed to ImGui::Image
		ImTextureID add_image(Value<ImageAttachment> image);
	};

	/// @brief Initialize ImGui integration with Vuk
	/// @param allocator Allocator to use for ImGui data
	/// @return Initialized ImGui data
	ImGuiData ImGui_ImplVuk_Init(Allocator& allocator);

	/// @brief Render ImGui into given target
	/// @param allocator Allocator to use to allocate additional resources
	/// @param target ImageAttachment to render ImGui into
	/// @param data ImGui data to use for rendering
	Value<ImageAttachment> ImGui_ImplVuk_Render(Allocator& allocator, Value<ImageAttachment> target, ImGuiData& data);

} // namespace vuk::extra