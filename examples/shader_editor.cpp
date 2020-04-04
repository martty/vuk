#include "example_runner.hpp"
#include "example_runner.hpp"
#include "TextEditor.h"
#include <fstream>
#include <stb_image.h>
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include "vush.hpp"
#include <ImGuiFileBrowser.h>

TextEditor editor;
static const char* fileToEdit = "../../examples/test.vush";

float angle = 0.f;
auto box = util::generate_cube();

struct program_parameters {
	std::unordered_map<uint32_t, std::vector<char>> buffer;
	std::unordered_map<uint32_t, vuk::ImageView> ivs;
};

struct push_connection {
	std::vector<std::string> name;

	template<class T>
	bool push(vuk::Program refl, program_parameters& params, T data) {
		auto nq = name;
		for (auto& [index, set] : refl.sets) {
			for (auto& un : set.uniform_buffers) {
				nq = name;
				if (nq.front() == un.name) {
					nq.erase(nq.begin());
				} else {
					continue;
				}
				for (auto& m : un.members) {
					if (nq.front() == m.name) {
						nq.erase(nq.begin());
						if (nq.empty()) {
							auto& buf = params.buffer.at(un.binding);
							*(T*)(buf.data() + m.offset) = data;
							return true;
						}
					} else {
						continue;
					}
				}
			}

			if constexpr (std::is_same_v<T, vuk::ImageView>) {
				for (auto& s : set.samplers) {
					if (nq.size() == 1 && nq.front() == s.name) {
						params.ivs.at(s.binding) = data;
					}
				}
			}
		}
		return false;
	}
};

struct transform {
	glm::vec3 position;
	glm::quat orientation = glm::quat(1, 0, 0, 0);
	glm::vec3 scale = glm::vec3(1);
	bool invert = false;

	push_connection connection;

	glm::mat4 to_local() {
		glm::mat4 m = glm::mat4_cast(orientation);
		m[0] *= scale.x;
		m[1] *= scale.y;
		m[2] *= scale.z;
		m[3][0] = position[0];
		m[3][1] = position[1];
		m[3][2] = position[2];
		if (invert)
			return glm::inverse(m);
		else
			return m;
	}
};
#define VOOSH_PAYLOAD_TYPE_CONNECTION_PTR "voosh_payload_connection_ptr"

struct texture {
	vuk::Texture handle;
	std::string filename;

	push_connection connection;
};

static std::vector<texture> textures;

struct projection {
	float fovy = glm::radians(60.f);
	float aspect = 1.f;
	float near_plane = 0.1f;
	float far_plane = 100.f;

	push_connection connection;
	glm::mat4 to_mat() {
		return glm::perspective(fovy, aspect, near_plane, far_plane);
	}
};

std::string slurp(const std::string& path);

std::string stage_to_extension(stage_entry::type as) {
	switch (as) {
	case stage_entry::type::eVertex: return "vert";
	case stage_entry::type::eFragment: return "frag";
	default: assert(0); return "";
	}
}

void recompile(vuk::PerThreadContext& ptc, const std::string& src) {
	auto result = parse_generate(src, fileToEdit);
	for (const auto& [aspect, pa] : result.aspects) {
		for (auto& ps : pa.shaders) {
			auto dst = std::string(fileToEdit) + "." + aspect + "." + stage_to_extension(ps.stage);
			std::ofstream f(dst);
			if (f) {
				f << ps.source;
			}
			ptc.ctx.invalidate_shadermodule_and_pipelines(dst);
		}
	}
}

imgui_addons::ImGuiFileBrowser file_dialog; // As a class member or globally
void load_texture(vuk::PerThreadContext& ptc, const std::string& path) {
	// Use STBI to load the image
	int x, y, chans;
	auto image = stbi_load(path.c_str(), &x, &y, &chans, 4);
	auto [tex, _] = ptc.create_texture(vk::Format::eR8G8B8A8Srgb, vk::Extent3D(x, y, 1), image);
	stbi_image_free(image);

	texture t;
	t.handle = std::move(tex);
	t.filename = path;
	textures.emplace_back(std::move(t));
	ptc.wait_all_transfers();
}


vuk::ExampleRunner::ExampleRunner() {
	vkb::InstanceBuilder builder;
	builder
		.request_validation_layers()
		.set_debug_callback([](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT messageType,
			const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
			void* pUserData) -> VkBool32 {
				auto ms = vkb::to_string_message_severity(messageSeverity);
				auto mt = vkb::to_string_message_type(messageType);
				printf("[%s: %s](user defined)\n%s\n", ms, mt, pCallbackData->pMessage);
				return VK_FALSE;
			})
		.set_app_name("vuk_example")
				.set_engine_name("vuk")
				.require_api_version(1, 1, 0)
				.set_app_version(0, 1, 0);
			auto inst_ret = builder.build();
			if (!inst_ret.has_value()) {
				// error
			}
			vkbinstance = inst_ret.value();
			auto instance = vkbinstance.instance;
			vkb::PhysicalDeviceSelector selector{ vkbinstance };
			window = create_window_glfw("Voosh", false);
			surface = create_surface_glfw(vkbinstance.instance, window);
			selector.set_surface(surface)
				.set_minimum_version(1, 0);
			auto phys_ret = selector.select();
			if (!phys_ret.has_value()) {
				// error
			}
			vkb::PhysicalDevice vkbphysical_device = phys_ret.value();
			physical_device = vkbphysical_device.physical_device;

			vkb::DeviceBuilder device_builder{ vkbphysical_device };
			auto dev_ret = device_builder.build();
			if (!dev_ret.has_value()) {
				// error
			}
			vkbdevice = dev_ret.value();
			graphics_queue = vkbdevice.get_queue(vkb::QueueType::graphics).value();
			device = vkbdevice.device;

			context.emplace(instance, device, physical_device, graphics_queue);

			swapchain = context->add_swapchain(util::make_swapchain(vkbdevice));

			///////////////////////////////////////////////////////////////////////
	// TEXT EDITOR SAMPLE
			auto lang = TextEditor::LanguageDefinition::GLSL();

			// set your own known preprocessor symbols...
			static const char* ppnames[] = { "NULL", "PM_REMOVE",
				"ZeroMemory", "DXGI_SWAP_EFFECT_DISCARD", "D3D_FEATURE_LEVEL", "D3D_DRIVER_TYPE_HARDWARE", "WINAPI","D3D11_SDK_VERSION", "assert" };
			// ... and their corresponding values
			static const char* ppvalues[] = {
				"#define NULL ((void*)0)",
				"#define PM_REMOVE (0x0001)",
				"Microsoft's own memory zapper function\n(which is a macro actually)\nvoid ZeroMemory(\n\t[in] PVOID  Destination,\n\t[in] SIZE_T Length\n); ",
				"enum DXGI_SWAP_EFFECT::DXGI_SWAP_EFFECT_DISCARD = 0",
				"enum D3D_FEATURE_LEVEL",
				"enum D3D_DRIVER_TYPE::D3D_DRIVER_TYPE_HARDWARE  = ( D3D_DRIVER_TYPE_UNKNOWN + 1 )",
				"#define WINAPI __stdcall",
				"#define D3D11_SDK_VERSION (7)",
				" #define assert(expression) (void)(                                                  \n"
				"    (!!(expression)) ||                                                              \n"
				"    (_wassert(_CRT_WIDE(#expression), _CRT_WIDE(__FILE__), (unsigned)(__LINE__)), 0) \n"
				" )"
			};

			for (int i = 0; i < sizeof(ppnames) / sizeof(ppnames[0]); ++i) {
				TextEditor::Identifier id;
				id.mDeclaration = ppvalues[i];
				lang.mPreprocIdentifiers.insert(std::make_pair(std::string(ppnames[i]), id));
			}

			// set your own identifiers
			static const char* identifiers[] = {
				"HWND", "HRESULT", "LPRESULT","D3D11_RENDER_TARGET_VIEW_DESC", "DXGI_SWAP_CHAIN_DESC","MSG","LRESULT","WPARAM", "LPARAM","UINT","LPVOID",
				"ID3D11Device", "ID3D11DeviceContext", "ID3D11Buffer", "ID3D11Buffer", "ID3D10Blob", "ID3D11VertexShader", "ID3D11InputLayout", "ID3D11Buffer",
				"ID3D10Blob", "ID3D11PixelShader", "ID3D11SamplerState", "ID3D11ShaderResourceView", "ID3D11RasterizerState", "ID3D11BlendState", "ID3D11DepthStencilState",
				"IDXGISwapChain", "ID3D11RenderTargetView", "ID3D11Texture2D", "TextEditor" };
			static const char* idecls[] =
			{
				"typedef HWND_* HWND", "typedef long HRESULT", "typedef long* LPRESULT", "struct D3D11_RENDER_TARGET_VIEW_DESC", "struct DXGI_SWAP_CHAIN_DESC",
				"typedef tagMSG MSG\n * Message structure","typedef LONG_PTR LRESULT","WPARAM", "LPARAM","UINT","LPVOID",
				"ID3D11Device", "ID3D11DeviceContext", "ID3D11Buffer", "ID3D11Buffer", "ID3D10Blob", "ID3D11VertexShader", "ID3D11InputLayout", "ID3D11Buffer",
				"ID3D10Blob", "ID3D11PixelShader", "ID3D11SamplerState", "ID3D11ShaderResourceView", "ID3D11RasterizerState", "ID3D11BlendState", "ID3D11DepthStencilState",
				"IDXGISwapChain", "ID3D11RenderTargetView", "ID3D11Texture2D", "class TextEditor" };
			for (int i = 0; i < sizeof(identifiers) / sizeof(identifiers[0]); ++i) {
				TextEditor::Identifier id;
				id.mDeclaration = std::string(idecls[i]);
				lang.mIdentifiers.insert(std::make_pair(std::string(identifiers[i]), id));
			}
			editor.SetLanguageDefinition(lang);
			editor.SetShowWhitespaces(false);

			/*
			// error markers
			TextEditor::ErrorMarkers markers;
			markers.insert(std::make_pair<int, std::string>(6, "Example error here:\nInclude file not found: \"TextEditor.h\""));
			markers.insert(std::make_pair<int, std::string>(41, "Another example error"));
			editor.SetErrorMarkers(markers);

			// "breakpoint" markers
			TextEditor::Breakpoints bpts;
			bpts.insert(24);
			bpts.insert(11);
			editor.SetBreakpoints(bpts);
			*/

			add_rules(json::parse(slurp("../../vush/builtin_cfg.json")));
			{
				std::ifstream t(fileToEdit);
				if (t) {
					std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
					editor.SetText(str);

					vuk::PipelineCreateInfo pci;
					auto result = parse_generate(str, fileToEdit);
					for (const auto& [aspect, pa] : result.aspects) {
						for (auto& ps : pa.shaders) {
							auto dst = std::string(fileToEdit) + "." + aspect + "." + stage_to_extension(ps.stage);
							std::ofstream f(dst);
							if (f) {
								f << ps.source;
							}

							pci.shaders.push_back(dst);
						}
					}
					context->create_named_pipeline("sut", pci);
				}
			}
		auto ifc = context->begin();
		auto ptc = ifc.begin();

		load_texture(ptc, "../../examples/doge.png");
}

void vuk::ExampleRunner::render() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		auto ifc = context->begin();
		auto ptc = ifc.begin();

		auto cpos = editor.GetCursorPosition();
		ImGui::Begin("Shader Editor", nullptr, ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_MenuBar);
		ImGui::SetWindowSize(ImVec2(400, 400), ImGuiCond_FirstUseEver);
		if (ImGui::BeginMenuBar()) {
			if (ImGui::BeginMenu("File")) {
				if (ImGui::MenuItem("Save")) {
					auto textToSave = editor.GetText();
					std::ofstream t(fileToEdit);
					if (t) {
						t << textToSave;
					}
					t.close();
					recompile(ptc, textToSave);
				}
				if (ImGui::MenuItem("Quit", "Alt-F4"))
					break;
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Edit")) {
				bool ro = editor.IsReadOnly();
				if (ImGui::MenuItem("Read-only mode", nullptr, &ro))
					editor.SetReadOnly(ro);
				ImGui::Separator();

				if (ImGui::MenuItem("Undo", "ALT-Backspace", nullptr, !ro && editor.CanUndo()))
					editor.Undo();
				if (ImGui::MenuItem("Redo", "Ctrl-Y", nullptr, !ro && editor.CanRedo()))
					editor.Redo();

				ImGui::Separator();

				if (ImGui::MenuItem("Copy", "Ctrl-C", nullptr, editor.HasSelection()))
					editor.Copy();
				if (ImGui::MenuItem("Cut", "Ctrl-X", nullptr, !ro && editor.HasSelection()))
					editor.Cut();
				if (ImGui::MenuItem("Delete", "Del", nullptr, !ro && editor.HasSelection()))
					editor.Delete();
				if (ImGui::MenuItem("Paste", "Ctrl-V", nullptr, !ro && ImGui::GetClipboardText() != nullptr))
					editor.Paste();

				ImGui::Separator();

				if (ImGui::MenuItem("Select all", nullptr, nullptr))
					editor.SetSelection(TextEditor::Coordinates(), TextEditor::Coordinates(editor.GetTotalLines(), 0));

				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("View")) {
				bool show_ws = editor.IsShowingWhitespaces();
				if (ImGui::MenuItem("Show whitespace", nullptr, &show_ws)) {
					editor.SetShowWhitespaces(show_ws);
				}
				if (ImGui::MenuItem("Dark palette"))
					editor.SetPalette(TextEditor::GetDarkPalette());
				if (ImGui::MenuItem("Light palette"))
					editor.SetPalette(TextEditor::GetLightPalette());
				if (ImGui::MenuItem("Retro blue palette"))
					editor.SetPalette(TextEditor::GetRetroBluePalette());
				ImGui::EndMenu();
			}
			ImGui::EndMenuBar();
		}

		ImGui::Text("%4d/%-4d %3d lines  | %s | %s | %s | %s", cpos.mLine + 1, cpos.mColumn + 1, editor.GetTotalLines(),
			editor.IsOverwrite() ? "Ovr" : "Ins",
			editor.CanUndo() ? "*" : " ",
			editor.GetLanguageDefinition().mName.c_str(), fileToEdit);

		editor.Render("TextEditor");
		ImGui::End();

		ImGui::Begin("Parameters");
		auto refl = ptc.get_pipeline_reflection_info(ptc.ctx.get_named_pipeline("sut"));
		if (ImGui::CollapsingHeader("Attributes")) {
			for (auto& att : refl.attributes) {
				ImGui::Button(att.name.c_str()); ImGui::SameLine();
			}
		ImGui::NewLine();
		}
		// 1 buffer per binding
		// TODO: multiple set support
		static program_parameters program_params;
		// init
		for (auto& [set_index, set] : refl.sets) {
			for (auto& u : set.uniform_buffers) {
				auto it = program_params.buffer.find(u.binding);
				if (it == program_params.buffer.end()) {
					auto& b = program_params.buffer[u.binding];
					b.resize(u.size, 0);
					for (auto& m : u.members) {
						switch (m.type) {
						case vuk::Program::Type::evec3:
							new (b.data() + m.offset) float[3]{ 1,1,1 }; break;
						case vuk::Program::Type::emat4:
							glm::mat4 id(1.f);
							new (b.data() + m.offset) float[16]();
							::memcpy(b.data() + m.offset, &id[0], sizeof(float) * 16); break;
						}
					}
				}
			}

			for (auto& s : set.samplers) {
				program_params.ivs.emplace(s.binding, *textures.front().handle.view);
			}
		}

		if (ImGui::CollapsingHeader("Bindings", ImGuiTreeNodeFlags_DefaultOpen)) {
			for (auto& [set_index, set] : refl.sets) {
				for (auto& u : set.uniform_buffers) {
					auto& b = program_params.buffer[u.binding];
					for (auto& m : u.members) {
						if (m.type == vuk::Program::Type::estruct) {
							if (ImGui::CollapsingHeader(m.name.c_str())) {
								for (auto& mm : m.members) {
								}
							}
						} else {
							switch (m.type) {
							case vuk::Program::Type::evec3:
								ImGui::DragFloat3(m.name.c_str(), (float*)(b.data() + m.offset), 0.01f); break;
							case vuk::Program::Type::emat4:
								ImGui::Button(m.name.c_str()); 
								if (ImGui::BeginDragDropTarget()) {
									if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(VOOSH_PAYLOAD_TYPE_CONNECTION_PTR)) {
										(*(push_connection**)payload->Data)->name = { u.name, m.name };
									}
										
									ImGui::EndDragDropTarget();
								}
								break;
							}
						}
					}
				}
				for (auto& s : set.samplers) {
					ImGui::PushID("...");
					ImGui::Image(&ptc.make_sampled_image(program_params.ivs[s.binding], {}), ImVec2(100.f, 100.f));
					if (ImGui::BeginDragDropTarget()) {
						if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(VOOSH_PAYLOAD_TYPE_CONNECTION_PTR)) {
							(*(push_connection**)payload->Data)->name = { s.name };
						}
						ImGui::EndDragDropTarget();
					}
					ImGui::PopID();
				}
			}
		}
		ImGui::End();
		bool optx = false;
		{
			ImGui::Begin("Textures");
			ImGui::Columns(3, "textures", true);
			ImGui::SetColumnWidth(0, 30.f);
			if (ImGui::Button("+")) {
				optx = true;
			}
			ImGui::NextColumn();
			ImGui::SetColumnWidth(1, 114.f);
			ImGui::Text("Preview");
			ImGui::NextColumn();
			ImGui::Text("Filename");
			ImGui::NextColumn();
			ImGui::Separator();

			size_t i = 0;
			for (auto& tex : textures) {
				ImGui::PushID(i);
				// push data
				if (!tex.connection.name.empty())
					tex.connection.push(refl, program_params, *tex.handle.view);

				if (tex.connection.name.empty())
					ImGui::Button("O");
				else
					if (ImGui::Button("0")) tex.connection.name.clear();
				if (ImGui::BeginDragDropSource()) {
					auto ptrt = &tex.connection;
					ImGui::SetDragDropPayload(VOOSH_PAYLOAD_TYPE_CONNECTION_PTR, &ptrt, sizeof(ptrt));
					ImGui::Image(&ptc.make_sampled_image(*tex.handle.view, {}), ImVec2(50.f, 50.f));
					ImGui::EndDragDropSource();
				}

				ImGui::NextColumn();
				ImGui::SetNextItemWidth(ImGui::GetColumnWidth() - 14);
				ImGui::Image(&ptc.make_sampled_image(*tex.handle.view, {}), ImVec2(100.f, 100.f));
				ImGui::NextColumn();
				ImGui::SetNextItemWidth(ImGui::GetColumnWidth() - 14);
				ImGui::Text("%s", tex.filename.c_str());
				ImGui::NextColumn();
				ImGui::Separator();
				i++;
				ImGui::PopID();
			}
			ImGui::Columns(1);
			ImGui::End();
		}
		if(optx) ImGui::OpenPopup("Add Texture");
		if (file_dialog.showFileDialog("Add Texture", imgui_addons::ImGuiFileBrowser::DialogMode::OPEN, ImVec2(700, 310), ".png,.jpg,.bmp,.tga,*.*")) {
			load_texture(ptc, file_dialog.selected_path);
		}

		static std::vector<transform> tfs = { {}, {{}, glm::quat(1,0,0,0), glm::vec3(1.f), true} };
		static std::vector<projection> projections = { {} };

		ImGui::Begin("Transforms");
		ImGui::Columns(5, "instances", true);
		ImGui::SetColumnWidth(0, 30.f);
		if (ImGui::Button("+"))
			tfs.push_back({});
		ImGui::NextColumn();
		ImGui::Text("Invert");
		ImGui::NextColumn();
		ImGui::Text("Position");
		ImGui::NextColumn();
		ImGui::Text("Orientation");
		ImGui::NextColumn();
		ImGui::Text("Scale");
		ImGui::NextColumn();
		ImGui::Separator();

		size_t i = 0;
		for (auto& tf : tfs) {
			ImGui::PushID(i);
			// push data
			if (!tf.connection.name.empty())
				tf.connection.push(refl, program_params, tf.to_local());

			if (tf.connection.name.empty())
				ImGui::Button("O##t");
			else
				if (ImGui::Button("0##t")) tf.connection.name.clear();
			if (ImGui::BeginDragDropSource()) {
				auto ptrt = &tf.connection;
				ImGui::SetDragDropPayload(VOOSH_PAYLOAD_TYPE_CONNECTION_PTR, &ptrt, sizeof(ptrt));
				ImGui::Text("Transform for (%d)", i);
				ImGui::EndDragDropSource();
			}

			ImGui::NextColumn();
			ImGui::Checkbox("##inv", &tf.invert);
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(ImGui::GetColumnWidth() - 14);
			ImGui::DragFloat3("##pos", &tf.position.x, 0.01f);
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(ImGui::GetColumnWidth() - 14);
			ImGui::DragFloat4("##ori", &tf.orientation[0], 0.01f);
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(ImGui::GetColumnWidth() - 14);
			ImGui::DragFloat3("##sca", &tf.scale.x, 0.1f);
			ImGui::NextColumn();
			ImGui::Separator();
			i++;
			ImGui::PopID();
		}
		ImGui::Separator();
		ImGui::Separator();
		if (ImGui::Button("+##proj"))
			projections.push_back({});
		ImGui::NextColumn();
		ImGui::Text("FovY");
		ImGui::NextColumn();
		ImGui::Text("Aspect");
		ImGui::NextColumn();
		ImGui::Text("Near");
		ImGui::NextColumn();
		ImGui::Text("Far");
		ImGui::NextColumn();
		ImGui::Separator();

		i = 0;
		for (auto& p : projections) {
			ImGui::PushID(i);
			// push data
			if (!p.connection.name.empty())
				p.connection.push(refl, program_params, p.to_mat());

			if (p.connection.name.empty())
				ImGui::Button("O##p");
			else
				if (ImGui::Button("0##p")) p.connection.name.clear();
			if (ImGui::BeginDragDropSource()) {
				auto ptrt = &p.connection;
				ImGui::SetDragDropPayload(VOOSH_PAYLOAD_TYPE_CONNECTION_PTR, &ptrt, sizeof(ptrt));
				ImGui::Text("Projection (%d)", i);
				ImGui::EndDragDropSource();
			}

			ImGui::NextColumn();
			ImGui::SetNextItemWidth(ImGui::GetColumnWidth() - 14);
			ImGui::DragFloat("##fovy", &p.fovy, 0.01f);
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(ImGui::GetColumnWidth() - 14);
			ImGui::DragFloat("##asp", &p.aspect, 0.01f);
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(ImGui::GetColumnWidth() - 14);
			ImGui::DragFloat("##near", &p.near_plane, 0.1f);
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(ImGui::GetColumnWidth() - 14);
			ImGui::DragFloat("##far", &p.far_plane, 0.1f);
			ImGui::NextColumn();
			ImGui::Separator();
			i++;
			ImGui::PopID();
		}

		ImGui::Columns(1);
		ImGui::End();

		ImGui::ShowDemoWindow();
		// We set up the cube data, same as in example 02_cube
		auto [bverts, stub1] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eVertexBuffer, gsl::span(&box.first[0], box.first.size()));
		auto verts = std::move(bverts);
		auto [binds, stub2] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eIndexBuffer, gsl::span(&box.second[0], box.second.size()));
		auto inds = std::move(binds);
		struct VP {
			glm::mat4 view;
			glm::mat4 proj;
		} vp;
		vp.view = glm::lookAt(glm::vec3(0, 1.5, 3.5), glm::vec3(0), glm::vec3(0, 1, 0));
		vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);

		auto [buboVP, stub3] = ptc.create_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vk::BufferUsageFlagBits::eUniformBuffer, gsl::span(&vp, 1));
		auto uboVP = buboVP;

		//new(buffer[1].data()) glm::mat4(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
		auto [user, stubx] = ptc.create_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vk::BufferUsageFlagBits::eUniformBuffer, gsl::span(program_params.buffer[1].data(), program_params.buffer[1].size()));
		ptc.wait_all_transfers();

		vuk::RenderGraph rg;

		// Set up the pass to draw the textured cube, with a color and a depth attachment
		rg.add_pass({
			.resources = {"04_texture_final"_image(vuk::eColorWrite), "04_texture_depth"_image(vuk::eDepthStencilRW)},
			.execute = [&](vuk::CommandBuffer& command_buffer) {
				command_buffer
				  .set_viewport(0, vuk::Area::Framebuffer{})
				  .set_scissor(0, vuk::Area::Framebuffer{})
				  .bind_vertex_buffer(0, verts, 0, vuk::Packed{vk::Format::eR32G32B32Sfloat, vuk::Ignore{offsetof(util::Vertex, uv_coordinates) - sizeof(util::Vertex::position)}, vk::Format::eR32G32Sfloat})
				  .bind_index_buffer(inds, vk::IndexType::eUint32);
				for (auto& [binding, iv] : program_params.ivs) {
					command_buffer.bind_sampled_image(0, binding, iv, vk::SamplerCreateInfo{});
				  }
				  command_buffer
					.bind_pipeline("sut")
					.bind_uniform_buffer(0, 0, uboVP)
					.bind_uniform_buffer(0, 1, user);
				  command_buffer
					.draw_indexed(box.second.size(), 1, 0, 0, 0);
				  }
			}
		);

		angle += 180.f * ImGui::GetIO().DeltaTime;

		rg.mark_attachment_internal("04_texture_depth", vk::Format::eD32Sfloat, vuk::Extent2D::Framebuffer{}, vuk::Samples::Framebuffer{}, vuk::ClearDepthStencil{ 1.0f, 0 });
		rg.mark_attachment_internal("04_texture_final", vk::Format::eR8G8B8A8Srgb, vk::Extent2D(300, 300), vuk::Samples::e1, vuk::ClearColor(0.1f, 0.2f, 0.3f, 1.f));
		
		ImGui::Begin("Preview");
		ImGui::Image(&ptc.make_sampled_image("04_texture_final", imgui_data.font_sci), ImVec2(200, 200));
		ImGui::End();

		ImGui::Render();
		std::string attachment_name = "voosh_final";
		rg.add_pass(util::ImGui_ImplVuk_Render(ptc, attachment_name, attachment_name, imgui_data, ImGui::GetDrawData()));
		rg.build();
		rg.bind_attachment_to_swapchain(attachment_name, swapchain, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		rg.build(ptc);
		execute_submit_and_present_to_one(ptc, rg, swapchain);
	}
}

int main() {
	vuk::ExampleRunner::get_runner().setup();
	vuk::ExampleRunner::get_runner().render();
	textures.clear();
	vuk::ExampleRunner::get_runner().cleanup();
}