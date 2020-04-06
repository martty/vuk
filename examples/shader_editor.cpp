#include "example_runner.hpp"
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
#define TINYGLTF_NO_INCLUDE_JSON
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

TextEditor editor;
static const char* fileToEdit = "../../examples/test.vush";
tinygltf::TinyGLTF loader;

float angle = 0.f;
auto box = util::generate_cube();

struct vattr {
	vuk::Buffer buffer;
	vk::Format format;
	size_t offset;
	size_t stride;
};

struct vinds {
	vuk::Buffer buffer;
	vk::IndexType type;
	size_t count;
};

struct program_parameters {
	std::unordered_map<uint32_t, std::vector<char>> buffer;
	std::unordered_map<uint32_t, vuk::ImageView> ivs;
	std::unordered_map<uint32_t, vattr> vattrs;
	std::optional<vinds> indices;
};

struct push_connection {
	std::vector<std::string> name;

	template<class T>
	bool push_to_member(vuk::Program::Member& m, uint32_t binding, program_parameters& params, std::vector<std::string> nq, T& data) {
		if (nq.empty()) {
			auto& buf = params.buffer.at(binding);
			*(T*)(buf.data() + m.offset) = data;
			return true;
		} else if (m.type == vuk::Program::Type::estruct) {
			auto nqd = nq;
			for (auto& mm : m.members) {
				if (nqd.front() == mm.name) {
					nqd.erase(nqd.begin());
					if (push_to_member(mm, binding, params, nqd, data)) return true;
				}
			}
		}
		return false;
	}

	template<class T>
	bool push(vuk::Program refl, program_parameters& params, T data) {
		auto nq = name;
		for (auto& [index, set] : refl.sets) {
			if constexpr (!std::is_same_v<T, vuk::ImageView>) {
				for (auto& un : set.uniform_buffers) {
					nq = name;
					if (nq.front() == un.name) {
						nq.erase(nq.begin());
					} else {
						continue;
					}
					for (auto& m : un.members) {
						auto nqd = nq;
						if (nqd.front() == m.name) {
							nqd.erase(nqd.begin());
							if (push_to_member(m, un.binding, params, nqd, data)) return true;
						}
					}
				}
			}

			if constexpr (std::is_same_v<T, vuk::ImageView>) {
				for (auto& s : set.samplers) {
					if (nq.size() == 1 && nq.front() == s.name) {
						params.ivs.at(s.binding) = data;
						return true;
					}
				}
			}
		}
		if constexpr (std::is_same_v<T, vattr>) {
			for (auto& s : refl.attributes) {
				if (nq.size() == 1 && nq.front() == s.name) {
					params.vattrs[s.location] = data;
					return true;
				}
			}
		}

		if constexpr (std::is_same_v<T, vinds>) {
			if (nq.size() == 1 && nq.front() == "indices") {
				params.indices = data;
				return true;
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
			return glm::lookAt(position, glm::vec3(0.f), scale);
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

struct st {
	vuk::Texture handle;
	vuk::SampledImage si;
};

static std::unordered_map<std::string, st> voosh_res;

struct buffer_source {
	vuk::Unique<vuk::Buffer> buffer;
	vk::Format format;
	size_t stride;
	std::string attr_name;
	std::string filename;

	push_connection connection;
};

/*struct submesh {
	std::unordered_map<std::string, vuk::Unique<vuk::Buffer>> attributes;
	vuk::Unique<vuk::Buffer> indices;
};

static submesh sm;*/
static std::unordered_map<std::string, std::vector<buffer_source>> buf_sources;

void load_model(vuk::PerThreadContext& ptc, const std::string& file) {
	std::string err;
	std::string warn;

	tinygltf::Model model;
	bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, file.c_str());
	//bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, argv[1]); // for binary glTF(.glb)

	if (!warn.empty()) {
		printf("Warn: %s\n", warn.c_str());
	}

	if (!err.empty()) {
		printf("Err: %s\n", err.c_str());
	}

	if (!ret) {
		printf("Failed to parse glTF\n");
	}

	for (auto& m : model.meshes) {
		for (auto& p : m.primitives) {
			//p.mode -> PTS/LINES/TRIS
			{
				auto& acc = model.accessors[p.indices];
				auto& buffer_view = model.bufferViews[acc.bufferView];
				auto& buffer = model.buffers[buffer_view.buffer];
				auto data = gsl::span(buffer.data.data() + buffer_view.byteOffset + acc.byteOffset, buffer_view.byteLength);
				buffer_source bf;
				bf.attr_name = "index";
				auto name = m.name + "/" + model.materials[p.material].name;
				bf.filename = file;
				assert(acc.type == TINYGLTF_TYPE_SCALAR);
				bf.stride = acc.ByteStride(buffer_view);
				bf.buffer = ptc.create_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eIndexBuffer, data).first;

				buf_sources[name].push_back(std::move(bf));
			}

			for (auto& [name, index] : p.attributes) {
				auto& acc = model.accessors[index];
				auto& buffer_view = model.bufferViews[acc.bufferView];
				auto& buffer = model.buffers[buffer_view.buffer];
				auto data = gsl::span(buffer.data.data() + buffer_view.byteOffset + acc.byteOffset, buffer_view.byteLength);
		
				auto stride = acc.ByteStride(buffer_view);
				buffer_source bf;
				bf.attr_name = name;
				bf.stride = stride;
				if (acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
					switch(acc.type){
					case TINYGLTF_TYPE_VEC3: bf.format = vk::Format::eR32G32B32Sfloat; break;
					case TINYGLTF_TYPE_VEC2: bf.format = vk::Format::eR32G32Sfloat; break;
					}
				}
				auto name = m.name + "/" + model.materials[p.material].name;
				bf.filename = file;

				bf.buffer = ptc.create_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eVertexBuffer, data).first;
				buf_sources[name].push_back(std::move(bf));
			}
		}
	}
	ptc.wait_all_transfers();

}

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

void init_members(vuk::Program::Member& m, std::vector<char>& b) {
	switch (m.type) {
	case vuk::Program::Type::evec3:
		new (b.data() + m.offset) float[3]{ 1,1,1 }; break;
	case vuk::Program::Type::emat4:
		glm::mat4 id(1.f);
		new (b.data() + m.offset) float[16]();
		::memcpy(b.data() + m.offset, &id[0], sizeof(float) * 16); break;
	case vuk::Program::Type::estruct:
		for (auto& mm : m.members) {
			init_members(mm, b);
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

void load_res_texture(vuk::PerThreadContext& ptc, const std::string& name, const std::string& path) {
	// Use STBI to load the image
	int x, y, chans;
	auto image = stbi_load(path.c_str(), &x, &y, &chans, 4);
	auto [tex, _] = ptc.create_texture(vk::Format::eR8G8B8A8Srgb, vk::Extent3D(x, y, 1), image);
	stbi_image_free(image);

	vk::SamplerCreateInfo sci;
	sci.minFilter = sci.magFilter = vk::Filter::eLinear;
	voosh_res.emplace(name, st{ std::move(tex), ptc.make_sampled_image(*tex.view, sci) });
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

		load_model(ptc, "../../examples/suzy.glb");
		load_texture(ptc, "../../examples/doge.png");
		load_res_texture(ptc, "mat4x4", "../../examples/mat4x4.jpg");
}

void droppable(std::vector<std::string>& name_stack, std::string name) {
	if (ImGui::BeginDragDropTarget()) {
		if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(VOOSH_PAYLOAD_TYPE_CONNECTION_PTR)) {
			(*(push_connection**)payload->Data)->name = name_stack;
			(*(push_connection**)payload->Data)->name.push_back(name);
		}
		ImGui::EndDragDropTarget();
	}
}

void parameter_ui(vuk::Program::Member& m, std::vector<char>& b, std::vector<std::string> name_stack) {
	switch (m.type) {
	case vuk::Program::Type::evec3:
		ImGui::PushID(m.name.c_str());
		ImGui::SetNextItemWidth(ImGui::GetColumnWidth() - 14);
		ImGui::DragFloat3("", (float*)(b.data() + m.offset), 0.01f); 
		ImGui::PopID();
		break;
	case vuk::Program::Type::emat4:
		ImGui::ImageButton(&voosh_res.at("mat4x4").si, ImVec2(50, 50));
		break;
	case vuk::Program::Type::estruct:
		bool open = ImGui::TreeNodeEx(m.name.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
		ImGui::NextColumn();
		ImGui::Text(m.name.c_str());
		ImGui::NextColumn();

		if(open) {
			name_stack.push_back(m.name);
			for (auto& mm : m.members) {
				parameter_ui(mm, b, name_stack);
			}
			name_stack.pop_back();
			ImGui::TreePop();
		}
		return;

	}
	droppable(name_stack, m.name);
	ImGui::NextColumn();
	ImGui::Text(m.name.c_str());
	ImGui::NextColumn();
}

#include <regex>

std::regex error_regex(R"((\S*):(\d+): (.+?)\n)");

// 1 buffer per binding
// TODO: multiple set support
static program_parameters program_params;

void save(vuk::PerThreadContext& ptc) {
	auto textToSave = editor.GetText();
	std::ofstream t(fileToEdit);
	if (t) {
		t << textToSave;
	}
	t.close();
	recompile(ptc, textToSave);
	program_params.buffer.clear();
	program_params.ivs.clear();

}

#include <imgui.h>

void vuk::ExampleRunner::render() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		auto ifc = context->begin();
		auto ptc = ifc.begin();

		auto cpos = editor.GetCursorPosition();
		ImGui::Begin("Shader Editor", nullptr, ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_MenuBar);
		ImGuiIO& io = ImGui::GetIO();
		auto shift = io.KeyShift;
		auto ctrl = io.ConfigMacOSXBehaviors ? io.KeySuper : io.KeyCtrl;
		auto alt = io.ConfigMacOSXBehaviors ? io.KeyCtrl : io.KeyAlt;

		if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
			if (ctrl && !shift && !alt && ImGui::IsKeyPressed(GLFW_KEY_S))
				save(ptc);
		}

		ImGui::SetWindowSize(ImVec2(400, 400), ImGuiCond_FirstUseEver);
		if (ImGui::BeginMenuBar()) {
			if (ImGui::BeginMenu("File")) {
				if (ImGui::MenuItem("Save", "Ctrl-S")) {
					save(ptc);
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

		vuk::Program refl;
		bool will_we_render = true;
		try {
			refl = ptc.get_pipeline_reflection_info(ptc.ctx.get_named_pipeline("sut"));
			// clear error marks here
			editor.SetErrorMarkers({});
		} catch (vuk::ShaderCompilationException& e) {
			auto words_begin = std::sregex_iterator(e.error_message.begin(), e.error_message.end(), error_regex);
			auto words_end = std::sregex_iterator();
			TextEditor::ErrorMarkers markers;
			for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
				std::smatch match = *it;

				// TODO: multiple files
				auto file = match[1].str();
				auto lstr = match[2].str();
				int line = std::stoi(lstr);
				auto message = match[3].str();

				// merge markers for the same line
				auto mit = markers.find(line);
				if (mit == markers.end()) {
					markers.insert(std::pair<int, std::string>(line, message));
				} else {
					mit->second += "\n" + message;
				}
			}
			editor.SetErrorMarkers(markers);
			will_we_render = false;
		}

		ImGui::Begin("Parameters");
		if (ImGui::CollapsingHeader("Attributes", ImGuiTreeNodeFlags_DefaultOpen)) {
			std::vector<std::string> name_stack;
			for (auto& att : refl.attributes) {
				ImGui::Button(att.name.c_str());
				droppable(name_stack, att.name);
			}
			ImGui::Button("Indices");
			droppable(name_stack, "indices");
		ImGui::NewLine();
		}
		// init
		for (auto& [set_index, set] : refl.sets) {
			for (auto& u : set.uniform_buffers) {
				auto it = program_params.buffer.find(u.binding);
				if (it == program_params.buffer.end()) {
					auto& b = program_params.buffer[u.binding];
					b.resize(u.size, 0);
					for (auto& m : u.members) {
						init_members(m, b);
					}
				} 
			}

			for (auto& s : set.samplers) {
				program_params.ivs.emplace(s.binding, *textures.front().handle.view);
			}
		}

		if (ImGui::CollapsingHeader("Bindings", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Columns(2, "textures", true);
			for (auto& [set_index, set] : refl.sets) {
				std::vector<std::string> name_stack;
				for (auto& u : set.uniform_buffers) {
					name_stack.push_back(u.name);
					auto& b = program_params.buffer[u.binding];
					for (auto& m : u.members) {
						parameter_ui(m, b, name_stack);
					}
					name_stack.pop_back();
				}
				for (auto& s : set.samplers) {
					ImGui::Image(&ptc.make_sampled_image(program_params.ivs[s.binding], {}), ImVec2(ImGui::GetColumnWidth(), ImGui::GetColumnWidth() * /*aspect*/ 1.f));
					droppable(name_stack, s.name);
					ImGui::NextColumn();
					ImGui::Text(s.name.c_str());
					ImGui::NextColumn();
				}
			}
			ImGui::Columns(1);
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

		static std::vector<transform> tfs = { {}, {glm::vec3(0, 1.5, 3.5), glm::quat(1,0,0,0), glm::vec3(0, 1.f, 0), true} };
		static std::vector<projection> projections = { {} };

		ImGui::Begin("Transforms");
		ImGui::Columns(5, "instances", true);
		ImGui::SetColumnWidth(0, 30.f);
		if (ImGui::Button("+"))
			tfs.push_back({});
		ImGui::NextColumn();
		ImGui::Text("as lookAt()");
		ImGui::NextColumn();
		ImGui::Text("Position");
		ImGui::NextColumn();
		ImGui::Text("Orientation / Not used");
		ImGui::NextColumn();
		ImGui::Text("Scale / Up");
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
			if (ImGui::DragFloat4("##ori", &tf.orientation[0], 0.01f)) {
				tf.orientation = glm::normalize(tf.orientation);
			}
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
		{
			ImGui::Begin("Meshes");
			ImGui::Columns(3, "meshes", true);
			if (ImGui::Button("+")) {
			//	tfs.push_back({});
			}
			ImGui::NextColumn();
			ImGui::Text("Mesh");
			ImGui::NextColumn();
			ImGui::Text("Filename");
			ImGui::NextColumn();
			ImGui::Separator();

			size_t i = 0;
			for (auto& [name, bufs] : buf_sources) {
				ImGui::PushID(name.c_str());
				for (auto& bf : bufs) {
					ImGui::PushID(bf.attr_name.c_str());
					// push data
					if (!bf.connection.name.empty() && bf.connection.name.front() != "indices") {
						vattr v;
						v.buffer = *bf.buffer;
						v.format = bf.format;
						v.offset = 0;
						v.stride = bf.stride;
						bf.connection.push(refl, program_params, v);
					} else if (!bf.connection.name.empty() && bf.connection.name.front() == "indices") {
						vinds v;
						v.buffer = *bf.buffer;
						v.count = v.buffer.size / bf.stride;
						if (bf.stride == sizeof(uint16_t))
							v.type = vk::IndexType::eUint16;
						else
							assert(0);
						bf.connection.push(refl, program_params, v);
					}

					if (bf.connection.name.empty())
						ImGui::Button("O##b");
					else
						if (ImGui::Button("0##b")) bf.connection.name.clear();
					if (ImGui::BeginDragDropSource()) {
						auto ptrt = &bf.connection;
						ImGui::SetDragDropPayload(VOOSH_PAYLOAD_TYPE_CONNECTION_PTR, &ptrt, sizeof(ptrt));
						ImGui::Text("Attribute for (%d)", i);
						ImGui::EndDragDropSource();
					}
					ImGui::SameLine();
					ImGui::Text(bf.attr_name.c_str());
					ImGui::PopID();
				}

				ImGui::NextColumn();
				ImGui::Text("%s", name.c_str());
				ImGui::NextColumn();
				ImGui::Text("%s", bufs[0].filename.c_str());
				ImGui::NextColumn();
				ImGui::Separator();
				i++;
				ImGui::PopID();
			}
			ImGui::End();
		}
		//ImGui::ShowDemoWindow();
		vuk::RenderGraph rg;

		// Set up the pass to draw the textured cube, with a color and a depth attachment
		rg.add_pass({
			.resources = {"04_texture_final"_image(vuk::eColorWrite), "04_texture_depth"_image(vuk::eDepthStencilRW)},
			.execute = [&](vuk::CommandBuffer& command_buffer) {
				command_buffer
				  .set_viewport(0, vuk::Area::Framebuffer{})
				  .set_scissor(0, vuk::Area::Framebuffer{});

				for (auto& [location, vattr] : program_params.vattrs) {
					command_buffer.bind_vertex_buffer(location, vattr.buffer, location, vuk::Packed{vattr.format});
				}
				if (program_params.indices) {
					command_buffer.bind_index_buffer(program_params.indices->buffer, vk::IndexType::eUint16);
				}
				for (auto& [binding, iv] : program_params.ivs) {
					command_buffer.bind_sampled_image(0, binding, iv, vk::SamplerCreateInfo{});
				}
				command_buffer.bind_pipeline("sut");
				for (auto& [binding, buffer] : program_params.buffer) {
					void * dst = command_buffer._map_scratch_uniform_binding(0, binding, program_params.buffer[binding].size() * sizeof(char));
					memcpy(dst, program_params.buffer[binding].data(), program_params.buffer[binding].size() * sizeof(char));
				}
				if(will_we_render && program_params.indices)
					command_buffer.draw_indexed(program_params.indices->count, 1, 0, 0, 0);
				}
			});

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

		try {
			execute_submit_and_present_to_one(ptc, rg, swapchain);
		} catch (vuk::ShaderCompilationException& e) {
			std::cout << e.what();
		}
	}
}

int main() {
	vuk::ExampleRunner::get_runner().setup();
	vuk::ExampleRunner::get_runner().render();
	textures.clear();
	voosh_res.clear();
	buf_sources.clear();
	vuk::ExampleRunner::get_runner().cleanup();
}