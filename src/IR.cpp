#include <vuk/IR.hpp>

#include "vuk/runtime/vk/VkSwapchain.hpp"

namespace vuk {
	thread_local std::shared_ptr<IRModule> current_module = std::make_shared<IRModule>();

	auto Type::stripped(std::shared_ptr<Type> const t) -> std::shared_ptr<Type> {
		switch (t->kind) {
		case Type::IMBUED_TY:
			return stripped(*t->imbued.T);
		case Type::ALIASED_TY:
			return stripped(*t->aliased.T);
		default:
			return t;
		}
	}

	auto Type::extract(std::shared_ptr<Type> const t, size_t const index) -> std::shared_ptr<Type> {
		assert(t->kind == COMPOSITE_TY);
		assert(index < t->composite.types.size());
		return t->composite.types[index];
	}

	auto Type::hash_integer(size_t const width) -> Hash {
		Hash v = (Hash)Type::INTEGER_TY;
		hash_combine_direct(v, width);
		return v;
	}

	auto Type::hash(Type const* const t) -> Hash {
		Hash v = (Hash)t->kind;
		switch (t->kind) {
		case Type::VOID_TY:
			return v;
		case Type::IMBUED_TY:
			hash_combine_direct(v, Type::hash(t->imbued.T->get()));
			hash_combine_direct(v, (uint32_t)t->imbued.access);
			return v;
		case Type::ALIASED_TY:
			hash_combine_direct(v, Type::hash(t->aliased.T->get()));
			hash_combine_direct(v, (uint32_t)t->aliased.ref_idx);
			return v;
		case Type::MEMORY_TY:
			hash_combine_direct(v, (uint32_t)t->size);
			return v;
		case Type::INTEGER_TY:
			hash_combine_direct(v, t->integer.width);
			return v;
		case Type::ARRAY_TY:
			hash_combine_direct(v, Type::hash(t->array.T->get()));
			hash_combine_direct(v, (uint32_t)t->array.count);
			return v;
		case Type::UNION_TY:
		case Type::COMPOSITE_TY: {
			for (size_t i = 0; i < t->composite.types.size(); i++) {
				hash_combine_direct(v, Type::hash(t->composite.types[i].get()));
			}
			hash_combine_direct(v, (uint32_t)t->composite.tag);
			return v;
		}
		case Type::OPAQUE_FN_TY:
			hash_combine_direct(v, (uintptr_t)t->opaque_fn.hash_code >> 32);
			hash_combine_direct(v, (uintptr_t)t->opaque_fn.hash_code & 0xffffffff);
			return v;
		case Type::SHADER_FN_TY:
			hash_combine_direct(v, (uintptr_t)t->shader_fn.shader >> 32);
			hash_combine_direct(v, (uintptr_t)t->shader_fn.shader & 0xffffffff);
			return v;
		}
		assert(0);
		return v;
	}

	auto Type::to_sv(Access const acc) -> std::string_view {
		// TODO: handle multiple flags
		switch (acc) {
		case eNone:
			return "None";
		case eClear:
			return "Clear";
		case eColorWrite:
			return "ColorW";
		case eColorRead:
			return "ColorR";
		case eColorRW:
			return "ColorRW";
		case eDepthStencilRead:
			return "DSRead";
		case eDepthStencilWrite:
			return "DSWrite";
		case eDepthStencilRW:
			return "DSRW";
		case eVertexSampled:
			return "VtxS";
		case eVertexRead:
			return "VtxR";
		case eAttributeRead:
			return "AttrR";
		case eIndexRead:
			return "IdxR";
		case eIndirectRead:
			return "IndirR";
		case eFragmentSampled:
			return "FragS";
		case eFragmentRead:
			return "FragR";
		case eFragmentWrite:
			return "FragW";
		case eFragmentRW:
			return "FragRW";
		case eTransferRead:
			return "XferR";
		case eTransferWrite:
			return "XferW";
		case eTransferRW:
			return "XferRW";
		case eComputeRead:
			return "CompR";
		case eComputeWrite:
			return "CompW";
		case eComputeRW:
			return "CompRW";
		case eComputeSampled:
			return "CompS";
		case eRayTracingRead:
			return "RTR";
		case eRayTracingWrite:
			return "RTW";
		case eRayTracingRW:
			return "RTRW";
		case eRayTracingSampled:
			return "RTS";
		case eAccelerationStructureBuildRead:
			return "ASBuildR";
		case eAccelerationStructureBuildWrite:
			return "ASBuildW";
		case eAccelerationStructureBuildRW:
			return "ASBuildRW";
		case eHostRead:
			return "HostR";
		case eHostWrite:
			return "HostW";
		case eHostRW:
			return "HostRW";
		case eMemoryRead:
			return "MemR";
		case eMemoryWrite:
			return "MemW";
		case eMemoryRW:
			return "MemRW";
		default:
			return "<multiple>";
		}
	}

	auto Type::to_string(Type* t) -> std::string {
		switch (t->kind) {
		case Type::VOID_TY:
			return "void";
		case Type::IMBUED_TY:
			return to_string(t->imbued.T->get()) + std::string(":") + std::string(to_sv(t->imbued.access));
		case Type::ALIASED_TY:
			return to_string(t->aliased.T->get()) + std::string("@") + std::to_string(t->aliased.ref_idx);
		case Type::MEMORY_TY:
			return "mem";
		case Type::INTEGER_TY:
			return t->integer.width == 32 ? "i32" : "i64";
		case Type::ARRAY_TY:
			return to_string(t->array.T->get()) + "[" + std::to_string(t->array.count) + "]";
		case Type::COMPOSITE_TY:
			if (!t->debug_info.name.empty()) {
				return std::string(t->debug_info.name);
			}
			return "composite:" + std::to_string(t->composite.tag);
		case Type::UNION_TY:
			if (!t->debug_info.name.empty()) {
				return std::string(t->debug_info.name);
			}
			return "union:" + std::to_string(t->composite.tag);
		case Type::OPAQUE_FN_TY:
			return "ofn";
		case Type::SHADER_FN_TY:
			return "sfn";
		default:
			assert(0);
			return "?";
		}
	}

	auto Node::kind_to_sv(Node::Kind const kind) -> std::string_view {
		switch (kind) {
		case Node::PLACEHOLDER:
			return "placeholder";
		case Node::CONSTANT:
			return "constant";
		case Node::IMPORT:
			return "import";
		case Node::CONSTRUCT:
			return "construct";
		case Node::ACQUIRE_NEXT_IMAGE:
			return "acquire_next_image";
		case Node::CALL:
			return "call";
		case Node::MATH_BINARY:
			return "math_b";
		case Node::SLICE:
			return "slice";
		case Node::CONVERGE:
			return "converge";
		case Node::CLEAR:
			return "clear";
		case Node::CAST:
			return "cast";
		case Node::GARBAGE:
			return "garbage";
		case Node::RELEASE:
			return "release";
		case Node::ACQUIRE:
			return "acquire";
		case Node::USE:
			return "use";
		case Node::SET:
			return "set";
		case Node::LOGICAL_COPY:
			return "lcopy";
		case Node::COMPILE_PIPELINE:
			return "compile_pipeline";
		}
		assert(0);
		return "";
	}

	std::shared_ptr<Type> IRModule::Types::make_void_ty() {
		auto t = new Type{ .kind = Type::VOID_TY };
		return emplace_type(std::shared_ptr<Type>(t));
	}

	std::shared_ptr<Type> IRModule::Types::make_imbued_ty(std::shared_ptr<Type> ty, Access access) {
		auto t = new Type{ .kind = Type::IMBUED_TY, .size = ty->size, .imbued = { .access = access } };
		t->imbued.T = &t->child_types.emplace_back(ty);
		return emplace_type(std::shared_ptr<Type>(t));
	}

	std::shared_ptr<Type> IRModule::Types::make_aliased_ty(std::shared_ptr<Type> ty, size_t ref_idx) {
		auto t = new Type{ .kind = Type::ALIASED_TY, .size = ty->size, .aliased = { .ref_idx = ref_idx } };
		t->imbued.T = &t->child_types.emplace_back(ty);
		return emplace_type(std::shared_ptr<Type>(t));
	}

	std::shared_ptr<Type> IRModule::Types::make_array_ty(std::shared_ptr<Type> ty, size_t count) {
		auto t = new Type{ .kind = Type::ARRAY_TY, .size = count * ty->size, .array = { .count = count, .stride = ty->size } };
		t->array.T = &t->child_types.emplace_back(ty);
		return emplace_type(std::shared_ptr<Type>(t));
	}

	std::shared_ptr<Type> IRModule::Types::make_union_ty(std::vector<std::shared_ptr<Type>> types) {
		std::vector<size_t> offsets;
		size_t offset = 0;
		for (auto& t : types) {
			offsets.push_back(offset);
			offset += t->size;
		}
		auto union_type = emplace_type(std::shared_ptr<Type>(
		    new Type{ .kind = Type::UNION_TY, .size = offset, .offsets = offsets, .composite = { .types = types, .tag = union_tag_type_counter++ } }));
		union_type->child_types = std::move(types);
		return union_type;
	}

	std::shared_ptr<Type> IRModule::Types::make_opaque_fn_ty(std::span<std::shared_ptr<Type> const> args,
	                                                         std::span<std::shared_ptr<Type> const> ret_types,
	                                                         DomainFlags execute_on,
	                                                         size_t hash_code,
	                                                         UserCallbackType callback,
	                                                         std::string_view name) {
		auto arg_ptr_ret_ty_ptr = std::vector<std::shared_ptr<Type>>(args.size() + ret_types.size());
		auto it = std::copy(args.begin(), args.end(), arg_ptr_ret_ty_ptr.begin());
		std::copy(ret_types.begin(), ret_types.end(), it);
		auto t = new Type{ .kind = Type::OPAQUE_FN_TY,
			                 .opaque_fn = { .args = std::span{ arg_ptr_ret_ty_ptr.data(), args.size() },
			                                .return_types = std::span{ arg_ptr_ret_ty_ptr.data() + args.size(), ret_types.size() },
			                                .hash_code = hash_code,
			                                .execute_on = execute_on.m_mask } };
		t->callback = std::make_unique<UserCallbackType>(std::move(callback));
		t->child_types = std::move(arg_ptr_ret_ty_ptr);
		t->debug_info = allocate_type_debug_info(std::string(name));
		return emplace_type(std::shared_ptr<Type>(t));
	}

	std::shared_ptr<Type> IRModule::Types::make_shader_fn_ty(std::span<std::shared_ptr<Type> const> args,
	                                                         std::span<std::shared_ptr<Type> const> ret_types,
	                                                         DomainFlags execute_on,
	                                                         void* shader,
	                                                         std::string_view name) {
		auto arg_ptr_ret_ty_ptr = std::vector<std::shared_ptr<Type>>(args.size() + ret_types.size());
		auto it = std::copy(args.begin(), args.end(), arg_ptr_ret_ty_ptr.begin());
		std::copy(ret_types.begin(), ret_types.end(), it);
		auto t = new Type{ .kind = Type::SHADER_FN_TY,
			                 .shader_fn = { .shader = shader,
			                                .args = std::span{ arg_ptr_ret_ty_ptr.data(), args.size() },
			                                .return_types = std::span{ arg_ptr_ret_ty_ptr.data() + args.size(), ret_types.size() },
			                                .execute_on = execute_on.m_mask } };
		t->child_types = std::move(arg_ptr_ret_ty_ptr);
		t->debug_info = allocate_type_debug_info(std::string(name));
		return emplace_type(std::shared_ptr<Type>(t));
	}

	std::shared_ptr<Type> IRModule::Types::u64() {
		auto hash = Type::hash_integer(64);
		auto it = type_map.find(hash);
		if (it != type_map.end()) {
			if (auto ty = it->second.lock()) {
				return ty;
			}
		}

		return emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::INTEGER_TY, .size = sizeof(uint64_t), .integer = { .width = 64 } }));
	}

	std::shared_ptr<Type> IRModule::Types::u32() {
		auto hash = Type::hash_integer(32);
		auto it = type_map.find(hash);
		if (it != type_map.end()) {
			if (auto ty = it->second.lock()) {
				return ty;
			}
		}

		return emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::INTEGER_TY, .size = sizeof(uint32_t), .integer = { .width = 32 } }));
	}

	std::shared_ptr<Type> IRModule::Types::memory(size_t size) {
		Type ty{ .kind = Type::MEMORY_TY, .size = size };
		auto it = type_map.find(Type::hash(&ty));
		if (it != type_map.end()) {
			if (auto ty = it->second.lock()) {
				return ty;
			}
		}
		return emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::MEMORY_TY, .size = size }));
	}

	std::shared_ptr<Type> IRModule::Types::get_builtin_image() {
		if (builtin_image) {
			auto it = type_map.find(builtin_image);
			if (it != type_map.end()) {
				if (auto ty = it->second.lock()) {
					return ty;
				}
			}
		}

		auto u32_t = u32();
		auto image_ = std::vector<std::shared_ptr<Type>>{ u32_t, u32_t, u32_t, memory(sizeof(Format)), memory(sizeof(Samples)), u32_t, u32_t, u32_t, u32_t };
		// TODO: crimes
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-offsetof"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
		auto image_offsets = std::vector<size_t>{ offsetof(ImageAttachment, extent) + offsetof(Extent3D, width),
			                                        offsetof(ImageAttachment, extent) + offsetof(Extent3D, height),
			                                        offsetof(ImageAttachment, extent) + offsetof(Extent3D, depth),
			                                        offsetof(ImageAttachment, format),
			                                        offsetof(ImageAttachment, sample_count),
			                                        offsetof(ImageAttachment, base_layer),
			                                        offsetof(ImageAttachment, layer_count),
			                                        offsetof(ImageAttachment, base_level),
			                                        offsetof(ImageAttachment, level_count) };
#pragma GCC diagnostic pop
#pragma clang diagnostic pop
		auto image_type = emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
		                                                               .size = sizeof(ImageAttachment),
		                                                               .debug_info = allocate_type_debug_info("image"),
		                                                               .offsets = image_offsets,
		                                                               .composite = { .types = image_, .tag = 0 } }));
		image_type->child_types = std::move(image_);
		builtin_image = Type::hash(image_type.get());

		return image_type;
	}

	std::shared_ptr<Type> IRModule::Types::get_builtin_buffer() {
		if (builtin_buffer) {
			auto it = type_map.find(builtin_buffer);
			if (it != type_map.end()) {
				if (auto ty = it->second.lock()) {
					return ty;
				}
			}
		}

		auto buffer_ = std::vector<std::shared_ptr<Type>>{ u64() };
		auto buffer_offsets = std::vector<size_t>{ offsetof(Buffer, size) };
		auto buffer_type = emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
		                                                                .size = sizeof(Buffer),
		                                                                .debug_info = allocate_type_debug_info("buffer"),
		                                                                .offsets = buffer_offsets,
		                                                                .composite = { .types = buffer_, .tag = 1 } }));
		buffer_type->child_types = std::move(buffer_);

		builtin_buffer = Type::hash(buffer_type.get());
		return buffer_type;
	}

	std::shared_ptr<Type> IRModule::Types::get_builtin_swapchain() {
		if (builtin_swapchain) {
			auto it = type_map.find(builtin_swapchain);
			if (it != type_map.end()) {
				if (auto ty = it->second.lock()) {
					return ty;
				}
			}
		}
		auto arr_ty = make_array_ty(get_builtin_image(), 16);
		auto swp_ = std::vector<std::shared_ptr<Type>>{ arr_ty };
		auto offsets = std::vector<size_t>{ 0 };

		auto swapchain_type = emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
		                                                                   .size = sizeof(Swapchain*),
		                                                                   .debug_info = allocate_type_debug_info("swapchain"),
		                                                                   .offsets = offsets,
		                                                                   .composite = { .types = swp_, .tag = 2 } }));
		builtin_swapchain = Type::hash(swapchain_type.get());
		return swapchain_type;
	}

	std::shared_ptr<Type> IRModule::Types::get_builtin_sampler() {
		if (builtin_sampler) {
			auto it = type_map.find(builtin_sampler);
			if (it != type_map.end()) {
				if (auto ty = it->second.lock()) {
					return ty;
				}
			}
		}
		auto sampler_type = emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
		                                                                 .size = sizeof(SamplerCreateInfo),
		                                                                 .debug_info = allocate_type_debug_info("sampler"),
		                                                                 .offsets = {},
		                                                                 .composite = { .types = {}, .tag = 3 } }));
		builtin_sampler = Type::hash(sampler_type.get());
		return sampler_type;
	}

	std::shared_ptr<Type> IRModule::Types::get_builtin_sampled_image() {
		if (builtin_sampled_image) {
			auto it = type_map.find(builtin_sampled_image);
			if (it != type_map.end()) {
				if (auto ty = it->second.lock()) {
					return ty;
				}
			}
		}
		auto sampled_image_type = emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
		                                                                       .size = sizeof(SampledImage),
		                                                                       .debug_info = allocate_type_debug_info("sampled_image"),
		                                                                       .offsets = {},
		                                                                       .composite = { .types = {}, .tag = 4 } }));
		builtin_sampled_image = Type::hash(sampled_image_type.get());
		return sampled_image_type;
	}

	std::shared_ptr<Type> IRModule::Types::emplace_type(std::shared_ptr<Type> t) {
		auto unify_type = [&](std::shared_ptr<Type>& t) {
			auto th = Type::hash(t.get());
			auto [v, succ] = type_map.try_emplace(th, t);
			if (succ) {
				t->hash_value = th;
			} else if (!v->second.lock()) {
				type_map[th] = t;
			}
			t->hash_value = th;
		};

		if (t->kind == Type::ALIASED_TY) {
			assert((*t->aliased.T)->kind != Type::ALIASED_TY);
			unify_type(*t->aliased.T);
		} else if (t->kind == Type::IMBUED_TY) {
			unify_type(*t->imbued.T);
		} else if (t->kind == Type::ARRAY_TY) {
			unify_type(*t->array.T);
		} else if (t->kind == Type::COMPOSITE_TY) {
			for (auto& elem_ty : t->composite.types) {
				unify_type(elem_ty);
			}
		}
		unify_type(t);

		return t;
	}

	TypeDebugInfo IRModule::Types::allocate_type_debug_info(std::string name) {
		return TypeDebugInfo{ name };
	}

	void IRModule::Types::collect() {
		for (auto it = type_map.begin(); it != type_map.end();) {
			if (it->second.expired()) {
				it = type_map.erase(it);
			} else {
				++it;
			}
		}
	}

	void IRModule::Types::destroy(Type* t, void* v) {
		if (t->hash_value == builtin_buffer) {
			std::destroy_at<Buffer>((Buffer*)v);
		} else if (t->hash_value == builtin_image) {
			std::destroy_at<ImageAttachment>((ImageAttachment*)v);
		} else if (t->hash_value == builtin_sampled_image) {
			std::destroy_at<SampledImage>((SampledImage*)v);
		} else if (t->hash_value == builtin_sampler) {
			std::destroy_at<SamplerCreateInfo>((SamplerCreateInfo*)v);
		} else if (t->hash_value == builtin_swapchain) {
			std::destroy_at<Swapchain*>((Swapchain**)v);
		} else if (t->kind == Type::INTEGER_TY) {
			// nothing to do
		} else if (t->kind == Type::MEMORY_TY) {
			// nothing to do
		} else if (t->kind == Type::IMBUED_TY) {
			destroy(t->imbued.T->get(), v);
		} else if (t->kind == Type::ALIASED_TY) {
			destroy(t->aliased.T->get(), v);
		} else if (t->kind == Type::ARRAY_TY || t->kind == Type::UNION_TY) {
			// currently arrays and unions don't own their values
			/* auto cv = (char*)v;
			for (auto i = 0; i < t->array.count; i++) {
			  destroy(t->array.T->get(), cv);
			  cv += t->array.stride;
			}*/
		} else {
			assert(0);
		}
		delete[] (std::byte*)v;
	}

	Node* IRModule::emplace_op(Node v) {
		v.index = module_id << 32 | node_counter++;
		return &*op_arena.emplace(std::move(v));
	}

	void IRModule::name_output(Ref ref, std::string_view name) {
		auto node = ref.node;
		if (!node->debug_info) {
			node->debug_info = new NodeDebugInfo;
		}
		auto& names = ref.node->debug_info->result_names;
		if (names.size() <= ref.index) {
			names.resize(ref.index + 1);
		}
		names[ref.index] = name;
	}

	void IRModule::set_source_location(Node* node, SourceLocationAtFrame loc) {
		if (!node->debug_info) {
			node->debug_info = new NodeDebugInfo;
		}
		auto p = &loc;
		size_t cnt = 0;
		do {
			cnt++;
			p = p->parent;
		} while (p != nullptr);
		if (node->debug_info->trace.data()) {
			delete[] node->debug_info->trace.data();
		}
		node->debug_info->trace = std::span(new vuk::source_location[cnt], cnt);
		p = &loc;
		cnt = 0;
		do {
			node->debug_info->trace[cnt] = p->location;
			cnt++;
			p = p->parent;
		} while (p != nullptr);
	}

	std::optional<plf::colony<Node>::iterator> IRModule::destroy_node(Node* node) {
		delete node->rel_acq;
		switch (node->kind) {
		case Node::CONSTANT: {
			if (node->constant.owned) {
				delete[] (char*)node->constant.value;
			}
			break;
		}
		case Node::ACQUIRE: {
			for (size_t i = 0; i < node->acquire.values.size(); i++) {
				auto& v = node->acquire.values[i];
				types.destroy(Type::stripped(node->type[i]).get(), v);
			}
			delete[] node->acquire.values.data();
			break;
		}
		default: // nothing extra to be done here
			break;
		}
		delete[] node->type.data();
		if (node->generic_node.arg_count == (uint8_t)~0u) {
			delete[] node->variable_node.args.data();
		}
		if (node->scheduling_info)
			delete node->scheduling_info;
		if (node->debug_info) {
			if (node->debug_info->trace.size() > 0) {
				delete[] node->debug_info->trace.data();
			}

			delete node->debug_info;
		}

		auto it = op_arena.get_iterator(node);
		if (it != op_arena.end()) {
#ifdef VUK_GARBAGE_SAN
			node->kind = Node::GARBAGE;
			node->generic_node.arg_count = 0;
			node->type = {};
#else
			return op_arena.erase(it);
#endif
		} else {
			node->kind = Node::GARBAGE;
			node->generic_node.arg_count = 0;
			node->type = {};
		}
		return {};
	}

	// OPS

	Ref IRModule::make_constant(std::shared_ptr<Type> type, void* value) {
		std::shared_ptr<Type>* ty = new std::shared_ptr<Type>[1]{ type };
		auto value_ptr = new char[type->size];
		memcpy(value_ptr, value, type->size);
		return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = value_ptr, .owned = true } }));
	}

	void IRModule::set_value(Ref ref, size_t index, Ref value) {
		emplace_op(Node{ .kind = Node::SET, .set = { .dst = ref, .value = value, .index = index } });
	}

	Ref IRModule::make_declare_image(ImageAttachment value) {
		auto ptr = new (new char[sizeof(ImageAttachment)])
		    ImageAttachment(value); /* rest extent_x extent_y extent_z format samples base_layer layer_count base_level level_count */
		auto args_ptr = new Ref[10];
		auto mem_ty = new std::shared_ptr<Type>[1]{ types.memory(sizeof(ImageAttachment)) };
		args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = ptr, .owned = true } }));
		if (value.extent.width > 0) {
			args_ptr[1] = make_constant(&ptr->extent.width);
		} else {
			args_ptr[1] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1]{ types.u32() }, 1 } }));
		}
		if (value.extent.height > 0) {
			args_ptr[2] = make_constant(&ptr->extent.height);
		} else {
			args_ptr[2] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1]{ types.u32() }, 1 } }));
		}
		if (value.extent.depth > 0) {
			args_ptr[3] = make_constant(&ptr->extent.depth);
		} else {
			args_ptr[3] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1]{ types.u32() }, 1 } }));
		}
		if (value.format != Format::eUndefined) {
			args_ptr[4] = make_constant(&ptr->format);
		} else {
			args_ptr[4] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1]{ types.memory(sizeof(Format)) }, 1 } }));
		}
		if (value.sample_count != Samples::eInfer) {
			args_ptr[5] = make_constant(&ptr->sample_count);
		} else {
			args_ptr[5] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1]{ types.memory(sizeof(Samples)) }, 1 } }));
		}
		if (value.base_layer != VK_REMAINING_ARRAY_LAYERS) {
			args_ptr[6] = make_constant(&ptr->base_layer);
		} else {
			args_ptr[6] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1]{ types.u32() }, 1 } }));
		}
		if (value.layer_count != VK_REMAINING_ARRAY_LAYERS) {
			args_ptr[7] = make_constant(&ptr->layer_count);
		} else {
			args_ptr[7] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1]{ types.u32() }, 1 } }));
		}
		if (value.base_level != VK_REMAINING_MIP_LEVELS) {
			args_ptr[8] = make_constant(&ptr->base_level);
		} else {
			args_ptr[8] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1]{ types.u32() }, 1 } }));
		}
		if (value.level_count != VK_REMAINING_MIP_LEVELS) {
			args_ptr[9] = make_constant(&ptr->level_count);
		} else {
			args_ptr[9] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1]{ types.u32() }, 1 } }));
		}

		return first(emplace_op(Node{ .kind = Node::CONSTRUCT,
		                              .type = std::span{ new std::shared_ptr<Type>[1]{ types.get_builtin_image() }, 1 },
		                              .construct = { .args = std::span(args_ptr, 10) } }));
	}

	Ref IRModule::make_declare_buffer(Buffer value) {
		auto buf_ptr = new (new char[sizeof(Buffer)]) Buffer(value); /* rest size */
		auto args_ptr = new Ref[2];
		auto mem_ty = new std::shared_ptr<Type>[1]{ types.memory(sizeof(Buffer)) };
		args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = buf_ptr, .owned = true } }));
		if (value.size != ~(0ULL)) {
			args_ptr[1] = make_constant(&buf_ptr->size);
		} else {
			args_ptr[1] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1]{ types.u64() }, 1 } }));
		}

		return first(emplace_op(Node{ .kind = Node::CONSTRUCT,
		                              .type = std::span{ new std::shared_ptr<Type>[1]{ types.get_builtin_buffer() }, 1 },
		                              .construct = { .args = std::span(args_ptr, 2) } }));
	}

	Ref IRModule::make_declare_array(std::shared_ptr<Type> type, std::span<Ref> args) {
		auto arr_ty = new std::shared_ptr<Type>[1]{ types.make_array_ty(type, args.size()) };
		auto args_ptr = new Ref[args.size() + 1];
		auto mem_ty = new std::shared_ptr<Type>[1]{ types.memory(0) };
		args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = nullptr } }));
		std::copy(args.begin(), args.end(), args_ptr + 1);
		return first(emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ arr_ty, 1 }, .construct = { .args = std::span(args_ptr, args.size() + 1) } }));
	}

	Ref IRModule::make_declare_union(std::span<Ref> args) {
		std::vector<std::shared_ptr<Type>> child_types;
		for (auto& arg : args) {
			child_types.push_back(Type::stripped(arg.type()));
		}
		auto union_ty = new std::shared_ptr<Type>[1]{ types.make_union_ty(std::move(child_types)) };
		auto args_ptr = new Ref[args.size() + 1];
		auto mem_ty = new std::shared_ptr<Type>[1]{ types.memory(0) };
		args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = nullptr } }));
		std::copy(args.begin(), args.end(), args_ptr + 1);
		return first(emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ union_ty, 1 }, .construct = { .args = std::span(args_ptr, args.size() + 1) } }));
	}

	Ref IRModule::make_declare_swapchain(Swapchain& bundle) {
		auto swpptr = new (new char[sizeof(Swapchain*)]) void*(&bundle);
		auto args_ptr = new Ref[2];
		auto mem_ty = new std::shared_ptr<Type>[1]{ types.memory(sizeof(Swapchain*)) };
		args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = swpptr, .owned = true } }));
		std::vector<Ref> imgs;
		for (size_t i = 0; i < bundle.images.size(); i++) {
			imgs.push_back(make_declare_image(bundle.images[i]));
		}
		args_ptr[1] = make_declare_array(types.get_builtin_image(), imgs);
		return first(emplace_op(Node{ .kind = Node::CONSTRUCT,
		                              .type = std::span{ new std::shared_ptr<Type>[1]{ types.get_builtin_swapchain() }, 1 },
		                              .construct = { .args = std::span(args_ptr, 2) } }));
	}

	Ref IRModule::make_sampled_image(Ref image, Ref sampler) {
		auto args_ptr = new Ref[3]{ make_constant(0), image, sampler };
		return first(emplace_op(Node{ .kind = Node::CONSTRUCT,
		                              .type = std::span{ new std::shared_ptr<Type>[1]{ types.get_builtin_sampled_image() }, 1 },
		                              .construct = { .args = std::span(args_ptr, 3) } }));
	}

	Ref IRModule::make_extract(Ref composite, Ref index) {
		auto stripped = Type::stripped(composite.type());
		assert(stripped->kind == Type::ARRAY_TY);
		auto ty = new std::shared_ptr<Type>[3]{ *stripped->array.T, stripped, stripped };
		return first(emplace_op(Node{
		    .kind = Node::SLICE, .type = std::span{ ty, 3 }, .slice = { .src = composite, .start = index, .count = make_constant<uint64_t>(1), .axis = 0 } }));
	}

	Ref IRModule::make_extract(Ref composite, uint64_t index) {
		auto ty = new std::shared_ptr<Type>[3];
		auto stripped = Type::stripped(composite.type());
		uint8_t axis = 0;
		if (stripped->kind == Type::ARRAY_TY) {
			ty[0] = *stripped->array.T;
		} else if (stripped->kind == Type::COMPOSITE_TY || stripped->kind == Type::UNION_TY) {
			ty[0] = stripped->composite.types[index];
			axis = Node::NamedAxis::FIELD;
		} else {
			assert(0);
		}
		ty[1] = ty[2] = stripped;
		return first(emplace_op(Node{ .kind = Node::SLICE,
		                              .type = std::span{ ty, 3 },
		                              .slice = { .src = composite, .start = make_constant<uint64_t>(index), .count = make_constant<uint64_t>(1), .axis = axis } }));
	}

	Ref IRModule::make_slice(Ref src, uint8_t axis, Ref base, Ref count) {
		auto stripped = Type::stripped(src.type());
		auto ty = new std::shared_ptr<Type>[3]{ stripped, stripped, stripped };
		return first(emplace_op(Node{ .kind = Node::SLICE, .type = std::span{ ty, 3 }, .slice = { .src = src, .start = base, .count = count, .axis = axis } }));
	}

	Ref IRModule::make_slice(std::shared_ptr<Type> type_ex, Ref src, uint8_t axis, Ref base, Ref count) {
		auto ty = new std::shared_ptr<Type>[3]{ Type::stripped(type_ex), Type::stripped(src.type()), Type::stripped(src.type()) };
		return first(emplace_op(Node{ .kind = Node::SLICE, .type = std::span{ ty, 3 }, .slice = { .src = src, .start = base, .count = count, .axis = axis } }));
	}

	Ref IRModule::make_converge(std::shared_ptr<Type> type, std::span<Ref> deps) {
		auto stripped = Type::stripped(type);
		auto ty = new std::shared_ptr<Type>[1]{ stripped };

		auto deps_ptr = new Ref[deps.size()];
		std::copy(deps.begin(), deps.end(), deps_ptr);
		return first(emplace_op(Node{ .kind = Node::CONVERGE, .type = std::span{ ty, 1 }, .converge = { .diverged = std::span{ deps_ptr, deps.size() } } }));
	}

	Ref IRModule::make_use(Ref src, Access acc) {
		auto ty = new std::shared_ptr<Type>[1]{ src.type() };
		return first(emplace_op(Node{ .kind = Node::USE, .type = std::span{ ty, 1 }, .use = { .src = src, .access = acc } }));
	}

	Ref IRModule::make_cast(std::shared_ptr<Type> dst_type, Ref src) {
		auto ty = new std::shared_ptr<Type>[1]{ dst_type };
		return first(emplace_op(Node{ .kind = Node::CAST, .type = std::span{ ty, 1 }, .cast = { .src = src } }));
	}

	Ref IRModule::make_acquire_next_image(Ref swapchain) {
		return first(emplace_op(Node{ .kind = Node::ACQUIRE_NEXT_IMAGE,
		                              .type = std::span{ new std::shared_ptr<Type>[1]{ types.get_builtin_image() }, 1 },
		                              .acquire_next_image = { .swapchain = swapchain } }));
	}

	Ref IRModule::make_clear_image(Ref dst, Clear cv) {
		return first(emplace_op(Node{ .kind = Node::CLEAR,
		                              .type = std::span{ new std::shared_ptr<Type>[1]{ types.get_builtin_image() }, 1 },
		                              .clear = { .dst = dst, .cv = new Clear(cv) } }));
	}

	Ref IRModule::make_declare_fn(std::shared_ptr<Type> const fn_ty) {
		auto ty = new std::shared_ptr<Type>[1]{ fn_ty };
		return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = nullptr } }));
	}

	Ref IRModule::make_release(Ref src, Access dst_access, DomainFlagBits dst_domain) {
		Ref* args_ptr = new Ref[1]{ src };
		auto tys = new std::shared_ptr<Type>[1]{ Type::stripped(src.type()) };
		return first(emplace_op(Node{ .kind = Node::RELEASE,
		                              .type = std::span{ tys, 1 },
		                              .release = { .src = std::span{ args_ptr, 1 }, .dst_access = dst_access, .dst_domain = dst_domain } }));
	}

	Ref IRModule::make_compile_pipeline(Ref src) {
		auto tys = new std::shared_ptr<Type>[1]{ types.memory(sizeof(PipelineBaseInfo*)) };
		return first(emplace_op(Node{ .kind = Node::COMPILE_PIPELINE, .type = std::span{ tys, 1 }, .compile_pipeline = { .src = src } }));
	}

	// MATH

	Ref IRModule::make_math_binary_op(Node::BinOp op, Ref a, Ref b) {
		std::shared_ptr<Type>* tys = new std::shared_ptr<Type>[1]{ a.type() };

		return first(emplace_op(Node{ .kind = Node::MATH_BINARY, .type = std::span{ tys, 1 }, .math_binary = { .a = a, .b = b, .op = op } }));
	}

	ExtNode::ExtNode(Node* node) : node(node) {
		acqrel = new AcquireRelease;
		node->rel_acq = acqrel;
		this->node->held = true;
		source_module = current_module;
	}

	ExtNode::ExtNode(Node* node, std::vector<std::shared_ptr<ExtNode>> deps) : deps(std::move(deps)), node(node) {
		acqrel = new AcquireRelease;
		node->rel_acq = acqrel;
		this->node->held = true;
		source_module = current_module;
	}

	ExtNode::ExtNode(Node* node, std::shared_ptr<ExtNode> dep) : node(node) {
		acqrel = new AcquireRelease;
		node->rel_acq = acqrel;
		this->node->held = true;
		deps.push_back(std::move(dep));

		source_module = current_module;
	}

	ExtNode::ExtNode(Ref ref, std::shared_ptr<ExtNode> dep, Access access, DomainFlagBits domain) {
		acqrel = new AcquireRelease;
		this->node = current_module->make_release(ref, access, domain).node;
		node->rel_acq = acqrel;
		this->node->held = true;
		deps.push_back(std::move(dep));

		source_module = current_module;
	}

	ExtNode::ExtNode(Node* node, ResourceUse use) : node(node) {
		acqrel = new AcquireRelease;
		acqrel->status = Signal::Status::eHostAvailable;
		acqrel->last_use.resize(1);
		acqrel->last_use[0] = use;

		node->rel_acq = acqrel;
		this->node->held = true;
		source_module = current_module;
	}

	ExtNode::~ExtNode() {
		if (acqrel) {
			node->held = false;
		}
	}

	void ExtNode::mutate(Node* new_node) {
		node->held = false;
		node = new_node;
		new_node->held = true;
	}
} // namespace vuk
