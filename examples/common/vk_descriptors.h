
#pragma once

// ---------------------------------------------------------------------------
// vk_descriptors.h  –  Descriptor system builders
//
// Three lightweight builder classes that eliminate the ~50-line descriptor
// boilerplate repeated in every example:
//
//   DescriptorLayoutBuilder  – build a VkDescriptorSetLayout
//   DescriptorPoolBuilder    – build a VkDescriptorPool
//   DescriptorWriter         – write bindings into an already-allocated set
//
// Typical usage per example:
//
//   // 1. Layout
//   descriptorSetLayout = DescriptorLayoutBuilder()
//       .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         VK_SHADER_STAGE_FRAGMENT_BIT)
//       .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
//       .build(device);
//
//   // 2. Pool  (count = MAX_FRAMES_IN_FLIGHT per type used)
//   descriptorPool = DescriptorPoolBuilder()
//       .addSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAX_FRAMES_IN_FLIGHT)
//       .addSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES_IN_FLIGHT)
//       .build(device, MAX_FRAMES_IN_FLIGHT);
//
//   // 3. Allocate sets (standard Vulkan – no helper needed)
//   VkDescriptorSetAllocateInfo ai{};
//   ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
//   ai.descriptorPool     = descriptorPool;
//   ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
//   std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
//   ai.pSetLayouts        = layouts.data();
//   descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
//   vkAllocateDescriptorSets(device, &ai, descriptorSets.data());
//
//   // 4. Write each set
//   for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
//       DescriptorWriter()
//           .writeBuffer(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
//                        uniformBuffers[i], 0, sizeof(MyUBO))
//           .writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
//                       textureImageView, textureSampler,
//                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
//           .update(device, descriptorSets[i]);
//
// ---------------------------------------------------------------------------

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// DescriptorLayoutBuilder
// ---------------------------------------------------------------------------

class DescriptorLayoutBuilder
{
public:
	// Add one binding to the layout.
	// count = 1 covers the common case; use > 1 for arrays of descriptors.
	DescriptorLayoutBuilder& addBinding(uint32_t binding,
		VkDescriptorType type, VkShaderStageFlags stages, uint32_t count = 1)
	{
		VkDescriptorSetLayoutBinding b{};
		b.binding         = binding;
		b.descriptorType  = type;
		b.descriptorCount = count;
		b.stageFlags      = stages;
		bindings_.push_back(b);
		return *this;
	}

	// Build and return a VkDescriptorSetLayout.  Caller must destroy it.
	VkDescriptorSetLayout build(VkDevice device) const
	{
		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = static_cast<uint32_t>(bindings_.size());
		ci.pBindings    = bindings_.empty() ? nullptr : bindings_.data();

		VkDescriptorSetLayout layout;
		if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout) != VK_SUCCESS)
			throw std::runtime_error("DescriptorLayoutBuilder: failed to create descriptor set layout!");
		return layout;
	}

private:
	std::vector<VkDescriptorSetLayoutBinding> bindings_;
};

// ---------------------------------------------------------------------------
// DescriptorPoolBuilder
// ---------------------------------------------------------------------------

class DescriptorPoolBuilder
{
public:
	// Add a pool size entry.  count is the total number of descriptors of this
	// type across ALL sets allocated from the pool.
	// Typical value: count = MAX_FRAMES_IN_FLIGHT (one descriptor per frame).
	DescriptorPoolBuilder& addSize(VkDescriptorType type, uint32_t count)
	{
		VkDescriptorPoolSize s{};
		s.type            = type;
		s.descriptorCount = count;
		sizes_.push_back(s);
		return *this;
	}

	// Build and return a VkDescriptorPool.  Caller must destroy it.
	// maxSets = maximum number of descriptor sets that can be allocated.
	VkDescriptorPool build(VkDevice device, uint32_t maxSets) const
	{
		VkDescriptorPoolCreateInfo ci{};
		ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		ci.maxSets       = maxSets;
		ci.poolSizeCount = static_cast<uint32_t>(sizes_.size());
		ci.pPoolSizes    = sizes_.empty() ? nullptr : sizes_.data();

		VkDescriptorPool pool;
		if (vkCreateDescriptorPool(device, &ci, nullptr, &pool) != VK_SUCCESS)
			throw std::runtime_error("DescriptorPoolBuilder: failed to create descriptor pool!");
		return pool;
	}

private:
	std::vector<VkDescriptorPoolSize> sizes_;
};

// ---------------------------------------------------------------------------
// DescriptorWriter
// ---------------------------------------------------------------------------

class DescriptorWriter
{
public:
	// Write a uniform / storage buffer descriptor.
	DescriptorWriter& writeBuffer(uint32_t binding, VkDescriptorType type,
		VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range)
	{
		bufferInfos_.push_back({buffer, offset, range});

		VkWriteDescriptorSet w{};
		w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w.dstBinding      = binding;
		w.descriptorCount = 1;
		w.descriptorType  = type;
		// pBufferInfo pointer is filled in update() after all infos are collected.
		w.pBufferInfo     = nullptr; // patched in update()
		writes_.push_back(w);
		bufferWriteIdx_.push_back(static_cast<uint32_t>(bufferInfos_.size() - 1));
		imageWriteIdx_ .push_back(UINT32_MAX);
		return *this;
	}

	// Write a combined image sampler or storage image descriptor.
	DescriptorWriter& writeImage(uint32_t binding, VkDescriptorType type,
		VkImageView view, VkSampler sampler, VkImageLayout layout)
	{
		VkDescriptorImageInfo info{};
		info.imageView   = view;
		info.sampler     = sampler;
		info.imageLayout = layout;
		imageInfos_.push_back(info);

		VkWriteDescriptorSet w{};
		w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w.dstBinding      = binding;
		w.descriptorCount = 1;
		w.descriptorType  = type;
		w.pImageInfo      = nullptr; // patched in update()
		writes_.push_back(w);
		bufferWriteIdx_.push_back(UINT32_MAX);
		imageWriteIdx_ .push_back(static_cast<uint32_t>(imageInfos_.size() - 1));
		return *this;
	}

	// Patch pointers and call vkUpdateDescriptorSets for the given set.
	void update(VkDevice device, VkDescriptorSet set)
	{
		for (size_t i = 0; i < writes_.size(); ++i)
		{
			writes_[i].dstSet = set;
			if (bufferWriteIdx_[i] != UINT32_MAX)
				writes_[i].pBufferInfo = &bufferInfos_[bufferWriteIdx_[i]];
			else
				writes_[i].pImageInfo  = &imageInfos_ [imageWriteIdx_ [i]];
		}
		vkUpdateDescriptorSets(device,
			static_cast<uint32_t>(writes_.size()), writes_.data(), 0, nullptr);
	}

private:
	std::vector<VkWriteDescriptorSet>  writes_;
	std::vector<VkDescriptorBufferInfo> bufferInfos_;
	std::vector<VkDescriptorImageInfo>  imageInfos_;
	std::vector<uint32_t>               bufferWriteIdx_;
	std::vector<uint32_t>               imageWriteIdx_;
};
