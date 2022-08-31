#![allow(dead_code, unused)]
use std::time::Instant;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::{self, GpuFuture};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[];
} buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] *= 12;
}"
    }
}

fn main() {
    // Create application's entry to Vulkan API
    let instance = Instance::new(InstanceCreateInfo::default()).unwrap();

    // List all the physical devices that supports Vulkan
    let physical_device = PhysicalDevice::enumerate(&instance).next().unwrap();

    // Find and select the first queue family (threads group) that supports graphics and compute
    let queue_family = physical_device
        .queue_families()
        .find(|&queue_family| queue_family.supports_compute())
        .unwrap();

    // Create vulkan context from the physical device using the selected queue family.
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();

    // Selecting the first queue from the selected queue family
    let queue = queues.next().unwrap();
    let samples = 2048*1024*64;

    let data_iter = 0..samples;
    let data_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter)
            .expect("failed to create buffer");

    let shader = cs::load(device.clone()).unwrap();

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .unwrap();

    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0, // 0 is the index of our set
            set,
        )
        .dispatch([samples/64, 1, 1])
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    let start = Instant::now();
    future.wait(None).unwrap();
    let finish = Instant::now();
    let gpu_time = (finish-start).as_micros();
    println!("The GPU took {} micro-seconds", gpu_time);

    let data = (0..samples).collect::<Vec<u32>>();

    let start = Instant::now();
    let result = data.iter().map(|&x| x * 12).collect::<Vec<u32>>();
    let finish = Instant::now();
    let cpu_time = (finish-start).as_micros();
    println!("The CPU took {} micro-seconds", cpu_time);
    println!("GPU Speed = x{} CPU Speed", cpu_time/gpu_time);

    // Test the results between the CPU and GPU
    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, result[n]);
    }
}
