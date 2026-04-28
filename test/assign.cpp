#include <darkside/assign.hpp>
#include <startorch/common.hpp>
#include <startorch/device.hpp>
#include <startorch/memory.hpp>

#include <gtest/gtest.h>

namespace darkside {
TEST(AssignTest, FillDataCPUTest) {
  int size = 10;
  startorch::Device cpu(startorch::CPU, startorch::DEFAULT);

  int *data = (int *)startorch::makeData(sizeof(int) * size, cpu);

  fillData<int>(data, size, 42, cpu);

  for (int i = 0; i < size; i++)
    EXPECT_EQ(data[i], 42);

  startorch::freeData(data, cpu);
}

TEST(AssignTest, FillIncreaseCPUTest) {
  int size = 10;
  startorch::Device cpu(startorch::CPU, startorch::DEFAULT);

  int *data = (int *)startorch::makeData(sizeof(int) * size, cpu);

  fillIncreaseData<int>(data, size, cpu);

  for (int i = 0; i < size; i++)
    EXPECT_EQ(data[i], i);

  startorch::freeData(data, cpu);
}

TEST(AssignTest, FillDecreaseCPUTest) {
  int size = 10;
  startorch::Device cpu(startorch::CPU, startorch::DEFAULT);

  int *data = (int *)startorch::makeData(sizeof(int) * size, cpu);

  fillDecreaseData<int>(data, size, cpu);

  for (int i = 0; i < size; i++)
    EXPECT_EQ(data[i], size - 1 - i);

  startorch::freeData(data, cpu);
}

TEST(AssignTest, FillDataGPUTest) {
  int size = 10;

  startorch::Device cpu(startorch::CPU, startorch::DEFAULT);
  startorch::Device gpu(startorch::GPU, startorch::DEFAULT);

  int *gpu_data = (int *)startorch::makeData(sizeof(int) * size, gpu);

  fillData<int>(gpu_data, size, 7, gpu);

  int res[10];
  startorch::copyData(res, gpu_data, sizeof(int) * size,
                      startorch::DevicePair(gpu, cpu));

  for (int i = 0; i < size; i++)
    EXPECT_EQ(res[i], 7);

  startorch::freeData(gpu_data, gpu);
}

TEST(AssignTest, FillIncreaseGPUTest) {
  int size = 10;

  startorch::Device cpu(startorch::CPU, startorch::DEFAULT);
  startorch::Device gpu(startorch::GPU, startorch::DEFAULT);

  int *gpu_data = (int *)startorch::makeData(sizeof(int) * size, gpu);

  fillIncreaseData<int>(gpu_data, size, gpu);

  int res[10];
  startorch::copyData(res, gpu_data, sizeof(int) * size,
                      startorch::DevicePair(gpu, cpu));

  for (int i = 0; i < size; i++)
    EXPECT_EQ(res[i], i);

  startorch::freeData(gpu_data, gpu);
}

TEST(AssignTest, FillDecreaseGPUTest) {
  int size = 10;

  startorch::Device cpu(startorch::CPU, startorch::DEFAULT);
  startorch::Device gpu(startorch::GPU, startorch::DEFAULT);

  int *gpu_data = (int *)startorch::makeData(sizeof(int) * size, gpu);

  fillDecreaseData<int>(gpu_data, size, gpu);

  int res[10];
  startorch::copyData(res, gpu_data, sizeof(int) * size,
                      startorch::DevicePair(gpu, cpu));

  for (int i = 0; i < size; i++)
    EXPECT_EQ(res[i], size - 1 - i);

  startorch::freeData(gpu_data, gpu);
}
} // namespace darkside
