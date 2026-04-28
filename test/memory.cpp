#include <startorch/common.hpp>
#include <startorch/device.hpp>
#include <startorch/memory.hpp>

#include <darkside/memory.hpp>

#include <gtest/gtest.h>

namespace startorch {

TEST(DataTest, CopyCPUtoGPUtoCPUTest) {
  int size = 10;

  Device cpu(CPU, DEFAULT);
  Device gpu(GPU, DEFAULT);

  DevicePair p0(cpu, gpu);
  DevicePair p1(gpu, cpu);

  int *src = (int *)darkside::makeData(sizeof(int) * size, cpu);
  int *dst = (int *)darkside::makeData(sizeof(int) * size, gpu);
  int *res = (int *)darkside::makeData(sizeof(int) * size, cpu);

  for (int i = 0; i < size; i++)
    src[i] = i;

  darkside::copyData(dst, src, sizeof(int) * size, p0);
  darkside::copyData(res, dst, sizeof(int) * size, p1);

  for (int i = 0; i < size; i++)
    EXPECT_EQ(res[i], src[i]);

  darkside::freeData(src, cpu);
  darkside::freeData(dst, gpu);
  darkside::freeData(res, cpu);
}

TEST(DataTest, CopyCPUToCPUTest) {
  int size = 10;
  Device cpu(CPU, DEFAULT);

  int *src = (int *)darkside::makeData(sizeof(int) * size, cpu);
  int *dst = (int *)darkside::makeData(sizeof(int) * size, cpu);

  for (int i = 0; i < size; i++)
    src[i] = i;

  darkside::copyData(dst, src, sizeof(int) * size, DevicePair(cpu, cpu));

  for (int i = 0; i < size; i++)
    EXPECT_EQ(dst[i], i);

  darkside::freeData(src, cpu);
  darkside::freeData(dst, cpu);
}

TEST(DataTest, CopyGPUToGPUTest) {
  int size = 10;
  Device cpu(CPU, DEFAULT);
  Device gpu(GPU, DEFAULT);

  int tmp[10];
  for (int i = 0; i < size; i++)
    tmp[i] = i;

  int *src = (int *)darkside::makeData(sizeof(int) * size, gpu);
  int *dst = (int *)darkside::makeData(sizeof(int) * size, gpu);

  darkside::copyData(src, tmp, sizeof(int) * size, DevicePair(cpu, gpu));
  darkside::copyData(dst, src, sizeof(int) * size, DevicePair(gpu, gpu));

  int res[10];
  darkside::copyData(res, dst, sizeof(int) * size, DevicePair(gpu, cpu));

  for (int i = 0; i < size; i++)
    EXPECT_EQ(res[i], i);

  darkside::freeData(src, gpu);
  darkside::freeData(dst, gpu);
}

TEST(StorageTest, ConstructorTest) {
  Device cpu(CPU, DEFAULT);

  Storage s(100, ScalarType::INT_32, cpu);

  EXPECT_NE(s.getData(), nullptr);
  EXPECT_EQ(s.getSize(), 100);
  EXPECT_EQ(s.getScalarType(), ScalarType::INT_32);
}

TEST(StorageTest, CopyConstructorTest) {
  Device cpu(CPU, DEFAULT);

  Storage a(100, ScalarType::INT_32, cpu);
  int *data = (int *)a.getData();

  for (int i = 0; i < 25; i++)
    data[i] = i;

  Storage b = a;
  int *bdata = (int *)b.getData();

  for (int i = 0; i < 25; i++)
    EXPECT_EQ(bdata[i], i);
}

TEST(StorageTest, MoveConstructorTest) {
  Device cpu(CPU, DEFAULT);

  Storage a(100, ScalarType::INT_32, cpu);
  void *ptr = a.getData();

  Storage b = std::move(a);

  EXPECT_EQ(b.getData(), ptr);
  EXPECT_EQ(a.getData(), nullptr);
}

TEST(StorageTest, CopyAssignmentTest) {
  Device cpu(CPU, DEFAULT);

  Storage a(100, ScalarType::INT_32, cpu);
  Storage b;

  int *data = (int *)a.getData();
  for (int i = 0; i < 25; i++)
    data[i] = i;

  b = a;

  int *bdata = (int *)b.getData();
  for (int i = 0; i < 25; i++)
    EXPECT_EQ(bdata[i], i);
}

TEST(StorageTest, MoveAssignmentTest) {
  Device cpu(CPU, DEFAULT);

  Storage a(100, ScalarType::INT_32, cpu);
  void *ptr = a.getData();

  Storage b;
  b = std::move(a);

  EXPECT_EQ(b.getData(), ptr);
  EXPECT_EQ(a.getData(), nullptr);
}

TEST(StorageTest, SetDevice) {
  Device cpu(CPU, DEFAULT);
  Device gpu(GPU, DEFAULT);

  Storage s(100, ScalarType::INT_32, cpu);
  int *data = (int *)s.getData();

  for (int i = 0; i < 25; i++)
    data[i] = i;

  s.setDevice(gpu);

  int res[25];
  darkside::copyData(res, s.getData(), sizeof(int) * 25, DevicePair(gpu, cpu));

  for (int i = 0; i < 25; i++)
    EXPECT_EQ(res[i], i);
}

TEST(StorageTest, FillDataInt64Test) {
  Device cpu(CPU, DEFAULT);

  Storage s(100, ScalarType::INT_32, cpu);
  s.fillData(42);

  int *data = (int *)s.getData();
  for (int i = 0; i < 25; i++)
    EXPECT_EQ(data[i], 42);
}

TEST(StorageTest, FillDataFloat64Test) {
  Device cpu(CPU, DEFAULT);

  Storage s(100, ScalarType::FLOAT_64, cpu);
  s.fillData(3.14);

  double *data = (double *)s.getData();
  for (int i = 0; i < 25; i++)
    EXPECT_DOUBLE_EQ(data[i], 3.14);
}

TEST(StorageTest, FillDataUnsignedInt64Test) {
  Device cpu(CPU, DEFAULT);

  Storage s(100, ScalarType::UNSIGNED_INT_64, cpu);
  s.fillData(123);

  uint64_t *data = (uint64_t *)s.getData();
  for (int i = 0; i < 25; i++)
    EXPECT_EQ(data[i], 123);
}

TEST(StorageTest, FillRandomDataTest) {
  Device cpu(CPU, DEFAULT);
  Device gpu(GPU, DEFAULT);

  Storage s(100, ScalarType::INT_32, gpu);
  s.fillData(0);
  s.fillRandomData();
  s.setDevice(cpu);

  int *data = (int *)s.getData();

  bool all_zero = true;
  for (int i = 0; i < 25; i++) {
    if (data[i] != 0) {
      all_zero = false;
      break;
    }
  }

  EXPECT_FALSE(all_zero);
}

TEST(StorageTest, FillIncreaseDataCPUTest) {
  Device cpu(CPU, DEFAULT);

  Storage s(100, ScalarType::INT_32, cpu);
  s.fillIncreaseData();

  int *data = (int *)s.getData();
  for (int i = 0; i < 25; i++)
    EXPECT_EQ(data[i], i);
}

TEST(StorageTest, FillDecreaseDataCPUTest) {
  Device cpu(CPU, DEFAULT);

  Storage s(100, ScalarType::INT_32, cpu);
  s.fillDecreaseData();

  int *data = (int *)s.getData();
  for (int i = 0; i < 25; i++)
    EXPECT_EQ(data[i], 100 - 1 - i);
}

TEST(StorageTest, FillIncreaseDataGPUTest) {
  Device cpu(CPU, DEFAULT);
  Device gpu(GPU, DEFAULT);

  Storage s(100, ScalarType::INT_32, gpu);
  s.fillIncreaseData();

  int res[25];
  darkside::copyData(res, s.getData(), sizeof(int) * 25, DevicePair(gpu, cpu));

  for (int i = 0; i < 25; i++)
    EXPECT_EQ(res[i], i);
}

TEST(StorageTest, FillDecreaseDataGPUTest) {
  Device cpu(CPU, DEFAULT);
  Device gpu(GPU, DEFAULT);

  Storage s(100, ScalarType::INT_32, gpu);
  s.fillDecreaseData();

  int res[25];
  darkside::copyData(res, s.getData(), sizeof(int) * 25, DevicePair(gpu, cpu));

  for (int i = 0; i < 25; i++)
    EXPECT_EQ(res[i], 100 - 1 - i);
}
} // namespace startorch
