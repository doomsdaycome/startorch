#include <startorch/common.hpp>
#include <startorch/device.hpp>
#include <startorch/memory.hpp>

#include <gtest/gtest.h>

namespace startorch {

TEST(DataTest, CopyCPUtoGPUtoCPUTest) {
  int size = 10;

  Device cpu(CPU, DEFAULT);
  Device gpu(GPU, DEFAULT);

  DevicePair p0(cpu, gpu);
  DevicePair p1(gpu, cpu);

  int *src = (int *)makeData(sizeof(int) * size, cpu);
  int *dst = (int *)makeData(sizeof(int) * size, gpu);
  int *res = (int *)makeData(sizeof(int) * size, cpu);

  for (int i = 0; i < size; i++)
    src[i] = i;

  copyData(dst, src, sizeof(int) * size, p0);
  copyData(res, dst, sizeof(int) * size, p1);

  for (int i = 0; i < size; i++)
    EXPECT_EQ(res[i], src[i]);

  freeData(src, cpu);
  freeData(dst, gpu);
  freeData(res, cpu);
}

TEST(DataTest, CopyCPUToCPUTest) {
  int size = 10;
  Device cpu(CPU, DEFAULT);

  int *src = (int *)makeData(sizeof(int) * size, cpu);
  int *dst = (int *)makeData(sizeof(int) * size, cpu);

  for (int i = 0; i < size; i++)
    src[i] = i;

  copyData(dst, src, sizeof(int) * size, DevicePair(cpu, cpu));

  for (int i = 0; i < size; i++)
    EXPECT_EQ(dst[i], i);

  freeData(src, cpu);
  freeData(dst, cpu);
}

TEST(DataTest, CopyGPUToGPUTest) {
  int size = 10;
  Device cpu(CPU, DEFAULT);
  Device gpu(GPU, DEFAULT);

  int tmp[10];
  for (int i = 0; i < size; i++)
    tmp[i] = i;

  int *src = (int *)makeData(sizeof(int) * size, gpu);
  int *dst = (int *)makeData(sizeof(int) * size, gpu);

  copyData(src, tmp, sizeof(int) * size, DevicePair(cpu, gpu));
  copyData(dst, src, sizeof(int) * size, DevicePair(gpu, gpu));

  int res[10];
  copyData(res, dst, sizeof(int) * size, DevicePair(gpu, cpu));

  for (int i = 0; i < size; i++)
    EXPECT_EQ(res[i], i);

  freeData(src, gpu);
  freeData(dst, gpu);
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
  copyData(res, s.getData(), sizeof(int) * 25, DevicePair(gpu, cpu));

  for (int i = 0; i < 25; i++)
    EXPECT_EQ(res[i], i);
}

} // namespace startorch
