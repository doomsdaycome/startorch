#include <startorch/common.hpp>
#include <startorch/device.hpp>
#include <startorch/memory.hpp>
#include <startorch/random.hpp>

#include <gtest/gtest.h>

namespace startorch {
TEST(MemoryTest, CopyTest) {
  int size = 10;

  Device d0(CPU, DEFAULT);
  Device d1(GPU, DEFAULT);

  DevicePair p0(d0, d1);
  DevicePair p1(d1, d0);

  int *src = (int *)makeMemory(sizeof(int) * size, d0);
  int *res = (int *)makeMemory(sizeof(int) * size, d0);
  int *dst = (int *)makeMemory(sizeof(int) * size, d1);

  for (int i = 0; i < size; i++)
    src[i] = i;

  copyMemory(dst, src, sizeof(int) * size, p0);
  copyMemory(res, dst, sizeof(int) * size, p1);

  for (int i = 0; i < size; i++)
    EXPECT_EQ(res[i], src[i]);
}
} // namespace startorch
