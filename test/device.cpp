#include <startorch/common.hpp>
#include <startorch/device.hpp>

#include <gtest/gtest.h>

namespace startorch {
TEST(DeviceTest, DefaultConstructorTest) {
  Device d0;

  EXPECT_EQ(d0.getDeviceType(), DeviceType::CPU);
  EXPECT_EQ(d0.getMemoryType(), MemoryType::DEFAULT);
}

TEST(DeviceTest, CustomConstructorTest) {
  Device d0(DeviceType::GPU, MemoryType::UNIFIED);
  Device d1(DeviceType::CPU, MemoryType::PINNED);
  Device d2(DeviceType::CPU, MemoryType::UNIFIED);
  Device d3(DeviceType::GPU, MemoryType::PINNED);

  EXPECT_EQ(d0.getDeviceType(), DeviceType::GPU);
  EXPECT_EQ(d0.getMemoryType(), MemoryType::UNIFIED);

  EXPECT_EQ(d1.getDeviceType(), DeviceType::CPU);
  EXPECT_EQ(d1.getMemoryType(), MemoryType::PINNED);

  EXPECT_EQ(d2.getDeviceType(), DeviceType::CPU);
  EXPECT_EQ(d2.getMemoryType(), MemoryType::DEFAULT);

  EXPECT_EQ(d3.getDeviceType(), DeviceType::GPU);
  EXPECT_EQ(d3.getMemoryType(), MemoryType::DEFAULT);
}

TEST(DeviceTest, EqualityOperatorTest) {
  Device d0(DeviceType::GPU, MemoryType::UNIFIED);
  Device d1(DeviceType::GPU, MemoryType::UNIFIED);
  Device d2(DeviceType::CPU, MemoryType::DEFAULT);

  EXPECT_TRUE(d0 == d1);
  EXPECT_TRUE(d0 != d2);
}

TEST(DevicePairTest, CustomConstructorTest) {
  Device d0(DeviceType::CPU, MemoryType::PINNED);
  Device d1(DeviceType::GPU, MemoryType::UNIFIED);

  DevicePair p0(d0, d1);

  EXPECT_EQ(p0.getFirstDevice(), d0);
  EXPECT_EQ(p0.getSecondDevice(), d1);

  EXPECT_EQ(p0.getFirstDevice().getMemoryType(), MemoryType::PINNED);
  EXPECT_EQ(p0.getSecondDevice().getMemoryType(), MemoryType::UNIFIED);
}

TEST(DevicePairTest, EqualityOperatorTest) {
  Device d0(DeviceType::CPU, MemoryType::DEFAULT);
  Device d1(DeviceType::GPU, MemoryType::DEFAULT);

  DevicePair p0(d0, d1);
  DevicePair p1(d0, d1);
  DevicePair p2(d1, d0);

  EXPECT_EQ(p0, p1);
  EXPECT_NE(p0, p2);
}

TEST(DevicePairTest, DefaultConstructor) {
  Device d0;

  DevicePair p3;

  EXPECT_EQ(p3.getFirstDevice(), d0);
  EXPECT_EQ(p3.getSecondDevice(), d0);
}
} // namespace startorch
