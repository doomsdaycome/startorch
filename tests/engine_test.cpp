#include <gtest/gtest.h>

#include <startorch/common/literals.hpp>
#include <startorch/engine/memory.hpp>

#include "startorch/common/types.hpp"

namespace startorch {

TEST(MemoryTest, ConstructorTest) {
  Memory m0;
  Memory m1(1_KB, 1_KB, SystemType::kHost);
  Memory m2(1_KB, 1_KB, SystemType::kDevice);
  Memory m3(1_KB, 1_KB, SystemType::kUndefined);
  Memory m4(1_KB, 1_KB, SystemType::kOptionCount);
  Memory m5(1_KB, 0_KB, SystemType::kHost);
  Memory m6(0_KB, 1_KB, SystemType::kHost);
  Memory m7(0_KB, 0_KB, SystemType::kHost);

  EXPECT_EQ(m0.GetPhysicalAllocator().IsNull(), true);
  EXPECT_EQ(m0.GetVirtualAllocator().IsNull(), true);
  EXPECT_EQ(m0.GetSystemType(), SystemType::kUndefined);

  EXPECT_EQ(m1.GetPhysicalAllocator().GetBuffer().GetBytes(), 1_KB);
  EXPECT_EQ(m1.GetVirtualAllocator().GetBuffer().GetBytes(), 1_KB);
  EXPECT_EQ(m1.GetSystemType(), SystemType::kHost);

  EXPECT_EQ(m2.GetPhysicalAllocator().GetBuffer().GetBytes(), 1_KB);
  EXPECT_EQ(m2.GetVirtualAllocator().GetBuffer().GetBytes(), 1_KB);
  EXPECT_EQ(m2.GetSystemType(), SystemType::kDevice);
}

}  // namespace startorch
