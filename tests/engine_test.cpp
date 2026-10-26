#include <gtest/gtest.h>

#include "startorch/common/literals.hpp"
#include "startorch/common/types.hpp"
#include "startorch/engine/memory.hpp"
#include "startorch/engine/system.hpp"
#include "startorch/memory/buffer.hpp"

namespace startorch {

TEST(MemoryTest, ConstructorTest) {
  Memory m0;
  Memory m1(1_MB, 2_MB, SystemType::kHost);
  Memory m2(1_MB, 2_MB, SystemType::kDevice);
  Memory m3(0_MB, 0_MB, SystemType::kHost);
  Memory m4(1_MB, 2_MB, SystemType::kUndefined);
  Memory m5(1_MB, 2_MB, SystemType::kOptionCount);

  EXPECT_EQ(m0.IsNull(), true);
  EXPECT_EQ(m0.GetSystemType(), SystemType::kUndefined);

  EXPECT_EQ(m1.IsNull(), false);
  EXPECT_EQ(m1.GetSystemType(), SystemType::kHost);
  EXPECT_EQ(m1.GetPhysicalAllocator().GetAlignedSize(), 64ul);
  EXPECT_EQ(m1.GetVirtualAllocator().GetAlignedSize(), 4096ul);

  EXPECT_EQ(m2.IsNull(), false);
  EXPECT_EQ(m2.GetSystemType(), SystemType::kDevice);
  EXPECT_EQ(m2.GetPhysicalAllocator().GetAlignedSize(), 256ul);
  EXPECT_EQ(m2.GetVirtualAllocator().GetAlignedSize(), 256ul);

  EXPECT_EQ(m3.IsNull(), true);
  EXPECT_EQ(m4.IsNull(), true);
  EXPECT_EQ(m5.IsNull(), true);
}

TEST(MemoryTest, OperatorBoolTest) {
  Memory m0;
  Memory m1(1_MB, 2_MB, SystemType::kHost);

  EXPECT_EQ(static_cast<bool>(m0), false);
  EXPECT_EQ(!m0, true);

  EXPECT_EQ(static_cast<bool>(m1), true);
  EXPECT_EQ(!m1, false);
}

TEST(MemoryTest, GetPhysicalAllocatorTest) {
  Memory m0(1_MB, 2_MB, SystemType::kHost);
  const Memory m1(1_MB, 2_MB, SystemType::kHost);

  EXPECT_NE(m0.GetPhysicalAllocator().IsNull(), true);
  EXPECT_NE(m1.GetPhysicalAllocator().IsNull(), true);
  EXPECT_EQ(m0.GetPhysicalAllocator().GetType(), MallocType::kHost);
  EXPECT_EQ(m1.GetPhysicalAllocator().GetType(), MallocType::kHost);
}

TEST(MemoryTest, GetVirtualAllocatorTest) {
  Memory m0(1_MB, 2_MB, SystemType::kHost);
  const Memory m1(1_MB, 2_MB, SystemType::kHost);

  EXPECT_NE(m0.GetVirtualAllocator().IsNull(), true);
  EXPECT_NE(m1.GetVirtualAllocator().IsNull(), true);
  EXPECT_EQ(m0.GetVirtualAllocator().GetType(), MallocType::kPinned);
  EXPECT_EQ(m1.GetVirtualAllocator().GetType(), MallocType::kPinned);
}

TEST(MemoryTest, GetSystemTypeTest) {
  Memory m0(1_MB, 2_MB, SystemType::kHost);
  Memory m1(1_MB, 2_MB, SystemType::kDevice);

  EXPECT_EQ(m0.GetSystemType(), SystemType::kHost);
  EXPECT_EQ(m1.GetSystemType(), SystemType::kDevice);
}

TEST(MemoryTest, IsNullTest) {
  Memory m0;
  Memory m1(1_MB, 2_MB, SystemType::kHost);
  Memory m2(0_MB, 0_MB, SystemType::kHost);

  EXPECT_EQ(m0.IsNull(), true);
  EXPECT_EQ(m1.IsNull(), false);
  EXPECT_EQ(m2.IsNull(), true);
}

TEST(MemoryTest, GetNullTest) {
  Memory& m0 = Memory::GetNull();

  EXPECT_EQ(m0.IsNull(), true);
  EXPECT_EQ(m0.GetSystemType(), SystemType::kUndefined);
  EXPECT_EQ(m0.GetPhysicalAllocator().IsNull(), true);
  EXPECT_EQ(m0.GetVirtualAllocator().IsNull(), true);
}

TEST(SystemTest, ConstructorTest) {
  System s0;
  System s1(0ul, 1_MB, 2_MB, SystemType::kHost);
  System s2(1ul, 1_MB, 2_MB, SystemType::kDevice);
  System s3(2ul, 0_MB, 0_MB, SystemType::kHost);
  System s4(3ul, 1_MB, 2_MB, SystemType::kUndefined);

  EXPECT_EQ(s0.IsNull(), true);
  EXPECT_EQ(s0.GetIndex(), 0ul);
  EXPECT_EQ(s0.GetType(), SystemType::kUndefined);

  EXPECT_EQ(s1.IsNull(), false);
  EXPECT_EQ(s1.GetIndex(), 0ul);
  EXPECT_EQ(s1.GetType(), SystemType::kHost);

  EXPECT_EQ(s2.IsNull(), false);
  EXPECT_EQ(s2.GetIndex(), 1ul);
  EXPECT_EQ(s2.GetType(), SystemType::kDevice);

  EXPECT_EQ(s3.IsNull(), true);
  EXPECT_EQ(s4.IsNull(), true);
}

TEST(SystemTest, OperatorBoolTest) {
  System s0;
  System s1(0ul, 1_MB, 2_MB, SystemType::kHost);

  EXPECT_EQ(static_cast<bool>(s0), false);
  EXPECT_EQ(!s0, true);

  EXPECT_EQ(static_cast<bool>(s1), true);
  EXPECT_EQ(!s1, false);
}

TEST(SystemTest, GetIndexTest) {
  System s0(5ul, 1_MB, 2_MB, SystemType::kHost);
  System s1(10ul, 1_MB, 2_MB, SystemType::kDevice);

  EXPECT_EQ(s0.GetIndex(), 5ul);
  EXPECT_EQ(s1.GetIndex(), 10ul);
}

TEST(SystemTest, GetMemoryTest) {
  System s0(0ul, 1_MB, 2_MB, SystemType::kHost);
  const System s1(1ul, 1_MB, 2_MB, SystemType::kDevice);

  EXPECT_EQ(s0.GetMemory().IsNull(), false);
  EXPECT_EQ(s1.GetMemory().IsNull(), false);
  EXPECT_EQ(s0.GetMemory().GetSystemType(), SystemType::kHost);
  EXPECT_EQ(s1.GetMemory().GetSystemType(), SystemType::kDevice);
}

TEST(SystemTest, GetTypeTest) {
  System s0(0ul, 1_MB, 2_MB, SystemType::kHost);
  System s1(0ul, 1_MB, 2_MB, SystemType::kDevice);

  EXPECT_EQ(s0.GetType(), SystemType::kHost);
  EXPECT_EQ(s1.GetType(), SystemType::kDevice);
}

TEST(SystemTest, IsNullTest) {
  System s0;
  System s1(0ul, 1_MB, 2_MB, SystemType::kHost);
  System s2(0ul, 0_MB, 0_MB, SystemType::kHost);

  EXPECT_EQ(s0.IsNull(), true);
  EXPECT_EQ(s1.IsNull(), false);
  EXPECT_EQ(s2.IsNull(), true);
}

TEST(SystemTest, GetNullTest) {
  System& s0 = System::GetNull();

  EXPECT_EQ(s0.IsNull(), true);
  EXPECT_EQ(s0.GetIndex(), 0ul);
  EXPECT_EQ(s0.GetType(), SystemType::kUndefined);
  EXPECT_EQ(s0.GetMemory().IsNull(), true);
}

TEST(SystemTest, NewBufferTest) {
  System s0(0, 1_MB, 1_MB, SystemType::kHost);
  System s1(0, 1_MB, 1_MB, SystemType::kDevice);

  Buffer b0 = s0.NewBuffer(10ul, MallocType::kHost);
  Buffer b1 = s0.NewBuffer(10ul, MallocType::kPinned);
  Buffer b2 = s0.NewBuffer(10ul, MallocType::kDevice);

  Buffer b3 = s1.NewBuffer(10ul, MallocType::kDevice);
  Buffer b4 = s1.NewBuffer(10ul, MallocType::kUnified);
  Buffer b5 = s1.NewBuffer(10ul, MallocType::kHost);

  EXPECT_EQ(!b0, false);
  EXPECT_EQ(b0.GetMallocType(), MallocType::kHost);

  EXPECT_EQ(!b1, false);
  EXPECT_EQ(b1.GetMallocType(), MallocType::kPinned);

  EXPECT_EQ(!b2, true);

  EXPECT_EQ(!b3, false);
  EXPECT_EQ(b3.GetMallocType(), MallocType::kDevice);

  EXPECT_EQ(!b4, false);
  EXPECT_EQ(b4.GetMallocType(), MallocType::kUnified);

  EXPECT_EQ(!b5, true);
}

}  // namespace startorch
