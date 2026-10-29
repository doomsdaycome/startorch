#include <gtest/gtest.h>

#include "startorch/common/literals.hpp"
#include "startorch/common/types.hpp"
#include "startorch/device/engine.hpp"
#include "startorch/device/memory.hpp"
#include "startorch/memory/buffer.hpp"

namespace startorch {

TEST(MemoryTest, ConstructorTest) {
  Memory m0;
  Memory m1(1_MB, 2_MB, EngineType::kHost);
  Memory m2(1_MB, 2_MB, EngineType::kDevice);
  Memory m3(0_MB, 0_MB, EngineType::kHost);
  Memory m4(1_MB, 2_MB, EngineType::kUndefined);
  Memory m5(1_MB, 2_MB, EngineType::kOptionCount);

  EXPECT_EQ(m0.IsNull(), true);
  EXPECT_EQ(m0.GetEngineType(), EngineType::kUndefined);

  EXPECT_EQ(m1.IsNull(), false);
  EXPECT_EQ(m1.GetEngineType(), EngineType::kHost);
  EXPECT_EQ(m1.GetDefaultAllocator().GetAlignedSize(), 64ul);
  EXPECT_EQ(m1.GetVirtualAllocator().GetAlignedSize(), 4096ul);

  EXPECT_EQ(m2.IsNull(), false);
  EXPECT_EQ(m2.GetEngineType(), EngineType::kDevice);
  EXPECT_EQ(m2.GetDefaultAllocator().GetAlignedSize(), 256ul);
  EXPECT_EQ(m2.GetVirtualAllocator().GetAlignedSize(), 256ul);

  EXPECT_EQ(m3.IsNull(), true);
  EXPECT_EQ(m4.IsNull(), true);
  EXPECT_EQ(m5.IsNull(), true);
}

TEST(MemoryTest, OperatorBoolTest) {
  Memory m0;
  Memory m1(1_MB, 2_MB, EngineType::kHost);

  EXPECT_EQ(static_cast<bool>(m0), false);
  EXPECT_EQ(!m0, true);

  EXPECT_EQ(static_cast<bool>(m1), true);
  EXPECT_EQ(!m1, false);
}

TEST(MemoryTest, GetDefaultAllocatorTest) {
  Memory m0(1_MB, 2_MB, EngineType::kHost);
  const Memory m1(1_MB, 2_MB, EngineType::kHost);

  EXPECT_NE(m0.GetDefaultAllocator().IsNull(), true);
  EXPECT_NE(m1.GetDefaultAllocator().IsNull(), true);
  EXPECT_EQ(m0.GetDefaultAllocator().GetType(), MallocType::kHost);
  EXPECT_EQ(m1.GetDefaultAllocator().GetType(), MallocType::kHost);
}

TEST(MemoryTest, GetVirtualAllocatorTest) {
  Memory m0(1_MB, 2_MB, EngineType::kHost);
  const Memory m1(1_MB, 2_MB, EngineType::kHost);

  EXPECT_NE(m0.GetVirtualAllocator().IsNull(), true);
  EXPECT_NE(m1.GetVirtualAllocator().IsNull(), true);
  EXPECT_EQ(m0.GetVirtualAllocator().GetType(), MallocType::kPinned);
  EXPECT_EQ(m1.GetVirtualAllocator().GetType(), MallocType::kPinned);
}

TEST(MemoryTest, GetEngineTypeTest) {
  Memory m0(1_MB, 2_MB, EngineType::kHost);
  Memory m1(1_MB, 2_MB, EngineType::kDevice);

  EXPECT_EQ(m0.GetEngineType(), EngineType::kHost);
  EXPECT_EQ(m1.GetEngineType(), EngineType::kDevice);
}

TEST(MemoryTest, IsNullTest) {
  Memory m0;
  Memory m1(1_MB, 2_MB, EngineType::kHost);
  Memory m2(0_MB, 0_MB, EngineType::kHost);

  EXPECT_EQ(m0.IsNull(), true);
  EXPECT_EQ(m1.IsNull(), false);
  EXPECT_EQ(m2.IsNull(), true);
}

TEST(MemoryTest, GetNullTest) {
  Memory& m0 = Memory::GetNull();

  EXPECT_EQ(m0.IsNull(), true);
  EXPECT_EQ(m0.GetEngineType(), EngineType::kUndefined);
  EXPECT_EQ(m0.GetDefaultAllocator().IsNull(), true);
  EXPECT_EQ(m0.GetVirtualAllocator().IsNull(), true);
}

TEST(EngineTest, ConstructorTest) {
  Engine s0;
  Engine s1(0ul, 1_MB, 2_MB, EngineType::kHost);
  Engine s2(1ul, 1_MB, 2_MB, EngineType::kDevice);
  Engine s3(2ul, 0_MB, 0_MB, EngineType::kHost);
  Engine s4(3ul, 1_MB, 2_MB, EngineType::kUndefined);

  EXPECT_EQ(s0.IsNull(), true);
  EXPECT_EQ(s0.GetIndex(), 0ul);
  EXPECT_EQ(s0.GetType(), EngineType::kUndefined);

  EXPECT_EQ(s1.IsNull(), false);
  EXPECT_EQ(s1.GetIndex(), 0ul);
  EXPECT_EQ(s1.GetType(), EngineType::kHost);

  EXPECT_EQ(s2.IsNull(), false);
  EXPECT_EQ(s2.GetIndex(), 1ul);
  EXPECT_EQ(s2.GetType(), EngineType::kDevice);

  EXPECT_EQ(s3.IsNull(), true);
  EXPECT_EQ(s4.IsNull(), true);
}

TEST(EngineTest, OperatorBoolTest) {
  Engine s0;
  Engine s1(0ul, 1_MB, 2_MB, EngineType::kHost);

  EXPECT_EQ(static_cast<bool>(s0), false);
  EXPECT_EQ(!s0, true);

  EXPECT_EQ(static_cast<bool>(s1), true);
  EXPECT_EQ(!s1, false);
}

TEST(EngineTest, GetIndexTest) {
  Engine s0(5ul, 1_MB, 2_MB, EngineType::kHost);
  Engine s1(10ul, 1_MB, 2_MB, EngineType::kDevice);

  EXPECT_EQ(s0.GetIndex(), 5ul);
  EXPECT_EQ(s1.GetIndex(), 10ul);
}

TEST(EngineTest, GetMemoryTest) {
  Engine s0(0ul, 1_MB, 2_MB, EngineType::kHost);
  const Engine s1(1ul, 1_MB, 2_MB, EngineType::kDevice);

  EXPECT_EQ(s0.GetMemory().IsNull(), false);
  EXPECT_EQ(s1.GetMemory().IsNull(), false);
  EXPECT_EQ(s0.GetMemory().GetEngineType(), EngineType::kHost);
  EXPECT_EQ(s1.GetMemory().GetEngineType(), EngineType::kDevice);
}

TEST(EngineTest, GetTypeTest) {
  Engine s0(0ul, 1_MB, 2_MB, EngineType::kHost);
  Engine s1(0ul, 1_MB, 2_MB, EngineType::kDevice);

  EXPECT_EQ(s0.GetType(), EngineType::kHost);
  EXPECT_EQ(s1.GetType(), EngineType::kDevice);
}

TEST(EngineTest, IsNullTest) {
  Engine s0;
  Engine s1(0ul, 1_MB, 2_MB, EngineType::kHost);
  Engine s2(0ul, 0_MB, 0_MB, EngineType::kHost);

  EXPECT_EQ(s0.IsNull(), true);
  EXPECT_EQ(s1.IsNull(), false);
  EXPECT_EQ(s2.IsNull(), true);
}

TEST(EngineTest, GetNullTest) {
  Engine& s0 = Engine::GetNull();

  EXPECT_EQ(s0.IsNull(), true);
  EXPECT_EQ(s0.GetIndex(), 0ul);
  EXPECT_EQ(s0.GetType(), EngineType::kUndefined);
  EXPECT_EQ(s0.GetMemory().IsNull(), true);
}

// TEST(EngineTest, NewBufferTest) {
//   Engine s0(0, 1_MB, 1_MB, EngineType::kHost);
//   Engine s1(0, 1_MB, 1_MB, EngineType::kDevice);
//
//   Buffer b0 = s0.NewBuffer(10ul, MallocType::kHost);
//   Buffer b1 = s0.NewBuffer(10ul, MallocType::kPinned);
//   Buffer b2 = s0.NewBuffer(10ul, MallocType::kDevice);
//
//   Buffer b3 = s1.NewBuffer(10ul, MallocType::kDevice);
//   Buffer b4 = s1.NewBuffer(10ul, MallocType::kUnified);
//   Buffer b5 = s1.NewBuffer(10ul, MallocType::kHost);
//
//   EXPECT_EQ(!b0, false);
//   EXPECT_EQ(b0.GetMallocType(), MallocType::kHost);
//
//   EXPECT_EQ(!b1, false);
//   EXPECT_EQ(b1.GetMallocType(), MallocType::kPinned);
//
//   EXPECT_EQ(!b2, true);
//
//   EXPECT_EQ(!b3, false);
//   EXPECT_EQ(b3.GetMallocType(), MallocType::kDevice);
//
//   EXPECT_EQ(!b4, false);
//   EXPECT_EQ(b4.GetMallocType(), MallocType::kUnified);
//
//   EXPECT_EQ(!b5, true);
// }

}  // namespace startorch
