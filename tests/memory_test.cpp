#include <gtest/gtest.h>

#include <cstdint>

#include "startorch/common/literals.hpp"
#include "startorch/common/types.hpp"
#include "startorch/common/utilities.hpp"
#include "startorch/memory/allocator.hpp"
#include "startorch/memory/buffer.hpp"

namespace startorch {

TEST(BufferTest, ConstructorTest) {
  std::uint64_t v0 = 0ul;

  Buffer b0;
  Buffer b1(&v0, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b2(nullptr, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b3(&v0, 0ul, MallocType::kHost);
  Buffer b4(&v0, SizeOfCPPType(std::uint64_t), MallocType::kUndefined);
  Buffer b5(&v0, SizeOfCPPType(std::uint64_t), MallocType::kOptionCount);

  EXPECT_EQ(b0.GetData(), nullptr);
  EXPECT_EQ(b0.GetBytes(), 0ul);
  EXPECT_EQ(b0.GetMallocType(), MallocType::kUndefined);

  EXPECT_EQ(b1.GetData(), &v0);
  EXPECT_EQ(b1.GetBytes(), SizeOfCPPType(std::uint64_t));
  EXPECT_EQ(b1.GetMallocType(), MallocType::kHost);

  EXPECT_EQ(b2.GetData(), nullptr);
  EXPECT_EQ(b2.GetBytes(), 0ul);
  EXPECT_EQ(b2.GetMallocType(), MallocType::kUndefined);

  EXPECT_EQ(b3.GetData(), nullptr);
  EXPECT_EQ(b3.GetBytes(), 0ul);
  EXPECT_EQ(b3.GetMallocType(), MallocType::kUndefined);

  EXPECT_EQ(b4.GetData(), nullptr);
  EXPECT_EQ(b4.GetBytes(), 0ul);
  EXPECT_EQ(b4.GetMallocType(), MallocType::kUndefined);

  EXPECT_EQ(b5.GetData(), nullptr);
  EXPECT_EQ(b5.GetBytes(), 0ul);
  EXPECT_EQ(b5.GetMallocType(), MallocType::kUndefined);
}

TEST(BufferTest, OperatorBoolTest) {
  std::uint64_t v0 = 0ul;

  Buffer b0;
  Buffer b1(&v0, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b2(nullptr, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b3(&v0, 0ul, MallocType::kHost);
  Buffer b4(&v0, SizeOfCPPType(std::uint64_t), MallocType::kUndefined);
  Buffer b5(&v0, SizeOfCPPType(std::uint64_t), MallocType::kOptionCount);

  EXPECT_EQ(static_cast<bool>(b0), false);
  EXPECT_EQ(!b0, true);

  EXPECT_EQ(static_cast<bool>(b1), true);
  EXPECT_EQ(!b1, false);

  EXPECT_EQ(static_cast<bool>(b2), false);
  EXPECT_EQ(static_cast<bool>(b3), false);
  EXPECT_EQ(static_cast<bool>(b4), false);
  EXPECT_EQ(!b5, true);
}

TEST(BufferTest, GetDataTest) {
  std::uint8_t v0 = 7;
  std::uint8_t v1 = 9;

  Buffer b0(&v0, SizeOfCPPType(std::uint8_t), MallocType::kPinned);
  const Buffer b1(&v1, SizeOfCPPType(std::uint8_t), MallocType::kPinned);

  EXPECT_EQ(*b0.GetData<uint8_t>(), v0);
  EXPECT_EQ(*b1.GetData<uint8_t>(), v1);
}

TEST(BufferTest, GetBytesTest) {
  std::uint64_t v0 = 0;
  std::uint64_t v1 = 2048ul;

  Buffer b0(&v0, v1, MallocType::kHost);

  EXPECT_EQ(b0.GetBytes(), v1);
}

TEST(BufferTest, GetMallocTypeTest) {
  std::uint8_t v0 = 0;

  MallocType t0 = MallocType::kDevice;
  MallocType t1 = MallocType::kDevice;
  MallocType t2 = MallocType::kDevice;
  MallocType t3 = MallocType::kDevice;

  Buffer b0(&v0, 64ul, t0);
  Buffer b1(&v0, 64ul, t1);
  Buffer b2(&v0, 64ul, t2);
  Buffer b3(&v0, 64ul, t3);

  EXPECT_EQ(b0.GetMallocType(), t0);
  EXPECT_EQ(b1.GetMallocType(), t1);
  EXPECT_EQ(b2.GetMallocType(), t2);
  EXPECT_EQ(b3.GetMallocType(), t3);
}

TEST(BufferTest, IsNullTest) {
  std::uint64_t v0 = 0ul;

  Buffer b0;
  Buffer b1(&v0, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b2(nullptr, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b3(&v0, 0ul, MallocType::kHost);
  Buffer b4(&v0, SizeOfCPPType(std::uint64_t), MallocType::kUndefined);
  Buffer b5(&v0, SizeOfCPPType(std::uint64_t), MallocType::kUndefined);

  EXPECT_EQ(b0.IsNull(), true);
  EXPECT_EQ(b1.IsNull(), false);
  EXPECT_EQ(b2.IsNull(), true);
  EXPECT_EQ(b3.IsNull(), true);
  EXPECT_EQ(b4.IsNull(), true);
}

TEST(AllocatorTest, ConstructorTest) {
  Allocator a0;
  Allocator a1(1_MB, MallocType::kHost);
  Allocator a2(1_MB, MallocType::kDevice);
  Allocator a3(1_MB, MallocType::kPinned);
  Allocator a4(1_MB, MallocType::kUnified);
  Allocator a5(0_MB, MallocType::kHost);
  Allocator a6(1_MB, MallocType::kUndefined);
  Allocator a7(1_MB, MallocType::kOptionCount);

  EXPECT_EQ(a0.GetAlignedSize(), 0ul);
  EXPECT_EQ(a0.GetType(), MallocType::kUndefined);
  EXPECT_EQ(a0.IsNull(), true);

  EXPECT_GT(a1.GetAlignedSize(), 0ul);
  EXPECT_EQ(a1.GetType(), MallocType::kHost);
  EXPECT_EQ(a1.IsNull(), false);

  EXPECT_EQ(a2.GetType(), MallocType::kDevice);
  EXPECT_EQ(a3.GetType(), MallocType::kPinned);
  EXPECT_EQ(a4.GetType(), MallocType::kUnified);

  EXPECT_EQ(a5.GetAlignedSize(), 0ul);
  EXPECT_EQ(a6.GetType(), MallocType::kUndefined);
  EXPECT_EQ(a7.GetType(), MallocType::kUndefined);
}

TEST(AllocatorTest, OperatorBoolTest) {
  Allocator a0;
  Allocator a1(1_MB, MallocType::kHost);

  EXPECT_EQ(static_cast<bool>(a0), false);
  EXPECT_EQ(!a0, true);

  EXPECT_EQ(static_cast<bool>(a1), true);
  EXPECT_EQ(!a1, false);
}

TEST(AllocatorTest, GetBufferTest) {
  Allocator a0;
  Allocator a1(1_MB, MallocType::kOptionCount);
  Allocator a2(1_MB, MallocType::kHost);
  const Allocator a3(1_MB, MallocType::kHost);

  EXPECT_EQ(a0.GetBuffer().IsNull(), true);
  EXPECT_EQ(a1.GetBuffer().IsNull(), true);
  EXPECT_EQ(a0.GetBuffer().GetData(), nullptr);
  EXPECT_EQ(a1.GetBuffer().GetData(), nullptr);
  EXPECT_NE(a2.GetBuffer().GetData(), nullptr);
  EXPECT_NE(a3.GetBuffer().GetData(), nullptr);
  EXPECT_EQ(a2.GetBuffer().GetBytes(), 1_MB);
  EXPECT_EQ(a3.GetBuffer().GetBytes(), 1_MB);
}

TEST(AllocatorTest, GetOffsetTest) {
  Allocator a0(1_MB, MallocType::kHost);
  EXPECT_EQ(a0.GetOffset(), 0ul);

  Buffer b0 = a0.NewBuffer(1_KB);
  EXPECT_GT(a0.GetOffset(), 0ul);
  EXPECT_EQ(a0.GetOffset(), 1_KB);
}

TEST(AllocatorTest, GetAlignedSizeTest) {
  Allocator a0(1_MB, MallocType::kHost);
  Allocator a1(1_MB, MallocType::kDevice);
  Allocator a2(1_MB, MallocType::kPinned);

  EXPECT_EQ(a0.GetAlignedSize(), 64ul);
  EXPECT_EQ(a1.GetAlignedSize(), 256ul);
  EXPECT_EQ(a2.GetAlignedSize(), 4096ul);
}

TEST(AllocatorTest, GetTypeTest) {
  Allocator a0(1_MB, MallocType::kHost);
  Allocator a1(1_MB, MallocType::kDevice);

  EXPECT_EQ(a0.GetType(), MallocType::kHost);
  EXPECT_EQ(a1.GetType(), MallocType::kDevice);
}

TEST(AllocatorTest, IsNullTest) {
  Allocator a0;
  Allocator a1(1_MB, MallocType::kHost);
  Allocator a2(0_MB, MallocType::kHost);

  EXPECT_EQ(a0.IsNull(), true);
  EXPECT_EQ(a1.IsNull(), false);
  EXPECT_EQ(a2.IsNull(), true);
}

TEST(AllocatorTest, NewBufferTest) {
  Allocator a0(1_MB, MallocType::kHost);
  EXPECT_EQ(a0.GetAlignedSize(), 64ul);
  EXPECT_EQ(a0.GetOffset(), 0ul);

  Buffer b0 = a0.NewBuffer(10ul);
  EXPECT_NE(b0.GetData(), nullptr);
  EXPECT_EQ(b0.GetBytes(), 10ul);
  EXPECT_EQ(b0.GetMallocType(), MallocType::kHost);
  EXPECT_EQ(a0.GetOffset(), 64ul);

  Buffer b1 = a0.NewBuffer(100ul);
  EXPECT_NE(b1.GetData(), nullptr);
  EXPECT_EQ(b1.GetBytes(), 100ul);
  EXPECT_EQ(a0.GetOffset(), 192ul);

  Buffer b2 = a0.NewBuffer(0ul);
  EXPECT_EQ(b2.IsNull(), true);

  Allocator a1(1_MB, MallocType::kDevice);
  EXPECT_EQ(a1.GetAlignedSize(), 256ul);
  EXPECT_EQ(a1.GetOffset(), 0ul);

  Buffer b3 = a1.NewBuffer(100ul);
  EXPECT_NE(b3.GetData(), nullptr);
  EXPECT_EQ(b3.GetBytes(), 100ul);
  EXPECT_EQ(a1.GetOffset(), 256ul);

  Allocator a2(1_MB, MallocType::kUnified);
  EXPECT_EQ(a2.GetAlignedSize(), 256ul);
  EXPECT_EQ(a2.GetOffset(), 0ul);

  Buffer b4 = a2.NewBuffer(50ul);
  EXPECT_NE(b4.GetData(), nullptr);
  EXPECT_EQ(b4.GetBytes(), 50ul);
  EXPECT_EQ(a2.GetOffset(), 256ul);
}

TEST(AllocatorTest, DeleteBufferTest) {
  Allocator a0(1_MB, MallocType::kHost);

  Buffer b0 = a0.NewBuffer(10ul);
  Buffer b1 = a0.NewBuffer(100ul);
  Buffer b2 = a0.NewBuffer(50ul);

  EXPECT_EQ(a0.GetOffset(), 256ul);

  a0.DeleteBuffer(b1);
  EXPECT_EQ(a0.GetOffset(), 256ul);

  a0.DeleteBuffer(b2);
  EXPECT_EQ(a0.GetOffset(), 192ul);

  a0.DeleteBuffer(b1);
  EXPECT_EQ(a0.GetOffset(), 64ul);

  a0.DeleteBuffer(b0);
  EXPECT_EQ(a0.GetOffset(), 0ul);

  Allocator a1(1_MB, MallocType::kDevice);

  Buffer b3 = a1.NewBuffer(10ul);
  Buffer b4 = a1.NewBuffer(100ul);
  Buffer b5 = a1.NewBuffer(50ul);

  EXPECT_EQ(a1.GetOffset(), 768ul);

  a1.DeleteBuffer(b5);
  EXPECT_EQ(a1.GetOffset(), 512ul);

  a1.DeleteBuffer(b4);
  EXPECT_EQ(a1.GetOffset(), 256ul);

  a1.DeleteBuffer(b3);
  EXPECT_EQ(a1.GetOffset(), 0ul);
}

TEST(AllocatorTest, GetNullTest) {
  Allocator& a0 = Allocator::GetNull();

  EXPECT_EQ(a0.IsNull(), true);
  EXPECT_EQ(a0.GetAlignedSize(), 0ul);
  EXPECT_EQ(a0.GetType(), MallocType::kUndefined);
}

}  // namespace startorch
