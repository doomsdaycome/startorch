#include <gtest/gtest.h>

#include <cstdint>

#include "startorch/common/literals.hpp"
#include "startorch/common/types.hpp"
#include "startorch/common/utilities.hpp"
#include "startorch/memory/allocator.hpp"
#include "startorch/memory/buffer.hpp"

TEST(BufferTest, ConstructorTest) {
  std::uint64_t v0 = 0ul;

  startorch::Buffer b0;
  startorch::Buffer b1 = startorch::Buffer(&v0, SizeOfCPPType(std::uint64_t),
                                           startorch::MallocType::kHost);
  startorch::Buffer b2 = startorch::Buffer(
      nullptr, SizeOfCPPType(std::uint64_t), startorch::MallocType::kHost);
  startorch::Buffer b3 =
      startorch::Buffer(&v0, 0ul, startorch::MallocType::kHost);
  startorch::Buffer b4 = startorch::Buffer(&v0, SizeOfCPPType(std::uint64_t),
                                           startorch::MallocType::kUndefined);
  startorch::Buffer b5 = startorch::Buffer(&v0, SizeOfCPPType(std::uint64_t),
                                           startorch::MallocType::kOptionCount);

  EXPECT_EQ(b0.GetData(), nullptr);
  EXPECT_EQ(b0.GetBytes(), 0ul);
  EXPECT_EQ(b0.GetMallocType(), startorch::MallocType::kUndefined);

  EXPECT_EQ(b1.GetData(), &v0);
  EXPECT_EQ(b1.GetBytes(), SizeOfCPPType(std::uint64_t));
  EXPECT_EQ(b1.GetMallocType(), startorch::MallocType::kHost);

  EXPECT_EQ(b2.GetData(), nullptr);
  EXPECT_EQ(b2.GetBytes(), 0ul);
  EXPECT_EQ(b2.GetMallocType(), startorch::MallocType::kUndefined);

  EXPECT_EQ(b3.GetData(), nullptr);
  EXPECT_EQ(b3.GetBytes(), 0ul);
  EXPECT_EQ(b3.GetMallocType(), startorch::MallocType::kUndefined);

  EXPECT_EQ(b4.GetData(), nullptr);
  EXPECT_EQ(b4.GetBytes(), 0ul);
  EXPECT_EQ(b4.GetMallocType(), startorch::MallocType::kUndefined);

  EXPECT_EQ(b5.GetData(), nullptr);
  EXPECT_EQ(b5.GetBytes(), 0ul);
  EXPECT_EQ(b5.GetMallocType(), startorch::MallocType::kUndefined);
}

TEST(BufferTest, OperatorBoolTest) {
  std::uint64_t v0 = 0ul;

  startorch::Buffer b0;
  startorch::Buffer b1 = startorch::Buffer(&v0, SizeOfCPPType(std::uint64_t),
                                           startorch::MallocType::kHost);
  startorch::Buffer b2 = startorch::Buffer(
      nullptr, SizeOfCPPType(std::uint64_t), startorch::MallocType::kHost);
  startorch::Buffer b3 =
      startorch::Buffer(&v0, 0ul, startorch::MallocType::kHost);
  startorch::Buffer b4 = startorch::Buffer(&v0, SizeOfCPPType(std::uint64_t),
                                           startorch::MallocType::kUndefined);
  startorch::Buffer b5 = startorch::Buffer(&v0, SizeOfCPPType(std::uint64_t),
                                           startorch::MallocType::kOptionCount);

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

  startorch::Buffer b0(&v0, SizeOfCPPType(std::uint8_t),
                       startorch::MallocType::kPinned);
  const startorch::Buffer b1(&v1, SizeOfCPPType(std::uint8_t),
                             startorch::MallocType::kPinned);

  EXPECT_EQ(*b0.GetData<uint8_t>(), v0);
  EXPECT_EQ(*b1.GetData<uint8_t>(), v1);
}

TEST(BufferTest, GetBytesTest) {
  std::uint64_t v0 = 0;
  std::uint64_t v1 = 2048ul;

  startorch::Buffer b0(&v0, v1, startorch::MallocType::kHost);

  EXPECT_EQ(b0.GetBytes(), v1);
}

TEST(BufferTest, GetMallocTypeTest) {
  std::uint8_t v0 = 0;

  startorch::MallocType t0 = startorch::MallocType::kDevice;
  startorch::MallocType t1 = startorch::MallocType::kDevice;
  startorch::MallocType t2 = startorch::MallocType::kDevice;
  startorch::MallocType t3 = startorch::MallocType::kDevice;

  startorch::Buffer b0(&v0, 64ul, t0);
  startorch::Buffer b1(&v0, 64ul, t1);
  startorch::Buffer b2(&v0, 64ul, t2);
  startorch::Buffer b3(&v0, 64ul, t3);

  EXPECT_EQ(b0.GetMallocType(), t0);
  EXPECT_EQ(b1.GetMallocType(), t1);
  EXPECT_EQ(b2.GetMallocType(), t2);
  EXPECT_EQ(b3.GetMallocType(), t3);
}

TEST(BufferTest, IsNullTest) {
  std::uint64_t v0 = 0ul;

  startorch::Buffer b0;
  startorch::Buffer b1 = startorch::Buffer(&v0, SizeOfCPPType(std::uint64_t),
                                           startorch::MallocType::kHost);
  startorch::Buffer b2 = startorch::Buffer(
      nullptr, SizeOfCPPType(std::uint64_t), startorch::MallocType::kHost);
  startorch::Buffer b3 =
      startorch::Buffer(&v0, 0ul, startorch::MallocType::kHost);
  startorch::Buffer b4 = startorch::Buffer(&v0, SizeOfCPPType(std::uint64_t),
                                           startorch::MallocType::kUndefined);
  startorch::Buffer b5 = startorch::Buffer(&v0, SizeOfCPPType(std::uint64_t),
                                           startorch::MallocType::kUndefined);

  EXPECT_EQ(b0.IsNull(), true);
  EXPECT_EQ(b1.IsNull(), false);
  EXPECT_EQ(b2.IsNull(), true);
  EXPECT_EQ(b3.IsNull(), true);
  EXPECT_EQ(b4.IsNull(), true);
}

TEST(AllocatorTest, ConstructorTest) {
  startorch::Allocator a0;
  startorch::Allocator a1(1_MB, startorch::MallocType::kHost);
  startorch::Allocator a2(1_MB, startorch::MallocType::kDevice);
  startorch::Allocator a3(1_MB, startorch::MallocType::kPinned);
  startorch::Allocator a4(1_MB, startorch::MallocType::kUnified);
  startorch::Allocator a5(0_MB, startorch::MallocType::kHost);
  startorch::Allocator a6(1_MB, startorch::MallocType::kUndefined);
  startorch::Allocator a7(1_MB, startorch::MallocType::kOptionCount);

  EXPECT_EQ(a0.GetAlignedSize(), 0ul);
  EXPECT_EQ(a0.GetType(), startorch::MallocType::kUndefined);
  EXPECT_EQ(a0.IsNull(), true);

  EXPECT_GT(a1.GetAlignedSize(), 0ul);
  EXPECT_EQ(a1.GetType(), startorch::MallocType::kHost);
  EXPECT_EQ(a1.IsNull(), false);

  EXPECT_EQ(a2.GetType(), startorch::MallocType::kDevice);
  EXPECT_EQ(a3.GetType(), startorch::MallocType::kPinned);
  EXPECT_EQ(a4.GetType(), startorch::MallocType::kUnified);

  EXPECT_EQ(a5.GetAlignedSize(), 0ul);
  EXPECT_EQ(a6.GetType(), startorch::MallocType::kUndefined);
  EXPECT_EQ(a7.GetType(), startorch::MallocType::kUndefined);
}

TEST(AllocatorTest, OperatorBoolTest) {
  startorch::Allocator a0;
  startorch::Allocator a1(1_MB, startorch::MallocType::kHost);

  EXPECT_EQ(static_cast<bool>(a0), false);
  EXPECT_EQ(!a0, true);

  EXPECT_EQ(static_cast<bool>(a1), true);
  EXPECT_EQ(!a1, false);
}

TEST(AllocatorTest, GetBufferTest) {
  startorch::Allocator a0;
  startorch::Allocator a1(1_MB, startorch::MallocType::kOptionCount);
  startorch::Allocator a2(1_MB, startorch::MallocType::kHost);
  const startorch::Allocator a3(1_MB, startorch::MallocType::kHost);

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
  startorch::Allocator a0(1_MB, startorch::MallocType::kHost);
  EXPECT_EQ(a0.GetOffset(), 0ul);

  startorch::Buffer b0 = a0.NewBuffer(1_KB);
  EXPECT_GT(a0.GetOffset(), 0ul);
  EXPECT_EQ(a0.GetOffset(), 1_KB);
}

TEST(AllocatorTest, GetAlignedSizeTest) {
  startorch::Allocator a0(1_MB, startorch::MallocType::kHost);
  startorch::Allocator a1(1_MB, startorch::MallocType::kDevice);
  startorch::Allocator a2(1_MB, startorch::MallocType::kPinned);

  EXPECT_EQ(a0.GetAlignedSize(), 64ul);
  EXPECT_EQ(a1.GetAlignedSize(), 256ul);
  EXPECT_EQ(a2.GetAlignedSize(), 4096ul);
}

TEST(AllocatorTest, GetTypeTest) {
  startorch::Allocator a0(1_MB, startorch::MallocType::kHost);
  startorch::Allocator a1(1_MB, startorch::MallocType::kDevice);

  EXPECT_EQ(a0.GetType(), startorch::MallocType::kHost);
  EXPECT_EQ(a1.GetType(), startorch::MallocType::kDevice);
}

TEST(AllocatorTest, IsNullTest) {
  startorch::Allocator a0;
  startorch::Allocator a1(1_MB, startorch::MallocType::kHost);
  startorch::Allocator a2(0_MB, startorch::MallocType::kHost);

  EXPECT_EQ(a0.IsNull(), true);
  EXPECT_EQ(a1.IsNull(), false);
  EXPECT_EQ(a2.IsNull(), true);
}

TEST(AllocatorTest, NewBufferTest) {
  startorch::Allocator a0(1_MB, startorch::MallocType::kHost);
  EXPECT_EQ(a0.GetAlignedSize(), 64ul);
  EXPECT_EQ(a0.GetOffset(), 0ul);

  startorch::Buffer b0 = a0.NewBuffer(10ul);
  EXPECT_NE(b0.GetData(), nullptr);
  EXPECT_EQ(b0.GetBytes(), 10ul);
  EXPECT_EQ(b0.GetMallocType(), startorch::MallocType::kHost);
  EXPECT_EQ(a0.GetOffset(), 64ul);

  startorch::Buffer b1 = a0.NewBuffer(100ul);
  EXPECT_NE(b1.GetData(), nullptr);
  EXPECT_EQ(b1.GetBytes(), 100ul);
  EXPECT_EQ(a0.GetOffset(), 192ul);

  startorch::Buffer b2 = a0.NewBuffer(0ul);
  EXPECT_EQ(b2.IsNull(), true);

  startorch::Allocator a1(1_MB, startorch::MallocType::kDevice);
  EXPECT_EQ(a1.GetAlignedSize(), 256ul);
  EXPECT_EQ(a1.GetOffset(), 0ul);

  startorch::Buffer b3 = a1.NewBuffer(100ul);
  EXPECT_NE(b3.GetData(), nullptr);
  EXPECT_EQ(b3.GetBytes(), 100ul);
  EXPECT_EQ(a1.GetOffset(), 256ul);

  startorch::Allocator a2(1_MB, startorch::MallocType::kUnified);
  EXPECT_EQ(a2.GetAlignedSize(), 256ul);
  EXPECT_EQ(a2.GetOffset(), 0ul);

  startorch::Buffer b4 = a2.NewBuffer(50ul);
  EXPECT_NE(b4.GetData(), nullptr);
  EXPECT_EQ(b4.GetBytes(), 50ul);
  EXPECT_EQ(a2.GetOffset(), 256ul);
}

TEST(AllocatorTest, DeleteBufferTest) {
  startorch::Allocator a0(1_MB, startorch::MallocType::kHost);

  startorch::Buffer b0 = a0.NewBuffer(10ul);
  startorch::Buffer b1 = a0.NewBuffer(100ul);
  startorch::Buffer b2 = a0.NewBuffer(50ul);

  EXPECT_EQ(a0.GetOffset(), 256ul);

  a0.DeleteBuffer(b1);
  EXPECT_EQ(a0.GetOffset(), 256ul);

  a0.DeleteBuffer(b2);
  EXPECT_EQ(a0.GetOffset(), 192ul);

  a0.DeleteBuffer(b1);
  EXPECT_EQ(a0.GetOffset(), 64ul);

  a0.DeleteBuffer(b0);
  EXPECT_EQ(a0.GetOffset(), 0ul);

  startorch::Allocator a1(1_MB, startorch::MallocType::kDevice);

  startorch::Buffer b3 = a1.NewBuffer(10ul);
  startorch::Buffer b4 = a1.NewBuffer(100ul);
  startorch::Buffer b5 = a1.NewBuffer(50ul);

  EXPECT_EQ(a1.GetOffset(), 768ul);

  a1.DeleteBuffer(b5);
  EXPECT_EQ(a1.GetOffset(), 512ul);

  a1.DeleteBuffer(b4);
  EXPECT_EQ(a1.GetOffset(), 256ul);

  a1.DeleteBuffer(b3);
  EXPECT_EQ(a1.GetOffset(), 0ul);
}

TEST(AllocatorTest, GetNullTest) {
  startorch::Allocator& a0 = startorch::Allocator::GetNull();

  EXPECT_EQ(a0.IsNull(), true);
  EXPECT_EQ(a0.GetAlignedSize(), 0ul);
  EXPECT_EQ(a0.GetType(), startorch::MallocType::kUndefined);
}
