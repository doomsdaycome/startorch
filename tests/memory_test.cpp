#include <cstdint>

#include <gtest/gtest.h>

#include "darkside/memory/allocator.hpp"
#include "darkside/memory/buffer.hpp"
#include "startorch/common/types.hpp"

namespace darkside {

TEST(BufferTest, DefaultConstructorTest) {
  Buffer b0;

  EXPECT_EQ(b0.GetData(), nullptr);
  EXPECT_EQ(b0.GetBytes(), 0ul);
  EXPECT_EQ(b0.GetMallocType(), startorch::MallocType::kUndefined);
}

TEST(BufferTest, ParameterizedConstructorTest) {
  std::uint8_t d0 = 42;
  void *p0 = static_cast<void *>(&d0);
  std::uint64_t s0 = 1024ul;
  startorch::MallocType t0 = startorch::MallocType::kHost;

  Buffer b0(p0, s0, t0);

  EXPECT_EQ(b0.GetData(), p0);
  EXPECT_EQ(b0.GetBytes(), s0);
  EXPECT_EQ(b0.GetMallocType(), t0);
}

TEST(BufferTest, MoveConstructorTest) {
  std::uint8_t d0 = 100;
  void *p0 = static_cast<void *>(&d0);
  std::uint64_t s0 = 512ul;
  startorch::MallocType t0 = startorch::MallocType::kDevice;

  Buffer b0(p0, s0, t0);
  Buffer b1(std::move(b0));

  EXPECT_EQ(b1.GetData(), p0);
  EXPECT_EQ(b1.GetBytes(), s0);
  EXPECT_EQ(b1.GetMallocType(), t0);
}

TEST(BufferTest, MoveAssignmentOperatorTest) {
  std::uint8_t d0 = 200;
  void *p0 = static_cast<void *>(&d0);
  std::uint64_t s0 = 256ul;
  startorch::MallocType t0 = startorch::MallocType::kUnified;

  Buffer b0(p0, s0, t0);
  Buffer b1;
  b1 = std::move(b0);

  EXPECT_EQ(b1.GetData(), p0);
  EXPECT_EQ(b1.GetBytes(), s0);
  EXPECT_EQ(b1.GetMallocType(), t0);
}

TEST(BufferTest, ExplicitOperatorBoolTest) {
  Buffer b0;
  EXPECT_FALSE(static_cast<bool>(b0));

  std::uint8_t d0 = 1;
  void *p0 = static_cast<void *>(&d0);
  Buffer b1(p0, 16ul, startorch::MallocType::kHost);
  EXPECT_TRUE(static_cast<bool>(b1));
}

TEST(BufferTest, OperatorNotTest) {
  Buffer b0;
  EXPECT_TRUE(!b0);

  std::uint8_t d0 = 1;
  void *p0 = static_cast<void *>(&d0);
  Buffer b1(p0, 16ul, startorch::MallocType::kHost);
  EXPECT_FALSE(!b1);
}

TEST(BufferTest, GetDataTest) {
  std::uint8_t d0 = 7;
  void *p0 = static_cast<void *>(&d0);
  Buffer b0(p0, 8ul, startorch::MallocType::kPinned);

  EXPECT_EQ(b0.GetData(), p0);
}

TEST(BufferTest, GetConstDataTest) {
  std::uint8_t d0 = 9;
  void *p0 = static_cast<void *>(&d0);
  const Buffer b0(p0, 8ul, startorch::MallocType::kPinned);

  EXPECT_EQ(b0.GetData(), p0);
}

TEST(BufferTest, GetBytesTest) {
  std::uint8_t d0 = 0;
  void *p0 = static_cast<void *>(&d0);
  std::uint64_t s0 = 2048ul;
  Buffer b0(p0, s0, startorch::MallocType::kHost);

  EXPECT_EQ(b0.GetBytes(), s0);
}

TEST(BufferTest, GetMallocTypeTest) {
  std::uint8_t d0 = 0;
  void *p0 = static_cast<void *>(&d0);
  startorch::MallocType t0 = startorch::MallocType::kDevice;
  Buffer b0(p0, 64ul, t0);

  EXPECT_EQ(b0.GetMallocType(), t0);
}

TEST(BufferTest, IsNullTest) {
  Buffer b0;
  EXPECT_TRUE(b0.IsNull());

  std::uint8_t d0 = 123;
  void *p0 = static_cast<void *>(&d0);
  Buffer b1(p0, 128ul, startorch::MallocType::kHost);
  EXPECT_FALSE(b1.IsNull());
}

TEST(AllocatorTest, DefaultConstructorTest) {
  Allocator a0;
  EXPECT_EQ(a0.GetOffset(), 0ul);
  EXPECT_EQ(a0.GetAlignedSize(), 0ul);
  EXPECT_TRUE(a0.IsNull());
}

TEST(AllocatorTest, ParameterizedConstructorTest) {
  std::uint64_t s0 = 4096ul;
  startorch::MallocType t0 = startorch::MallocType::kHost;

  Allocator a0(s0, t0);

  EXPECT_FALSE(a0.IsNull());
  EXPECT_EQ(a0.GetOffset(), 0ul);
  EXPECT_NE(a0.GetAlignedSize(), 0ul);
  EXPECT_EQ(a0.GetBuffer().GetBytes(), s0);
  EXPECT_EQ(a0.GetBuffer().GetMallocType(), t0);
}

TEST(AllocatorTest, MoveConstructorTest) {
  std::uint64_t s0 = 2048ul;
  startorch::MallocType t0 = startorch::MallocType::kDevice;

  Allocator a0(s0, t0);
  std::uint64_t l0 = a0.GetAlignedSize();

  Allocator a1(std::move(a0));

  EXPECT_FALSE(a1.IsNull());
  EXPECT_EQ(a1.GetBuffer().GetBytes(), s0);
  EXPECT_EQ(a1.GetBuffer().GetMallocType(), t0);
  EXPECT_EQ(a1.GetAlignedSize(), l0);
}

TEST(AllocatorTest, MoveAssignmentOperatorTest) {
  std::uint64_t s0 = 1024ul;
  startorch::MallocType t0 = startorch::MallocType::kPinned;

  Allocator a0(s0, t0);
  std::uint64_t l0 = a0.GetAlignedSize();

  Allocator a1;
  a1 = std::move(a0);

  EXPECT_FALSE(a1.IsNull());
  EXPECT_EQ(a1.GetBuffer().GetBytes(), s0);
  EXPECT_EQ(a1.GetBuffer().GetMallocType(), t0);
  EXPECT_EQ(a1.GetAlignedSize(), l0);
}

TEST(AllocatorTest, ExplicitOperatorBoolTest) {
  Allocator a0;
  EXPECT_FALSE(static_cast<bool>(a0));

  Allocator a1(1024ul, startorch::MallocType::kHost);
  EXPECT_TRUE(static_cast<bool>(a1));
}

TEST(AllocatorTest, OperatorNotTest) {
  Allocator a0;
  EXPECT_TRUE(!a0);

  Allocator a1(1024ul, startorch::MallocType::kHost);
  EXPECT_FALSE(!a1);
}

TEST(AllocatorTest, GetBufferTest) {
  std::uint64_t s0 = 512ul;
  startorch::MallocType t0 = startorch::MallocType::kUnified;

  Allocator a0(s0, t0);
  Buffer &b0 = a0.GetBuffer();

  EXPECT_EQ(b0.GetBytes(), s0);
  EXPECT_EQ(b0.GetMallocType(), t0);
}

TEST(AllocatorTest, GetConstBufferTest) {
  std::uint64_t s0 = 512ul;
  startorch::MallocType t0 = startorch::MallocType::kUnified;

  const Allocator a0(s0, t0);
  const Buffer &b0 = a0.GetBuffer();

  EXPECT_EQ(b0.GetBytes(), s0);
  EXPECT_EQ(b0.GetMallocType(), t0);
}

TEST(AllocatorTest, GetOffsetTest) {
  Allocator a0(1024ul, startorch::MallocType::kHost);
  EXPECT_EQ(a0.GetOffset(), 0ul);

  a0.NewBuffer(10ul);
  EXPECT_GT(a0.GetOffset(), 0ul);
}

TEST(AllocatorTest, GetAlignedSizeTest) {
  Allocator a_host(1024ul, startorch::MallocType::kHost);
  EXPECT_EQ(a_host.GetAlignedSize(), 64ul);

  Allocator a_device(1024ul, startorch::MallocType::kDevice);
  EXPECT_EQ(a_device.GetAlignedSize(), 256ul);

  Allocator a_pinned(1024ul, startorch::MallocType::kPinned);
  EXPECT_EQ(a_pinned.GetAlignedSize(), 4096ul);

  Allocator a_unified(1024ul, startorch::MallocType::kUnified);
  EXPECT_EQ(a_unified.GetAlignedSize(), 256ul);
}

TEST(AllocatorTest, IsNullTest) {
  Allocator a0;
  EXPECT_TRUE(a0.IsNull());

  Allocator a1(1024ul, startorch::MallocType::kHost);
  EXPECT_FALSE(a1.IsNull());
}

TEST(AllocatorTest, NewBufferTest) {
  std::uint64_t s0 = 1024ul;
  Allocator a0(s0, startorch::MallocType::kHost);

  std::uint64_t l0 = a0.GetAlignedSize();

  Buffer b0 = a0.NewBuffer(0ul);
  EXPECT_TRUE(b0.IsNull());
  EXPECT_EQ(a0.GetOffset(), 0ul);

  Buffer b1 = a0.NewBuffer(10ul);
  EXPECT_FALSE(b1.IsNull());
  EXPECT_EQ(b1.GetBytes(), 10ul);

  std::uint64_t e0 = (10ul + l0 - 1ul) & ~(l0 - 1ul);
  EXPECT_EQ(a0.GetOffset(), e0);

  Buffer b2 = a0.NewBuffer(10ul);
  EXPECT_FALSE(b2.IsNull());

  std::uint64_t e1 = e0 + ((10ul + l0 - 1ul) & ~(l0 - 1ul));
  EXPECT_EQ(a0.GetOffset(), e1);

  uintptr_t p0 = reinterpret_cast<uintptr_t>(b1.GetData());
  uintptr_t p1 = reinterpret_cast<uintptr_t>(b2.GetData());
  EXPECT_EQ(p1 - p0, l0);

  Buffer b3 = a0.NewBuffer(a0.GetBuffer().GetBytes());
  EXPECT_TRUE(b3.IsNull());
  EXPECT_EQ(a0.GetOffset(), e1);
}

TEST(AllocatorTest, DeleteBufferTest) {
  std::uint64_t s0 = 1024ul;
  Allocator a0(s0, startorch::MallocType::kHost);

  Buffer b0 = a0.NewBuffer(10ul);
  Buffer b1 = a0.NewBuffer(10ul);

  std::uint64_t o0 = a0.GetOffset();

  a0.DeleteBuffer(b0);
  EXPECT_EQ(a0.GetOffset(), o0);

  a0.DeleteBuffer(b1);
  EXPECT_LT(a0.GetOffset(), o0);

  a0.DeleteBuffer(b0);
  EXPECT_EQ(a0.GetOffset(), 0ul);
}

} // namespace darkside
