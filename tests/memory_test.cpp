#include <cstdint>

#include <gtest/gtest.h>

#include "darkside/memory/allocator.hpp"
#include "darkside/memory/buffer.hpp"
#include "darkside/scalar/utilities.hpp"
#include "startorch/common/types.hpp"

using namespace darkside;
using namespace startorch;

TEST(BufferTest, ConstructorTest) {
  std::uint64_t v0 = 0ul;

  Buffer b0;
  Buffer b1 = Buffer(&v0, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b2 = Buffer(nullptr, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b3 = Buffer(&v0, 0ul, MallocType::kHost);
  Buffer b4 = Buffer(&v0, SizeOfCPPType(std::uint64_t), MallocType::kUndefined);

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
}

TEST(BufferTest, OperatorBoolTest) {
  std::uint64_t v0 = 0ul;

  Buffer b0;
  Buffer b1 = Buffer(&v0, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b2 = Buffer(nullptr, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b3 = Buffer(&v0, 0ul, MallocType::kHost);
  Buffer b4 = Buffer(&v0, SizeOfCPPType(std::uint64_t), MallocType::kUndefined);
  Buffer b5 = Buffer(&v0, SizeOfCPPType(std::uint64_t), MallocType::kUndefined);

  EXPECT_EQ(static_cast<bool>(b0), false);
  EXPECT_EQ(!b0, true);

  EXPECT_EQ(static_cast<bool>(b1), true);
  EXPECT_EQ(!b1, false);

  EXPECT_EQ(static_cast<bool>(b2), false);
  EXPECT_EQ(static_cast<bool>(b3), false);
  EXPECT_EQ(static_cast<bool>(b4), false);
}

TEST(BufferTest, GetDataTest) {
  std::uint8_t v0 = 7;
  std::uint8_t v1 = 9;

  Buffer b0(&v0, SizeOfCPPType(std::uint8_t), startorch::MallocType::kPinned);
  const Buffer b1(&v1, SizeOfCPPType(std::uint8_t),
                  startorch::MallocType::kPinned);

  EXPECT_EQ(*b0.GetData<uint8_t>(), v0);
  EXPECT_EQ(*b1.GetData<uint8_t>(), v1);
}

TEST(BufferTest, GetBytesTest) {
  std::uint64_t v0 = 0;
  std::uint64_t v1 = 2048ul;

  Buffer b0(&v0, v1, startorch::MallocType::kHost);

  EXPECT_EQ(b0.GetBytes(), v1);
}

TEST(BufferTest, GetMallocTypeTest) {
  std::uint8_t v0 = 0;

  startorch::MallocType t0 = startorch::MallocType::kDevice;
  startorch::MallocType t1 = startorch::MallocType::kDevice;
  startorch::MallocType t2 = startorch::MallocType::kDevice;
  startorch::MallocType t3 = startorch::MallocType::kDevice;

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
  Buffer b1 = Buffer(&v0, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b2 = Buffer(nullptr, SizeOfCPPType(std::uint64_t), MallocType::kHost);
  Buffer b3 = Buffer(&v0, 0ul, MallocType::kHost);
  Buffer b4 = Buffer(&v0, SizeOfCPPType(std::uint64_t), MallocType::kUndefined);
  Buffer b5 = Buffer(&v0, SizeOfCPPType(std::uint64_t), MallocType::kUndefined);

  EXPECT_EQ(b0.IsNull(), true);
  EXPECT_EQ(b1.IsNull(), false);
  EXPECT_EQ(b2.IsNull(), true);
  EXPECT_EQ(b3.IsNull(), true);
  EXPECT_EQ(b4.IsNull(), true);
}

TEST(AllocatorTest, ConstructorTest) {}
