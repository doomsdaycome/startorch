#include "darkside/scalar/dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdfloat>

#include "darkside/common/types.hpp"

using namespace std;

TEST(DispatchTest, ScalarSizeTest) {
  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kBool, C,
                                { EXPECT_EQ(sizeof(C), sizeof(bool)); })

  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kUnsignedInt8, C,
                                { EXPECT_EQ(sizeof(C), sizeof(uint8_t)); })
  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kUnsignedInt16, C,
                                { EXPECT_EQ(sizeof(C), sizeof(uint16_t)); })
  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kUnsignedInt32, C,
                                { EXPECT_EQ(sizeof(C), sizeof(uint32_t)); })
  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kUnsignedInt64, C,
                                { EXPECT_EQ(sizeof(C), sizeof(uint64_t)); })

  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kInt8, C,
                                { EXPECT_EQ(sizeof(C), sizeof(int8_t)); })
  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kInt16, C,
                                { EXPECT_EQ(sizeof(C), sizeof(int16_t)); })
  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kInt32, C,
                                { EXPECT_EQ(sizeof(C), sizeof(int32_t)); })
  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kInt64, C,
                                { EXPECT_EQ(sizeof(C), sizeof(int64_t)); })

  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kFloat16, C,
                                { EXPECT_EQ(sizeof(C), sizeof(float16_t)); })
  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kFloat32, C,
                                { EXPECT_EQ(sizeof(C), sizeof(float32_t)); })
  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kFloat64, C,
                                { EXPECT_EQ(sizeof(C), sizeof(float64_t)); })
  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kFloat128, C,
                                { EXPECT_EQ(sizeof(C), sizeof(float128_t)); })

  DARKSIDE_DISPATCH_SCALAR_TYPE(darkside::ScalarType::kBrainFloat16, C,
                                { EXPECT_EQ(sizeof(C), sizeof(bfloat16_t)); })

#undef TEST_DISPATCH
}
