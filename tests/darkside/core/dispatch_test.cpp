#include "darkside/core/dispatch.hpp"

#include <gtest/gtest.h>

#include "darkside/core/traits.hpp"
#include "darkside/core/types.hpp"
#include "darkside/macros/mapping.hpp"

TEST(DispatchTest, ScalarSizeTest) {
#define TEST_DISPATCH(S)         \
  DARKSIDE_DISPATCH_SCALAR_TYPE( \
      S, { EXPECT_EQ(sizeof(scalar_t), sizeof(darkside::scalar_t<S>)); });

  DARKSIDE_FORALL_SCALAR_TYPE(TEST_DISPATCH)
#undef TEST_DISPATCH
}
