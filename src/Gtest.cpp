#include <iostream>
#include <gtest/gtest.h>
#include "search_modules/strategy_providers/DDPRLabeler.h"
using std::shared_ptr;

// Test DDPRLabeler
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

class DDPRLabelerTest: public testing::Test {
  protected:
  shared_ptr<DDPRLabeler> labeler1;

  void SetUp() override{
    labeler1 = std::make_shared<DDPRLabeler>(8, 1, make_pair(-5, 5), "", "", "", "");
  }
};


TEST_F(DDPRLabelerTest, BasicAssertions) {
  // 
}