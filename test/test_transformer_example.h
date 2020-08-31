#pragma once

#include "gtest/gtest.h"
#include "src/transformer_example.h"

TEST(transformerexampleTest, create_transformer_example) {
  std::string text_a("this is a test sentence without a second text");
  std::string text_b("this is text_b");
  TransformerExample ex1 = {"example1", text_a, "", "0"};
  EXPECT_EQ(ex1.text_a, text_a);
  TransformerExample ex2 = {"example2", text_a, text_b, "0"};
  EXPECT_EQ(ex2.text_a, text_a);
  EXPECT_EQ(ex2.text_b, text_b);
  EXPECT_NE(ex1.guid, ex2.guid);
}
