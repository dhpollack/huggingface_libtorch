#pragma once

#include "catch2/catch.hpp"
#include "src/transformer_example.h"

using namespace hflt;

TEST_CASE("Transformer Example Test", "[examples]") {
  std::string text_a("this is a test sentence without a second text");
  std::string text_b("this is text_b");
  TransformerExample ex1 = {"example1", text_a, "", "0"};
  REQUIRE(ex1.text_a == text_a);
  TransformerExample ex2 = {"example2", text_a, text_b, "0"};
  REQUIRE(ex2.text_a == text_a);
  REQUIRE(ex2.text_b == text_b);
  REQUIRE(ex1.guid != ex2.guid);
}
