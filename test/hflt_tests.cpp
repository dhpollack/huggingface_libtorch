#include "gtest/gtest.h"
#include "test_squad_utils.h"
#include "test_transformer_config.h"
#include "test_transformer_example.h"
#include "test_dataset_and_processors.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
