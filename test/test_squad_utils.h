#pragma once

#include <filesystem>

#include "catch2/catch.hpp"
#include "src/squad_utils.h"
#include "utils.h"

using namespace hflt;

bool file_exists(const char *fp) {
  std::ifstream fd(fp);
  return fd.good();
}

size_t file_size(const char *fp) {
  std::filesystem::path p(fp);
  return std::filesystem::file_size(p);
}

void file_test(const char *fp, const bool remove_file = false) {
  REQUIRE(file_exists(fp) == true);
  REQUIRE(file_size(fp) > 0);
  if (remove_file)
    remove(fp);
}

// Number of examples from transformers script output,
// ../data/SQuAD/SQuAD_dev2/inputs_ids.csv
const size_t EXPECTED_V1_EXAMPLES = 10866;
const size_t EXPECTED_V2_EXAMPLES = 12272;

const char *squad_v1_path = "data/SQuAD/dev-v1.1.json";
const char *squad_v2_path = "data/SQuAD/dev-v2.0.json";

TEST_CASE("Test Utils on SQuAD2", "[squad]") {
  file_test(squad_v2_path);
  std::ifstream squad_file(squad_v2_path);
  std::vector<SquadExample> examples = read_squad_examples(squad_file, true);
  for (auto &example : examples)
    REQUIRE(example.paragraph_text.size() > 0);
  REQUIRE(examples.size() == EXPECTED_V2_EXAMPLES);
}

TEST_CASE("Test Utils on SQuAD1", "[squad]") {
  file_test(squad_v1_path);
  std::ifstream squad_file(squad_v1_path);
  std::vector<SquadExample> examples = read_squad_examples(squad_file, true);
  for (auto &example : examples)
    REQUIRE(example.paragraph_text.size() > 0);
  REQUIRE(examples.size() == EXPECTED_V1_EXAMPLES);
}

TEST_CASE("Check Output from Huggingface Utils", "[squad]") {
  std::string filepath = "data/SQuAD/SQuAD_dev2/input_ids.csv";
  std::vector<std::vector<int>> data = readConversionCheckCsv(filepath);
  REQUIRE(data.size() == EXPECTED_V2_EXAMPLES);
}
