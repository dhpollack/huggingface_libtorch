#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <torch/types.h>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "tokenizer_albert.h"
#include "tokenizer_base.h"
#include "transformer_example.h"

using json = nlohmann::json;

struct SquadExample {
  std::string qas_id;
  std::string question_text;
  std::string paragraph_text;
  std::string orig_answer_text;
  int start_position;
  int end_position;
  bool is_impossible;
  std::vector<int> q_tokens;
  std::vector<int> p_tokens;
  std::vector<int> a_tokens;
  std::vector<std::pair<size_t, size_t>> p_spans;
  SquadExample(std::string &qas_id, std::string &question_text,
               std::string &paragraph_text, std::string &orig_answer_text,
               int start_position, int end_position, bool is_impossible)
      : qas_id(qas_id), question_text(question_text),
        paragraph_text(paragraph_text), orig_answer_text(orig_answer_text),
        start_position(start_position), end_position(end_position),
        is_impossible(is_impossible){};
};

std::vector<SquadExample> readSquadExamples(const std::string &input_path);
std::vector<SquadExample> read_squad_examples(std::ifstream &input_file,
                                              bool is_training);
template <typename TokenizerType>
std::vector<std::pair<size_t, size_t>>
add_tokens_to_examples(std::vector<SquadExample> &examples,
                       TokenizerType tokenizer_, long msl_);
