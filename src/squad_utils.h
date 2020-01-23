#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct SquadExample {
  std::string qas_id;
  std::string question_text;
  std::string paragraph_text;
  std::string orig_answer_text;
  int start_position;
  int end_position;
  bool is_impossible;
  SquadExample(std::string qas_id, std::string question_text,
               std::string paragraph_text, std::string orig_answer_text = "",
               int start_position = -1, int end_position = -1,
               bool is_impossible = false)
      : qas_id(qas_id), question_text(question_text),
        paragraph_text(paragraph_text), orig_answer_text(orig_answer_text),
        start_position(start_position), end_position(end_position),
        is_impossible(is_impossible){};
};

struct InputFeatures {
  size_t unique_id;
  size_t example_index;
  int doc_span_index;
  int tok_start_to_orig_index;
  int tok_end_to_orig_index;
  bool token_is_max_context;
  std::vector<std::string> tokens;
  std::vector<int64_t> input_ids;
  std::vector<int64_t> input_mask;
  std::vector<int64_t> segments_ids;
  int paragraph_len;
  int start_position;
  int end_position;
  bool is_impossible;
  std::vector<int64_t> pmask;
};

std::vector<SquadExample> read_squad_examples(std::ifstream &input_file,
                                              bool is_training);
