#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

struct SquadExample {
  string qas_id;
  string question_text;
  string paragraph_text;
  string orig_answer_text;
  int start_position;
  int end_position;
  bool is_impossible;
  SquadExample(string qas_id, string question_text, string paragraph_text,
               string orig_answer_text = "", int start_position = -1,
               int end_position = -1, bool is_impossible = false)
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
  vector<string> tokens;
  vector<int64_t> input_ids;
  vector<int64_t> input_mask;
  vector<int64_t> segments_ids;
  int paragraph_len;
  int start_position;
  int end_position;
  bool is_impossible;
  vector<int64_t> pmask;
};

vector<SquadExample> read_squad_examples(ifstream &input_file,
                                         bool is_training);
