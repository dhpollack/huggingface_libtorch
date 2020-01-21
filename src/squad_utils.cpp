#include "squad_utils.h"

vector<SquadExample> read_squad_examples(ifstream &input_file, bool is_training) {
  json j;
  vector<SquadExample> examples;
  if (!input_file.is_open()) {
    return examples;
  } else {
    input_file >> j;

    for (auto &entry : j["data"]) {
      for (auto &paragraph : entry["paragraphs"]) {
        string paragraph_text = paragraph["context"].get<string>();
        for (auto &qa : paragraph["qas"]) {
          string qas_id = qa["id"].get<string>();
          string question_text = qa["question"].get<string>();
          int start_position = -1;
          string orig_answer_text("");
          bool is_impossible = false;
          if (is_training) {
            is_impossible = qa.contains("is_impossible") ? qa["is_impossible"].get<bool>() : false;
            if (!is_impossible) {
              orig_answer_text = qa["answers"][0]["text"].get<string>();
              start_position = qa["answers"][0]["answer_start"].get<int>();
            } // default values set to else statement
          }
          SquadExample example(qas_id, question_text, paragraph_text, orig_answer_text, start_position, -1, is_impossible);
          examples.push_back(example);
        }  // end qa
      }  // end paragraph
    }  // end element
  }  // end else
  return examples;
}
