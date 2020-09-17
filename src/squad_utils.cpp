#include "squad_utils.h"

using namespace std;

namespace hflt {

vector<SquadExample> readSquadExamples(const string &input_path) {
  ifstream ifs(input_path);
  return read_squad_examples(ifs, true);
}

/** Read SQuAD json into a vector of SquadExample structs
 *
 * based on:
 * https://github.com/huggingface/transformers/blob/master/src/transformers/data/processors/squad.py#L625
 *
 */
vector<SquadExample> read_squad_examples(ifstream &input_file,
                                         bool is_training) {
  json j;
  vector<SquadExample> examples;
  if (!input_file.is_open()) {
    cerr << "unable to open input file" << endl;
  } else {
    input_file >> j;

    for (auto &entry : j["data"]) {
      for (auto &paragraph : entry["paragraphs"]) {
        auto paragraph_text = paragraph["context"].get<string>();
        for (auto &qa : paragraph["qas"]) {
          auto qas_id = qa["id"].get<string>();
          auto question_text = qa["question"].get<string>();
          int start_position = -1;
          string orig_answer_text;
          auto is_impossible = qa.contains("is_impossible")
                                   ? qa["is_impossible"].get<bool>()
                                   : false;
          if (is_training) {
            if (!is_impossible) {
              orig_answer_text = qa["answers"][0]["text"].get<string>();
              start_position = qa["answers"][0]["answer_start"].get<int>();
            } // default values set to else statement
          }
          // huggingface trims paragraph_text and removes extraneous whitespace
          // to `doc_tokens`
          SquadExample example(qas_id, question_text, paragraph_text,
                               orig_answer_text, start_position, -1,
                               is_impossible);
          examples.emplace_back(example);
        } // end qa
      }   // end paragraph
    }     // end element
  }       // end else
  return examples;
}

template <typename T>
vector<pair<size_t, size_t>>
add_tokens_to_examples(vector<SquadExample> &examples, T tokenizer_, long msl_,
                       size_t doc_stride, size_t max_query_length,
                       size_t num_special_tokens) {
  // TODO find answer and use variables related to this
  // const size_t doc_stride = 128;
  // const size_t max_query_length = 64;
  // const size_t num_special_tokens = 3;
  vector<pair<size_t, size_t>> p_spans;
  size_t i = 0;
  for (auto &ex : examples) {
    if (ex.is_impossible) {
      // cerr << "skipping impossible question: " << ex.question_text << endl;
      // continue;
    }
    string preanswer_text = ex.paragraph_text.substr(0, ex.start_position);
    vector<int> pa_tokens = tokenizer_.template tokenize<int>(preanswer_text);
    long answer_start __attribute__((unused)) = pa_tokens.size();
    vector<int> q_tokens = tokenizer_.template tokenize<int>(ex.question_text);
    if (q_tokens.size() > max_query_length) {
      q_tokens.resize(max_query_length);
    }
    // insert special tokens into question
    // q_tokens.insert(q_tokens.begin(), tokenizer_.cls_token_id());
    // q_tokens.emplace_back(tokenizer_.sep_token_id());
    vector<int> p_tokens = tokenizer_.template tokenize<int>(ex.paragraph_text);
    vector<int> a_tokens =
        tokenizer_.template tokenize<int>(ex.orig_answer_text);
    // insert into example
    ex.q_tokens = q_tokens;
    ex.p_tokens = p_tokens;
    ex.a_tokens = a_tokens;
    size_t query_len = q_tokens.size();
    size_t j = 0;
    size_t span_max_len = msl_ - query_len - num_special_tokens;
    while (j * doc_stride < p_tokens.size()) {
      size_t begin_pos = j * doc_stride;
      size_t end_pos = min(begin_pos + span_max_len, p_tokens.size());
      bool is_last_span = (end_pos == p_tokens.size());
      // create span of paragraph tokens
      auto span_start __attribute__((unused)) = p_tokens.begin() + begin_pos;
      auto span_end __attribute__((unused)) = p_tokens.begin() + end_pos;
      ex.p_spans.emplace_back(begin_pos, end_pos);
      p_spans.emplace_back(i, j);
      ++j;
      if (is_last_span) {
        break;
      }
    }
    ++i;
  }
  return p_spans;
}

template vector<pair<size_t, size_t>>
add_tokens_to_examples(vector<SquadExample> &, TokenizerBase, long, size_t,
                       size_t, size_t);
template vector<pair<size_t, size_t>>
add_tokens_to_examples(vector<SquadExample> &, TokenizerAlbert, long, size_t,
                       size_t, size_t);

}; // namespace hflt
