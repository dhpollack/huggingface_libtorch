#include "processors.h"

using namespace std;

vector<TransformerExample> readGenericJsonFile(const string &filepath) {
  // assumes a json in the format {..., "data": [{data_obj},...]}, where
  // data_obj = {"guid": "guid_as_string", "text_a": "some text", "text_b":
  // "more text", "label": "label0"}
  vector<TransformerExample> examples;
  ifstream ifs(filepath);
  if (!ifs.is_open()) {
    cerr << "unable to open generic json datafile" << endl;
  } else {
    json j;
    ifs >> j;
    for (auto &item : j["data"]) {
      string guid = item["guid"].get<string>();
      string text_a = item["text_a"].get<string>();
      string text_b = item["text_b"].get<string>();
      string label = item["label"].get<string>();
      examples.emplace_back(guid, text_a, text_b, label);
    }
  }
  return examples;
}
vector<TransformerExample> readSST2CsvFile(const string &filepath) {
  // This function assumes the csv file is in the format `<sentence>\t<label>`
  vector<TransformerExample> examples;
  string line;
  ifstream ifs(filepath);
  if (!ifs.is_open()) {
    return examples;
  }
  using tokenizer = boost::tokenizer<boost::char_separator<char>>;
  boost::char_separator<char> sep("\t");
  size_t i = 0;
  while (getline(ifs, line)) {
    tokenizer tokens(line, sep);
    tokenizer::iterator tok_iter = tokens.begin();
    string sentence = *tok_iter;
    ++tok_iter;
    int64_t label = 0;
    istringstream i_str(*tok_iter);
    if (!(i_str >> label)) {
      // this should exclude the header of the csv
      // cout << "found invalid example: " << line << endl;
      continue;
    }
    TransformerExample ex = {std::to_string(i), sentence, "",
                             std::to_string(label)};
    examples.push_back(ex);
    ++i;
  }
  return examples;
}
