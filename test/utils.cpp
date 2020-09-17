#include "utils.h"

using namespace std;

vector<vector<int>> readConversionCheckCsv(const std::string &filepath) {
  vector<vector<int>> data;
  vector<int> row;
  string line;
  ifstream ifs(filepath);

  if (!ifs.is_open()) {
    cerr << "unable to open csv file" << endl;
    return data;
  }
  using tokenizer = boost::tokenizer<boost::char_separator<char>>;
  boost::char_separator<char> sep(",");
  while (getline(ifs, line)) {
    tokenizer tokens(line, sep);
    transform(tokens.begin(), tokens.end(), back_inserter(row),
              [](const std::string &s) { return stoi(s); });
    data.emplace_back(row);
    row.clear();
  }
  return data;
}
