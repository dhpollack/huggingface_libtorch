#pragma once

#include <boost/tokenizer.hpp>
#include <nlohmann/json.hpp>

#include <string>

#include <vector>

#include "transformer_example.h"

namespace hflt {

using json = nlohmann::json;

std::vector<TransformerExample>
readGenericJsonFile(const std::string &filepath);
std::vector<TransformerExample> readSST2CsvFile(const std::string &filepath);

}; // namespace hflt
