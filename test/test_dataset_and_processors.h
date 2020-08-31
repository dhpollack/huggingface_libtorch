#pragma once

#include <torch/torch.h>

#include "gtest/gtest.h"
#include "src/dataset_classification.h"
#include "src/dataset_qa.h"
#include "src/processors.h"

std::string pretrained_dir = "models/sst2_trained";

TEST(datasetprocessorsTest, tokenizerbasetests) {
  using psfn =
      std::pair<std::string, std::function<std::vector<TransformerExample>(
                                 const std::string)>>;
  std::vector<psfn> dataset_paths(
      {std::make_pair("assets/sst-2-head.tsv", readSST2CsvFile),
       std::make_pair("assets/sst-2-head.json", readGenericJsonFile)});
  for (auto p : dataset_paths) {
    TransformerClassificationDS<> ds(pretrained_dir, 128, p.second, p.first);
    size_t num_examples = ds.size().value();
    auto examples = ds.examples();
    auto ex = examples[0];
    EXPECT_EQ(num_examples, 9);
    EXPECT_EQ(ex.guid, "0");
    EXPECT_EQ(ex.text_a, "it 's a charming and often affecting journey . ");
    EXPECT_EQ(ex.text_b, "");
    EXPECT_EQ(ex.label, "1");
  }
}

std::string squad_data_path("data/SQuAD/dev-v2.0.json");

TEST(datasetprocessorsTest, qatests) {
  TransformerQADS<TokenizerAlbert, SquadExample> ds(pretrained_dir, 384, readSquadExamples, squad_data_path);
  std::vector<SquadExample> examples = ds.examples();
  EXPECT_EQ(examples.size(), 11873);
  std::cout << "Squad Examples: " << examples.size() << std::endl;
  std::cout << "Squad DS size(): " << ds.size().value() << std::endl;
  
  auto dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(ds), 16);
  size_t i = 0;
  for (auto &_mb : *dl) {
    ++i;
  }
  std::cout << "Num Batches: " << i << std::endl;
}
