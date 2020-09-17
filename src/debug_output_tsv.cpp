#include "debug_output_tsv.h"
#include "hflt.h"
#include "version.h"

using namespace std;
using namespace hflt;

struct Options {
  string model_path;
  string dataset_path;
};
STRUCTOPT(Options, model_path, dataset_path);

int main(int argc, char *argv[]) {
  // get cli args and check if files exist
  Options options;
  try {
    options =
        structopt::app("hflt_bin", PROJECT_VER).parse<Options>(argc, argv);
  } catch (structopt::exception &e) {
    cout << e.what() << endl;
    cout << e.help() << endl;
    return -1;
  }
  // set some variables
  const long MAXIMUM_SEQUENCE_LENGTH = 128;
  const int BATCH_SIZE = 64;
  // create variables needed from cli args
  string pretrained_dir(options.model_path);
  string ds_file_path(options.dataset_path);

  // create dataset and dataloader
  // auto sampler = data::samplers::SequentialSampler;
  TransformerQADS<TokenizerAlbert> ds(pretrained_dir, MAXIMUM_SEQUENCE_LENGTH,
                                      readSquadExamples, ds_file_path);
  auto num_examples = ds.size().value();
  cout << "Found " << num_examples << " examples in " << ds_file_path << endl;

  // run eval loop
  for (size_t i = 0; i < ds.size().value(); ++i) {
    auto example = ds.get(i);
    // capture labels for metrics
    auto labels = example.label;
    // prepare input tensors for jit'ed model
    auto token_ids = example.input_ids;
    auto attention_masks = example.attention_mask;
    auto token_type_ids = example.token_type_ids;
    auto position_ids = example.position_ids;
  }
  return 1;
}
