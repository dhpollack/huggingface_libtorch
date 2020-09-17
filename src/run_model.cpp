#include "run_model.h"
#include "hflt.h"
#include "version.h"

using namespace std;
using namespace torch;
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
  /*
  if (argc != 3) {
    string pn(argv[0]);
    string pn_nopath = pn.substr(pn.find_last_of("/\\") + 1);
    cout << "usage:  " << pn_nopath << " [model_path] [data_file_path]" << endl;
    cout << "   i.e. `" << pn
         << " ../models/sst2_trained "
            "../data/SST-2/dev.tsv`"
         << endl;
    return -1;
  }
  */
  // set some variables
  const long MAXIMUM_SEQUENCE_LENGTH = 128;
  const int BATCH_SIZE = 64;
  Device device(torch::cuda::is_available() ? "cuda" : "cpu");
  // create variables needed from cli args
  // string pretrained_dir(argv[1]);
  string pretrained_dir(options.model_path);
  string traced_model_path = pretrained_dir + "/traced_albert.pt";
  // string ds_file_path(argv[2]);
  string ds_file_path(options.dataset_path);

  // create dataset and dataloader
  // auto sampler = data::samplers::SequentialSampler;
  TransformerClassificationDS<TokenizerAlbert> ds(
      pretrained_dir, MAXIMUM_SEQUENCE_LENGTH, readSST2CsvFile, ds_file_path);
  auto ds_map = ds.map(data::transforms::Stack<TransformerFeatures<>>());
  auto dl = data::make_data_loader<data::samplers::SequentialSampler>(
      move(ds_map), BATCH_SIZE);
  auto num_examples = ds.size().value();
  size_t num_batches = ceil(num_examples / (float)BATCH_SIZE);
  cout << "Found " << num_examples << " examples in " << argv[2] << endl;
  cout << "Dataloader contains " << num_batches << " batches" << endl;

  // load albert model and put into eval mode
  jit::script::Module model;
  try {
    model = jit::load(traced_model_path, device);
  } catch (const c10::Error &e) {
    cerr << "error loading the model" << endl;
    return -1;
  }
  model.eval();

  // equivalent to with torch.no_grad() in python
  NoGradGuard no_grad_guard;
  // initialize vectors to capture results for metrics calculations
  vector<Tensor> preds_vec;
  vector<Tensor> labels_vec;
  preds_vec.reserve(num_batches);
  labels_vec.reserve(num_batches);
  // run eval loop
  for (auto &mb : *dl) {
    // cout << "Batch Size: " << mb.input_ids.sizes() << endl;
    // capture labels for metrics
    labels_vec.emplace_back(mb.label);
    // prepare input tensors for jit'ed model
    auto token_ids = mb.input_ids;
    auto attention_masks = mb.attention_mask;
    auto token_type_ids = mb.token_type_ids;
    auto position_ids = mb.position_ids;
    vector<jit::IValue> inputs;
    inputs.emplace_back(token_ids.to(device));
    inputs.emplace_back(attention_masks.to(device));
    inputs.emplace_back(token_type_ids.to(device));
    inputs.emplace_back(position_ids.to(device));
    // do inference and return results as a tuple
    auto out = model.forward(inputs).toTuple();
    // capture prediction log likelihoods for metrics
    preds_vec.emplace_back(out->elements()[0].toTensor().to(Device("cpu")));
  }
  // concatenate predictions and get most likely prediction
  Tensor preds = cat(preds_vec, 0).argmax(1);
  // concatenate labels
  Tensor labels = cat(labels_vec, 0).flatten();
  // calculate the number of correct predictions and total predictions
  auto correct = preds.eq(labels).sum().item<float>();
  float total = preds.size(0);
  // report accuracy
  cout << "Acc: " << correct / total << endl;
  return 1;
}
