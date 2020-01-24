#include "run_model.h"
#include "data_utils.h"
#include "transformer_stack.h"

using namespace std;

int main(int argc, char *argv[]) {
  // get cli args and check if files exist
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
  string pretrained_dir(argv[1]);
  string sentencepiece_model_path(argv[1]);
  sentencepiece_model_path.append("/spiece.model");
  string traced_model_path(argv[1]);
  traced_model_path.append("/traced_albert.pt");
  if (not ifstream(sentencepiece_model_path)) {
    cerr << sentencepiece_model_path << " does not exist" << endl;
    return -1;
  }
  if (not ifstream(traced_model_path)) {
    cerr << traced_model_path << " does not exist" << endl;
    return -1;
  }

  // set some variables
  int MAXIMUM_SEQUENCE_LENGTH = 128;
  int BATCH_SIZE = 64;
  torch::Device device(torch::cuda::is_available() ? "cuda" : "cpu");

  // create dataset and dataloader
  string fp(argv[2]);
  SST2<TokenizerAlbert> ds(fp, pretrained_dir, MAXIMUM_SEQUENCE_LENGTH);
  cout << "Found " << ds.size() << " examples in " << argv[2] << endl;
  auto ds_map = ds.map(torch::data::transforms::Stack<TransformerFeatures<>>());
  auto dl =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          move(ds_map), BATCH_SIZE);

  // load albert model and put into eval mode
  torch::jit::script::Module model;
  try {
    model = torch::jit::load(traced_model_path, device);
  } catch (const c10::Error &e) {
    cerr << "error loading the model" << endl;
    return -1;
  }
  model.eval();

  // equivalent to with torch.no_grad() in python
  torch::NoGradGuard no_grad_guard;
  // run eval loop
  vector<torch::Tensor> preds_vec;
  vector<torch::Tensor> labels_vec;
  for (auto &mb : *dl) {
    cout << "Batch Size: " << mb.input_ids.sizes() << endl;
    labels_vec.push_back(mb.label);
    auto token_ids = mb.input_ids;
    auto attention_masks = mb.attention_mask;
    auto token_type_ids = mb.token_type_ids;
    auto position_ids = mb.position_ids;
    vector<torch::jit::IValue> inputs;
    inputs.push_back(token_ids.to(device));
    inputs.push_back(attention_masks.to(device));
    inputs.push_back(token_type_ids.to(device));
    inputs.push_back(position_ids.to(device));
    auto out = model.forward(inputs).toTuple();
    preds_vec.push_back(out->elements()[0].toTensor().to(torch::Device("cpu")));
    // cout << "Output: " << out->elements()[0].toTensor() << endl;
  }
  torch::Tensor preds = torch::cat(preds_vec, 0).argmax(1);
  torch::Tensor labels = torch::cat(labels_vec, 0).flatten();
  float correct = preds.eq(labels).sum().item<float>();
  float total = preds.size(0);
  cout << "Acc: " << correct / total << endl;
  return 1;
}
