#include "run_model.h"
#include "data_utils.h"

using namespace std;

int main(int argc, char *argv[]) {
  // get cli args and check if files exist
  if (argc != 3) {
    cout << "usage:  huggingface-albert [model_path] [data_file_path]" << endl;
    cout << "   i.e. `./huggingface-albert ../models/sst2_trained "
            "../data/SST-2/dev.tsv`"
         << endl;
    return -1;
  }
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
  // read examples
  string fp(argv[2]);
  auto examples = readCsvFile(fp);
  if (examples.size() == 1) {
    std::cerr << "found 0 examples in the file: " << argv[2] << endl;
    return -1;
  }

  // create dataset and dataloader
  auto ds = SST2(fp, sentencepiece_model_path, MAXIMUM_SEQUENCE_LENGTH)
                .map(torch::data::transforms::Stack<>());
  ;
  auto dl =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          move(ds), BATCH_SIZE);

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
    cout << "Batch Size: " << mb.data.sizes() << endl;
    labels_vec.push_back(mb.target);
    auto token_ids = torch::select(mb.data, 1, 0);
    auto attention_masks = torch::select(mb.data, 1, 1);
    auto token_type_ids = torch::select(mb.data, 1, 2);
    auto position_ids = torch::select(mb.data, 1, 3);
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
