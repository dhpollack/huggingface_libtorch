Simple sentencepiece tests

```sh
# use this to compile the sentence piece stuff
g++ -o tmp/sp/AlbertSentencepiece tmp/sentencepiece.cpp -L/home/david/.local/lib -lsentencepiece
./tmp/sp/AlbertSentencepiece
# use cmake to compile
mkdir build && cd build
CMAKE_PREFIX_PATH=/home/david/.local cmake ../tmp/sp
make
./AlbertSentencepiece
```

## Resources

[boost tokenizer docs](https://www.boost.org/doc/libs/1_71_0/libs/tokenizer/doc/index.html)
[csv processing](https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c)
[pytorch-cpp examples](https://github.com/prabhuomkar/pytorch-cpp.)
[unique_ptr for sentencepiece](https://stackoverflow.com/questions/42595473/correct-usage-of-unique-ptr-in-class-member)
[batch function](https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/datasets/base.h#L69)
