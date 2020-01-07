# Notes to Self

## Resources

[boost tokenizer docs](https://www.boost.org/doc/libs/1_71_0/libs/tokenizer/doc/index.html)  
[csv processing](https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c)  
[pytorch-cpp examples](https://github.com/prabhuomkar/pytorch-cpp.)  
[unique_ptr for sentencepiece](https://stackoverflow.com/questions/42595473/correct-usage-of-unique-ptr-in-class-member)  
[batch function](https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/datasets/base.h#L69)  
[pkg-config and cmake](https://stackoverflow.com/questions/44487053/set-pkg-config-path-in-cmake) - sentencepiece doesn't have a proper cmake file
[libtorch no_grad()](https://discuss.pytorch.org/t/memory-leak-in-libtorch-extremely-simple-code/38149/5)

## C++ General Notes

how to compile sentencepiece manually
```sh
g++ -o out in.cpp -L<path to /lib with sentencepiece.so> -lsentencepiece
```
