#include "src/squad_utils.h"
#include "gtest/gtest.h"

using namespace std;

bool file_exists(const char *fp) {
  ifstream fd(fp);
  return fd.good();
}

size_t file_size(const char *fp) {
  struct stat st;
  if (stat(fp, &st) != 0) {
    return 0;
  }
  return st.st_size;
}

void file_test(const char *fp, const bool remove_file = false) {
  ASSERT_EQ(file_exists(fp), true);
  ASSERT_GT(file_size(fp), 0);
  if (remove_file)
    remove(fp);
}

const char *squad_path = "data/SQuAD/dev-v2.0.json";

TEST(squadutilsTest, read_squad_json) {
  file_test(squad_path);
  ifstream squad_file(squad_path);
  vector<SquadExample> examples = read_squad_examples(squad_file, true);
  for (auto &example : examples)
    EXPECT_GT(example.paragraph_text.size(), 0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
