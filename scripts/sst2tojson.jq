# jq -R -f scripts/sst2tojson.jq [input_tsv_file] > [output_json_file]
[inputs] |
{data: 
  [
    to_entries | 
    .[] | 
    (.key | tostring) + "\t" + .value | 
    split("\t") | 
    {guid: .[0], text_a: .[1], text_b: "", label: .[2]}
  ]
}
