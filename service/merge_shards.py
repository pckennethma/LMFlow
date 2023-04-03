import argparse
import jsonlines

def merge_shards(output_file, num_workers):
    input_files = [f"{output_file}_shard_{i}.jsonl" for i in range(num_workers)]
    with jsonlines.open(output_file, mode='w') as writer:
        for input_file in input_files:
            with jsonlines.open(input_file, mode='r') as reader:
                for obj in reader:
                    writer.write(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge output JSONL files from all shards.")
    parser.add_argument("output_file", type=str, help="Path to the merged output JSONL file")
    parser.add_argument("num_workers", type=int, help="Total number of workers")

    args = parser.parse_args()

    merge_shards(args.output_file, args.num_workers)