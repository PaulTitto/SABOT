from helper.chunk_graph_stats import count_graph_stats


def main():
    folder_path = "exp_batch_storage"
    reulsts = count_graph_stats(folder_path)
    print(reulsts['entities'])
    print(reulsts['relations'])

if __name__ == "__main__":
    main()