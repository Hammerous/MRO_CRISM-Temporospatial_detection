minimun_views = 2

def process_files(input_files, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file in input_files:
            with open(file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    parts = line.strip().split()
                    if parts:
                        last_element_split = parts[-1].split(',')
                        if len(last_element_split) > minimun_views:
                            out_f.write(line)

# Example usage:
input_files = ['IMG2SHP_results.txt']  # Replace with actual filenames
output_file = 'merged.txt'
process_files(input_files, output_file)
