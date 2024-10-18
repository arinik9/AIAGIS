import os

out_result_filepath = os.path.join("/Users/narinik/Mirror/workspace/GenomeDataPaper/out/genome_preprocessing-results/bvbrc/analysis", "genome_sim_analysis.csv")


serovar_list = ['h10n7', 'h10n8', 'h2n2', 'h3n1', 'h5n1', 'h5n2', 'h5n3', 'h5n4', 'h5n5', 'h5n6', 'h5n8', 'h5n9', 'h6n2', 'h7n1', 'h7n2', 'h7n3', 'h7n4', 'h7n6', 'h7n7', 'h7n8', 'h7n9', 'h9n2', 'h2', 'h3', 'h5', 'h6', 'h7', 'h9', 'h10']
tot = 0
count = 0
for i, s1 in enumerate(serovar_list):
    for j, s2 in enumerate(serovar_list):
        if i <= j:
            tot += 1
            curr_out_result_filepath = out_result_filepath.replace(".csv", "_"+s1+"_"+s2+".csv")
            if not os.path.exists(curr_out_result_filepath):
            	print(curr_out_result_filepath)
            	count += 1
print(count, "/", tot)