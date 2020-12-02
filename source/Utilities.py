
def write_results(results, filepath):
    # writes the output of test results to a file as a CSV
    # ie results = [1, 25, 23]
    # file will now read:
    #    "0,1\n1,25\n2,23\n"
    f = open(filepath, 'w')
    for i in range(results.size):
        f.write(str(i) + "," + str(results[i]) + "\n")
    f.close()
