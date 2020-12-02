

def write_to_output_file(filename, result_string):
    out_dir = "../output/"
    f = open(out_dir + filename, 'w')
    f.write(result_string)
    f.close()
