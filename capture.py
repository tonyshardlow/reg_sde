#
def get_supp_bvp():
    output = get_ipython().getoutput(u'python run_supp_bvp.py')
    filename='run_supp_bvp2.txt'
    write_to_file(filename,output)
#
def get_supp_lin():
    output = get_ipython().getoutput(u'python run_supp_lin.py')
    filename='run_supp_lin.txt'
    write_to_file(filename,output)
#
def get_supp_split1():
    output = get_ipython().getoutput(u'python run_supp_split1.py')
    filename='run_supp_split1.txt'
    write_to_file(filename,output)
#
def get_supp_split2():
    output = get_ipython().getoutput(u'python run_supp_split2.py')
    filename='run_supp_split2.txt'
    write_to_file(filename,output)
#
def write_to_file(filename,output):
    f = open(filename,'a')
    for x in output:
        f.write(x.decode("utf-8"))
        f.write(u"\n")
    f.close()
#
if __name__ == "__main__":
    get_supp_bvp()
    #get_supp_lin()
    #get_supp_split1()
    #get_supp_split2()
#
