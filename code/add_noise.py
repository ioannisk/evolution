# 1. add noise to web 20 %
# 1. add noise to web 50 %
# 2. remove all keywords of descriptions from websites
# 3. add noise to both 20 %
# 4. add noise to both 50 %

# def web_noise():





if __name__=="__main__":
    data_path = "/home/ioannis/data/recovery_test/"
    files =[data_path +"fold{}/".format(i)+"ranking_validation.json" for i in range(0,3)]
    print(files)

