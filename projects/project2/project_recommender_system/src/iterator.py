from run import run

def main():
    L_user = [0.001,0.01,0.1]
    L_item = [0.001, 0.01, 0.1]
    N_features = range(25,40)
    min_num_data = [10,50,100,250]
    error = {}
    minim = 9999
    for i in L_user:
        for j in L_item:
            for k in N_features:
                for p in min_num_data:
                    rmse = run(i,j,k,p,0.2)
                    if rmse < minim:
                        minim = rmse
                    error[rmse] = (i,j,k,p)
                    print("PARAMS: lambda_user{}, lambda_item{}, N_features{}, min_num_data{}".format(i,j,k,l))
                    print("ERROR TEST: {}".format(rmse))

    return error, minim
    #rmse = run(0.001, 0.001, 30, 0.2)
    #print(rmse)

if __name__ == '__main__':
    main()



