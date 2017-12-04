from run import run

def main():
    # L_user = [0, 0.001,0.01,0.1,0.2,0.5,1]
    # L_item = [0, 0.001, 0.01, 0.1, 0.2, 0.5, 1]
    # N_features = [10,20,30,35,40,45]
    # p_test = [0.1, 0.2, 0.3, 0.4, 0.5]
    # error = {}
    # minim = 9999
    # for i in L_user:
    #     for j in L_item:
    #         for k in N_features:
    #             for p in p_test:
    #                 rmse = run(i,j,k,p)
    #                 if rmse < minim:
    #                     minim = rmse
    #                 error[rmse] = (i,j,k,p)
    # return error, minim
    rmse = run(0.001, 0.001, 30, 0.2)
    print(rmse)

if __name__ == '__main__':
    main()



