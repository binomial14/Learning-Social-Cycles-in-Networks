import os
path='./learning-social-circles/Training'
def make_ground_truth(ego):
    truth = open(os.path.join(path,str(ego)+'.circles'),'r')
    result = open(os.path.join('./result',str(ego)+'.truth'),'w')
    result.write("UserId,Predicted\n")
    result.write(str(ego)+",")
    tLines = truth.readlines()
    truth.close()
    for circle in tLines:
        circle_name, idxs = circle.split(': ')
        # print(idxs.rstrip())
        result.write(idxs.rstrip())
        if circle!=tLines[-1]:
            result.write(';')
    result.write('\n')
    result.close()


if __name__ == "__main__":
    make_ground_truth(239)