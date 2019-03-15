f= open("data.txt","r")

lines =f.readlines()
# print(lines)
currStation = ''
currYear = -1
currCount = 0
currAmount = 0
for i in range(len(lines)):
    lines[i]=lines[i].split('\n')[0]
    lineArr = lines[i].split('\t')
    #print(lineArr)
    #if currCount > 1:
    #    print('count ' + str(currCount))
    #    print('amount ' + str(currAmount))
    #print(lineArr[0] + '  ' + currStation)
    #print(lineArr[2] + '  ' + str(currYear))
    if lineArr[0] == currStation and lineArr[2] == currYear:
        currCount += 1
        currAmount += int(lineArr[3])
    else:
        if i > 0:
           print (str(currYear) + '\t' + str(currAmount/currCount))
        currStation = lineArr[0]
        currYear = lineArr[2]
        currCount = 1
        currAmount = int(lineArr[3])


f.close() 