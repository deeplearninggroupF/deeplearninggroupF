import os
import pandas as pd
import datetime
import openpyxl

def processEachFile(filepath):
    pathDir = os.listdir(filepath)
    output = None
    for aDir in pathDir:
        thisFuckingFile = os.path.join(filepath, aDir)
        df = processFile(thisFuckingFile)
        if output is None:
            output = df
        else:
            output = output.append(df)
        # print(output.head())
    output.to_csv('processData.csv')
    output = processReturn(output)
    output.to_csv('processReturn.csv')
    return output

def processReturn(data):
    df = None
    processData = data.sort_values(['Date'])
    processData.index = range(len(processData))
    equityNames = data.Ticker.unique()
    print('There are', str(len(equityNames)), 'to process.')
    for name in equityNames:
        print('I am process', name)
        nameData = data[data.Ticker == name]
        processDate = nameData['Date']
        processDate = processDate.diff()
        processInd = list(processDate.index[processDate >= datetime.timedelta(days = 30)])
        if len(processInd) > 0:
            processStart = 0
            processInd.append(len(processInd) - 1)
            for i in range(len(processInd)):
                processSubNameData = nameData[processStart : processInd[i]]
                processPrices = processSubNameData['Price']
                processReturn = processPrices.pct_change()
                processSubNameData['Return'] = processReturn
                if df is None:
                    df = processSubNameData
                else:
                    df = df.append(processSubNameData)
                processStart = processInd[i]
        else:
            processPrices = nameData['Price']
            processReturn = processPrices.pct_change()
            nameData['Return'] = processReturn
            if df is None:
                df = nameData
            else:
                df = df.append(nameData)
    return df

def processFileName(fileName):
    subName = fileName.split('\\')
    nameList = subName[-1].split(' ')
    temp = nameList[-1].split('.')
    dateString = temp[0][:-1] + nameList[-3] + nameList[-2]
    return datetime.datetime.strptime(dateString, '%Y%b%d')

def processFile(fileName):
    processFile = openpyxl.load_workbook(fileName)
    processSheet = processFile['Worksheet']
    processTicker = []
    processPrice = []
    processDate = processFileName(fileName)
    processDates = []
    for (ind, processValue) in enumerate(processSheet.values):
        if ind == 0:
            continue
        else:
            try:
                aFuckingTicker = processValue[0]
                aFuckingPrice = float(processValue[-1])
                aFuckingDate = processDate
                processTicker.append(aFuckingTicker)
                processPrice.append(aFuckingPrice)
                processDates.append(aFuckingDate)
            except:
                continue
    df = pd.DataFrame({'Date': processDates, 'Ticker': processTicker, 'Price': processPrice})
    return df


# df = processEachFile('C:\\Users\\Dian\\OneDrive\\Studying\\Quantitative Finance\\Deep Learning\\Project\\final\\')
f = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('C:\\Users\\Dian\\OneDrive\\Studying\\Quantitative Finance\\Deep Learning\\Project\\processData.csv', converters={'Date':f})
output = processReturn(df)
output.to_csv('processReturn.csv')