%matplotlib inline
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
bulldozer=pd.read_csv('..\\data\\TrainAndValid.csv',low_memory=False)
print(bulldozer.info())
print(bulldozer.describe())
sns.histplot(bulldozer.SalePrice)
plt.figure(figsize=(50,50))
sns.pairplot(bulldozer)
#expand data by add date column 
bulldozer.saledate=pd.to_datetime(bulldozer.saledate)
bulldozer.sort_values(by=['saledate'],inplace=True)
bulldozer['saleYear']=bulldozer.saledate.dt.year
bulldozer['saleMonth']=bulldozer.saledate.dt.month
bulldozer['saleDay']=bulldozer.saledate.dt.day
bulldozer['saleDayMont']=bulldozer.saledate.dt.days_in_month
print(bulldozer.SalePrice.corr(bulldozer.MachineHoursCurrentMeter))
#bulldozer.to_csv('..\\data\\bulldozer_traing_1.csv',index=False)
bulldozer=pd.read_csv("..\\data\\bulldozer_traing_1.csv",low_memory=False)
bulldozer.drop("saledate",axis=1,inplace=True)
#filling missing value
bulldozer.MachineHoursCurrentMeter.fillna(bulldozer.MachineHoursCurrentMeter.median(), inplace=True)
average_usage = bulldozer.groupby('fiBaseModel')['MachineHoursCurrentMeter'].median().reset_index()
print(average_usage)
bulldozer = bulldozer.merge(average_usage, on='fiBaseModel', how='left')
def fillUB(data:pd.DataFrame):
    if pd.notna(data.UsageBand):
        return data.UsageBand
    else:
        if data.MachineHoursCurrentMeter_y*35>=data.MachineHoursCurrentMeter_x:
            return "Low"
        elif data.MachineHoursCurrentMeter_y*70<=data.MachineHoursCurrentMeter_x:
            return "high"
        else:
            return "medium"
bulldozer["UsageBand"]= bulldozer.apply(fillUB, axis=1)
print(bulldozer.info())
bulldozer.UsageBand=bulldozer.UsageBand.str.lower()
bulldozer.UsageBand= pd.Categorical(bulldozer.UsageBand, categories=['low', 'medium', 'high'], ordered=True)
print(bulldozer.UsageBand.value_counts())
#bulldozer.to_csv("..\\data\\bulldozer_traing_2.csv",index=False)
bulldozer.auctioneerID=bulldozer.auctioneerID.fillna(bulldozer.median)
for i in bulldozer.columns:
    if not pd.api.types.is_numeric_dtype(bulldozer[i]):
        bulldozer[i]=pd.Categorical(bulldozer[i]).codes+1
print(bulldozer.info())
#bulldozer.to_csv("..\\data\\bulldozer_traing_3.1.csv",index=False)
#data split
bulldozer_traing=bulldozer[bulldozer.saleYear!=2012]
bulldozer_validate=bulldozer[(bulldozer.saleYear==2012) & (bulldozer.saleMonth<=4)]
bulldozer_test=bulldozer[(bulldozer.saleYear==2012) &  (bulldozer.saleMonth>4)]
bulldozer_traing.to_csv("..\\data\\bulldozer_traing_1,1.csv",index=False)
bulldozer_validate.to_csv("..\\data\\bulldozer_validate_1,1.csv",index=False)
