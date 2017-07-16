from utilities import data_pipeline
import pickle


def wrt_dataframes():
    des_df, df_web = data_pipeline()
    pickle.dump(des_df, open("../data/des_df.pkl","wb"), protocol=2)
    pickle.dump(df_web, open("../data/df_web.pkl","wb"), protocol=2)

if __name__=="__main__":
    wrt_dataframes()
