from utilities import data_pipeline
import pickle


def wrt_dataframes():
    des_df, df_web = data_pipeline(1000)
    with open("../data/descriptions_data.txt","w") as file_:
        for txt, class_num in zip(des_df["txt"], des_df["class_num"]):
            file_.write("{0}\t{1}\n".format(class_num, txt))
    with open("../data/web_site_data.txt", "w") as file_:
        for txt, class_num, id_ in zip(df_web["class_txt"], df_web["class_num"],df_web["company_id"]):
            if txt is not "":
                for i in txt.split():
                    print txt.split()
                    print(i)
                    fknvfknv
                file_.write("{0}\t{1}\t{2}\n".format(id_, class_num, txt))

    # pickle.dump(des_df, open("../data/des_df.pkl","wb"), protocol=2)
    # pickle.dump(df_web, open("../data/df_web.pkl","wb"), protocol=2)

if __name__=="__main__":
    wrt_dataframes()
