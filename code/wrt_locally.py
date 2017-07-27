from utilities import data_pipeline
import pickle

def remove_words_less_than(txt, more_than):
    buffer_txt = ""
    for i in txt.split():
        # remove characters
        if len(i)>more_than:
            buffer_txt += i +" "
    return buffer_txt

def wrt_dataframes():
    des_df, df_web = data_pipeline()
    with open("../data/descriptions_data_1.txt","w") as file_:
        for txt, class_num in zip(des_df["txt"], des_df["class_num"]):
            file_.write("{0}\t{1}\n".format(class_num, txt))
    # stop
    with open("../data/web_site_data_1.txt", "w") as file_:
        for txt, class_num, id_ in zip(df_web["class_txt"], df_web["class_num"],df_web["company_id"]):
            if txt is "":
                continue
            buffer_txt = remove_words_less_than(txt, 1)
            file_.write("{0}\t{1}\t{2}\n".format(id_, class_num, buffer_txt))
    with open("../data/web_site_meta_1.txt", "w") as file_:
        for des, tit, class_num, id_ in zip(df_web["descriptions"], df_web["titles"], df_web["class_num"],df_web["company_id"]):
            if des=="" or tit =="":
                continue
            des = remove_words_less_than(des,1)
            tit = remove_words_less_than(tit,1)
            buffer_txt = des + tit
            if buffer_txt=="":
                continue
            file_.write("{0}\t{1}\t{2}\n".format(id_, class_num, buffer_txt))


if __name__=="__main__":
    wrt_dataframes()
