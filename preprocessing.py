# Reference:
# https://github.com/easezyc/WSDM2022-PTUPCDR

import pandas as pd
import gzip
import json
import tqdm
import random
import os


# class DataPreprocessingMid:
#     def __init__(self, root, dealing):
#         self.root = root
#         self.dealing = dealing

#     def main(self):
#         print("Parsing " + self.dealing + " Mid...")
#         re = []
#         with gzip.open(
#             self.root + "raw/reviews_" + self.dealing + "_5.json.gz", "rb"
#         ) as f:
#             for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
#                 line = json.loads(line)
#         #         re.append([line["reviewerID"], line["asin"], line["overall"]])
#         # re = pd.DataFrame(re, columns=["uid", "iid", "y"])
#                 re.append([line["reviewerID"],line["asin"],line["overall"],line["unixReviewTime"],])
#         re = pd.DataFrame(re, columns=["uid", "iid", "y", "t"])
#         print(self.dealing + " Mid Done.")
#         re.to_csv(self.root + "mid/" + self.dealing + ".csv", index=0)
#         return re

class DataPreprocessingMid:
    def __init__(self, root, dealing):
        self.root = root
        self.dealing = dealing

    def main(self):
        print("Parsing " + self.dealing + " Mid...")
        re = []
        with gzip.open(
            self.root + "mid/" + self.dealing + ".csv", "rb"
        ) as f:
            for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
                line = json.loads(line)
        #         re.append([line["reviewerID"], line["asin"], line["overall"]])
        # re = pd.DataFrame(re, columns=["uid", "iid", "y"])
                re.append([line["reviewerID"],line["asin"],line["overall"],line["unixReviewTime"],])
        re = pd.DataFrame(re, columns=["uid", "iid", "y", "t"])
        print(self.dealing + " Mid Done.")
        re.to_csv(self.root + "mid/" + self.dealing + ".csv", index=0)
        return re

class DataPreprocessingReady:
    def __init__(self, root, src_tgt_pairs, task, ratio):
        self.root = root
        self.src = src_tgt_pairs[task]["src"]
        self.tgt = src_tgt_pairs[task]["tgt"]
        self.ratio = ratio

    def read_mid(self, field):
        path = self.root + "mid/" + field + ".csv"
        re = pd.read_csv(path)
        return re

    def mapper(self, src, tgt):
        print(
            "Source inters: {}, uid: {}, iid: {}.".format(
                len(src), len(set(src.uid)), len(set(src.iid))
            )
        )
        print(
            "Target inters: {}, uid: {}, iid: {}.".format(
                len(tgt), len(set(tgt.uid)), len(set(tgt.iid))
            )
        )
        co_uid = set(src.uid) & set(tgt.uid)
        all_uid = set(src.uid) | set(tgt.uid)
        print("All uid: {}, Co uid: {}.".format(len(all_uid), len(co_uid)))
        uid_dict = dict(zip(all_uid, range(len(all_uid))))
        iid_dict_src = dict(zip(set(src.iid), range(len(set(src.iid)))))
        iid_dict_tgt = dict(
            zip(
                set(tgt.iid),
                range(len(set(src.iid)), len(set(src.iid)) + len(set(tgt.iid))),
            )
        )
        # uid, iid を 辞書を使って、新しい連続なIDに振り直している
        src.uid = src.uid.map(uid_dict)
        src.iid = src.iid.map(iid_dict_src)
        tgt.uid = tgt.uid.map(uid_dict)
        tgt.iid = tgt.iid.map(iid_dict_tgt)
        return src, tgt

    # def get_history(self, data, uid_set):
    #     pos_seq_dict = {}
    #     # data = ソースのデータフレーム全体、uid_set = 共通ユーザーのリスト
    #     for uid in tqdm.tqdm(uid_set):
    #         # uid_set、つまり、共通ユーザーのリストからユーザーIDを１つづつ取り出す
    #         # そのユーザーIDのyが3以上のアイテムIDのリスト化してシーケンス　pos を作成
    #         pos = data[(data.uid == uid) & (data.y > 3)].iid.values.tolist()
    #         # pos_seq_dictのuidの行にシーケンス pos を設定。uid と pos ポジティブシーケンスの辞書（対応表）を作成
    #         pos_seq_dict[uid] = pos
    #     return pos_seq_dict

    # def split(self, src, tgt):
    #     # src ソースドメインのデータフレーム
    #     # tgt ターゲットドメインのデータフレーム
    #     # いろいろな種類のデータフレームをsrc, tgtを元に分割などをして作成している
    #     print("All iid: {}.".format(len(set(src.iid) | set(tgt.iid))))

    #     # ソースユーザーは、ソースドメインのユニークなユーザーのリスト
    #     src_users = set(src.uid.unique())

    #     # ターゲットユーザーは、ターゲットドメインのユニークなユーザーのリスト
    #     tgt_users = set(tgt.uid.unique())

    #     # 共通ユーザーは、ソースドメインのユニークなユーザーとターゲットドメインのユニークなユーザーで
    #     # ANDをとったユーザーIDのリスト
    #     co_users = src_users & tgt_users

    #     # テストユーザーは、共通ユーザーの中から指定した比率だけランダムに抽出（20%なら、20%になるまでランダムに抽出）
    #     test_users = set(random.sample(co_users, round(self.ratio[1] * len(co_users))))

    #     # 学習用ソースデータフレームは、ソース全体（ユニークではない。）
    #     train_src = src

    #     # 学習用ターゲットデータフレームは、ターゲットユーザー（ユニーク）からテストユーザー（ユニーク）を引いて構成される
    #     train_tgt = tgt[tgt["uid"].isin(tgt_users - test_users)]
    #     # ターゲットデータフレームtgtの中から、ユーザーID列("uid")が特定のユーザーセット
    #     # (co_users - test_users)に含まれる行のみを選択しています。train_tgt　は、データフレーム

    #     # テスト用データフレームは、ターゲットの中でテストユーザーのみで構成される
    #     test = tgt[tgt["uid"].isin(test_users)]

    #     # ポジティブシーケンスの辞書は、ソースデータフレームと共通ユーザーのポジティブ評価の履歴から作成する。ソースから作成したポジティブシーケンス
    #     pos_seq_dict = self.get_history(src, co_users)

    #     # 学習用メタのデータフレームは、共通ユーザーからテストユーザーを引いたユーザーのターゲットのデータフレーム
    #     train_meta = tgt[tgt["uid"].isin(co_users - test_users)]

    #     # 学習用メタのデータフレーム（元はターゲットドメイン）のポジティブシーケンスのIDをpos_seq_dictを使って、IDのを振り直している
    #     # 学習用メタのデータフレーム（元はターゲットドメイン）にソースドメインのポジティブシーケンスをuidで結合している
    #     # これができるのは、uid がターゲットとソースで共通だから。
    #     train_meta["pos_seq"] = train_meta["uid"].map(pos_seq_dict)

    #     # テストのポジティブシーケンスのIDをpos_seq_dictを使って、IDのを振り直している
    #     # test のデータフレームも、uidでポジティブシーケンスを結合している
    #     # これができるのは、test の uid も、テストとソースで共通だから。
    #     test["pos_seq"] = test["uid"].map(pos_seq_dict)

    #     return train_src, train_tgt, train_meta, test

    def get_history(self, data, uid_set):
        # data = ソースのデータフレーム全体、uid_set = 共通ユーザーのリスト
        pos_seq_dict = {}
        for uid in tqdm.tqdm(uid_set):
            pos_y_dict = {}
            uid_data = data[data.uid == uid]
            # uid_data = ソースデータのuidと共通ユーザーのuidが同じになるデータフレーム
            
            # yの値のリストを取得
            y_values = uid_data.y.unique() # ソースデータの ユニークな y のリスト
            
            for y in y_values: # ソースデータの y　をとりだして、yが同じソースデータのiid のリストを取り出している
                pos = uid_data[uid_data.y == y].iid.values.tolist()
                pos_y_dict[(uid, y)] = pos # それをuid, y の辞書に登録
            
            # 全ての可能なyの値についてチェック
            for y in range(1, 6):
                if (uid, y) not in pos_y_dict or not pos_y_dict[(uid, y)]:
                    pos_y_dict[(uid, y)] = uid_data[uid_data.y > 3].iid.values.tolist()
            
            pos_seq_dict.update(pos_y_dict)

        return pos_seq_dict

    def split(self, src, tgt):   
        print('All iid: {}.'.format(len(set(src.iid) | set(tgt.iid))))
        src_users = set(src.uid.unique())
        tgt_users = set(tgt.uid.unique())
        co_users = src_users & tgt_users
        test_users = set(random.sample(co_users, round(self.ratio[1] * len(co_users))))
        train_src = src
        train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]
        test = tgt[tgt['uid'].isin(test_users)]

        # global pos_seq_dict1
        pos_seq_dict = self.get_history(src, co_users)
        train_meta = tgt[tgt['uid'].isin(co_users - test_users)]
        # train_meta['pos_seq'] = train_meta['uid'].map(pos_seq_dict)
        # test['pos_seq'] = test['uid'].map(pos_seq_dict)

        # pos_seq_dictをマッピングする関数
        def map_pos_seq(row, pos_seq_dict):
            uid = row['uid']
            y = row['y']
            return pos_seq_dict.get((uid, y), [])

        # train_metaに新しい列'pos_seq'を追加し、map_pos_seq関数を適用
        train_meta['pos_seq'] = train_meta.apply(lambda row: map_pos_seq(row, pos_seq_dict), axis=1)
        test['pos_seq'] = test.apply(lambda row: map_pos_seq(row, pos_seq_dict), axis=1)

        #行の表示を最大50行にする
        pd.set_option('display.max_rows', 30)
        
        # #列の表示を最大100列二設定する
        # pd.set_option('display.max_columns', 100)
    
        #  #行の表示設定を戻す
        # pd.reset_option('display.max_rows')
        
        # #列の表示設定を戻す
        # pd.reset_option('display.max_columns')
    
    
        print("***train_src***")
        print(train_src)
        
        
        print("***train_tgt***")
        print(train_tgt)
    
    
        print("***train_meta***")
        print(train_meta)
    
        print("***test***")
        print(test)

        return train_src, train_tgt, train_meta, test

    def save(self, train_src, train_tgt, train_meta, test):
        output_root = (
            self.root
            + "ready/_"
            + str(int(self.ratio[0] * 10))
            + "_"
            + str(int(self.ratio[1] * 10))
            + "/tgt_"
            + self.tgt
            + "_src_"
            + self.src
        )
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        print(output_root)
        train_src.to_csv(
            output_root + "/train_src.csv", sep=",", header=None, index=False
        )
        train_tgt.to_csv(
            output_root + "/train_tgt.csv", sep=",", header=None, index=False
        )
        train_meta.to_csv(
            output_root + "/train_meta.csv", sep=",", header=None, index=False
        )
        test.to_csv(output_root + "/test.csv", sep=",", header=None, index=False)

    def main(self):
        # CSVファイルを読み込んでソースとターゲットのデータフレームを作成
        src = self.read_mid(self.src)
        tgt = self.read_mid(self.tgt)

        # ソースとターゲットのデータフレームのユーザーとアイテムのIDを連続なIDで振り直している
        src, tgt = self.mapper(src, tgt)

        # ソースとターゲットのデータフレームから、学習用ソースと、学習用ターゲットと、学習用メタと、テストのデータフレームを作成
        train_src, train_tgt, train_meta, test = self.split(src, tgt)

        # ready フォルダに、学習用ソースと、学習用ターゲットと、学習用メタと、テストのデータフレームを書き込んでいる
        self.save(train_src, train_tgt, train_meta, test)
