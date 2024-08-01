# Reference:
# https://github.com/easezyc/WSDM2022-PTUPCDR

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
import random
from tensorflow import keras
from models import Model


class Run:
    def __init__(self, config):
        self.use_cuda = config["use_cuda"]
        self.root = config["root"]
        self.ratio = config["ratio"]
        self.task = config["task"]
        self.src = config["src_tgt_pairs"][self.task]["src"]
        self.tgt = config["src_tgt_pairs"][self.task]["tgt"]
        self.uid_all = config["src_tgt_pairs"][self.task]["uid"]
        self.iid_all = config["src_tgt_pairs"][self.task]["iid"]
        self.batchsize_src = config["src_tgt_pairs"][self.task]["batchsize_src"]
        self.batchsize_tgt = config["src_tgt_pairs"][self.task]["batchsize_tgt"]
        self.batchsize_meta = config["src_tgt_pairs"][self.task]["batchsize_meta"]
        self.batchsize_map = config["src_tgt_pairs"][self.task]["batchsize_map"]
        self.batchsize_test = config["src_tgt_pairs"][self.task]["batchsize_test"]
        self.batchsize_aug = self.batchsize_src
        self.epoch = config["epoch"]
        self.emb_dim = config["emb_dim"]
        self.meta_dim = config["meta_dim"]
        self.num_fields = config["num_fields"]
        self.lr = config["lr"]
        self.wd = config["wd"]
        self.input_root = (
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
        self.src_path = self.input_root + "/train_src.csv"
        self.tgt_path = self.input_root + "/train_tgt.csv"
        self.meta_path = self.input_root + "/train_meta.csv"
        self.test_path = self.input_root + "/test.csv"
        self.results = {
            "tgt_mae": 10,
            "tgt_rmse": 10,
            "aug_mae": 10,
            "aug_rmse": 10,
            "emcdr_mae": 10,
            "emcdr_rmse": 10,
            "dptupcdr_mae": 10,
            "dptupcdr_rmse": 10,
        }

    def seq_extractor(self, x):
        x = x.rstrip("]").lstrip("[").split(", ")
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)

    def read_log_data(self, path, batchsize, history=False):
        if not history:
            cols = ["uid", "iid", "y"]
            x_col = ["uid", "iid"]
            y_col = ["y"]
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ["uid", "iid", "y", "pos_seq"]
            x_col = ["uid", "iid"]
            y_col = ["y"]
            data.columns = cols

            print(path)
            print("*********1********* ")
            # データフレーム
            print(data)

            # ポジティブシーケンスを表示
            print("data.pos_seq = ")
            print(data.pos_seq)

            # data から 20個から成るポジティブシーケンスを取り出している
            pos_seq = keras.preprocessing.sequence.pad_sequences(
                data.pos_seq.map(self.seq_extractor), maxlen=20, padding="post"
            )

            # 履歴データの処理 (history=Trueの場合):

            # 履歴データが含まれる場合は、pos_seqカラムに含まれるシーケンスデータを処理します。
            # keras.preprocessing.sequence.pad_sequencesを使用して、
            # シーケンスデータをパディングし、一定の長さに揃えます。
            # シーケンスデータを含む新たなテンソルpos_seqを作成し、これを元の特徴量テンソルXと結合します。

            # data.pos_seq.map(self.seq_extractor)は、
            # Pandas のデータフレームのpos_seqカラムに含まれる各要素に対して、
            # self.seq_extractorメソッドを適用しています。
            # この操作は、データフレームのpos_seqカラムの各要素
            # （通常、シーケンスデータを表す文字列）を必要な形式
            # （この場合は数値のリスト）に変換するために用いられます。

            # seq_extractorメソッドの機能
            # seq_extractorメソッドは、文字列形式のシーケンスデータを受け取り、
            # 以下の処理を行います：

            # 文字列のクリーニング: 
            # 入力された文字列から余分な括弧[]を取り除きます。
            # これにより、例えば"[1, 2, 3]"という文字列が"1, 2, 3"に変換されます。

            # 分割: 
            # クリーニングされた文字列をカンマ,で分割して、
            # 個々の数値（まだ文字列形式）に分割します。

            # 数値変換: 
            # 分割された文字列の各要素を整数に変換します。
            # このとき、変換できない要素（例えば空文字や非数値の文字列）がある場合は、
            # 特定のデフォルト値（例えばself.iid_all）に置き換えられます。

            # NumPy配列の形成: 
            # 変換された数値のリストをNumPy配列に変換して返します。


            # このコード行は、pos_seqというカラムに含まれるシーケンスデータを整形し、
            # 機械学習モデルで扱いやすい形に変換するための処理を行っています。
            # この処理は、keras.preprocessing.sequence.pad_sequences関数を使用して実行されます。
            # 具体的には、以下のステップでデータを変換しています：

            # 処理の概要
            # シーケンスデータの抽出と変換:

            # data.pos_seq.map(self.seq_extractor)は、
            # データフレームのpos_seqカラムに含まれる各シーケンスデータ
            # （通常は文字列形式の数値リスト）をseq_extractor関数を用いて数値リストに変換します。
            # この関数は各シーケンスの文字列を数値のリストに変換し、
            # 不正な値があれば特定のデフォルト値（例えばself.iid_all）で置き換えます。
            # パディングの適用:

            # keras.preprocessing.sequence.pad_sequences関数を使用して、
            # 変換したシーケンスの長さを統一します。この関数は以下のパラメータを取ります：
            # maxlen: 各シーケンスの最大長さを指定します。
            # この例では20と設定されており、各シーケンスは最大で20の長さになります。
            # padding: パディングの適用方向を指定します。
            # "post"はシーケンスの後ろ（右側）にパディングを追加することを意味します。
            # これにより、シーケンスが20未満の場合、足りない部分に0が追加されて長さが20になります。

            # この処理は特に、シーケンスデータをモデルが処理可能な数値の形式に変換するために使用されます。
            # keras.preprocessing.sequence.pad_sequences関数によるパディングの前処理ステップの一部として機能し、
            # シーケンスデータが機械学習モデル、特にリカレントニューラルネットワークで扱うのに適した形式に整えられます。

            # このように、data.pos_seq.map(self.seq_extractor)はデータの前処理パイプラインにおいて重要な役割を果たし、
            # データを後続の分析や予測モデルに適した形に変換するための基礎的なステップです。

            # keras.preprocessing.sequence.pad_sequences関数は、与えられたシーケンスデータに対してパディング
            # （充填）や切り捨てを行いますが、元のシーケンス内の要素の順番を入れ替えたり、
            # 要素の値自体を変更したりすることはありません。この関数の主な目的は、
            # バッチ処理やリカレントニューラルネットワークでの使用に適した形式にシーケンスの長さを統一することです。

            # 動作の詳細
            # パディングの位置:

            # paddingパラメータにより、パディングがシーケンスの前（pre）または後（post）に追加されます。
            # "pre"を指定するとシーケンスの先頭にパディングが追加され、"post"を指定すると末尾に追加されます。
            # どちらの場合も、元のシーケンスデータの順序は保持されます。
            # パディングの値:

            # valueパラメータにより、パディングに使用する値を指定できます（デフォルトは0）。
            # この値で指定された数字がパディングとして追加されますが、
            # 既存のシーケンスデータの数値が変更されることはありません。
            # シーケンスの切り捨て:

            # シーケンスがmaxlenパラメータで指定された最大長よりも長い場合、
            # paddingパラメータに依存して先頭または末尾から要素が切り捨てられます。
            # ここでも、切り捨てられなかった部分のデータはその順序や値を保持します。
            # 例
            # たとえば、シーケンス [1, 2, 3] に対して maxlen=5、padding='post' 
            # の設定で pad_sequences を実行すると、結果は [1, 2, 3, 0, 0] となります。
            # ここで、元のシーケンス [1, 2, 3] の順序は変更されず、
            # パディングされた 0 が末尾に追加されるだけです。

            # data.pos_seq.map(self.seq_extractor)を実行するとき、
            # seq_extractor関数は入力されたシーケンスデータ（通常は文字列形式）を処理しますが、
            # この処理において元のシーケンスの各要素の順番を入れ替えることはありません。
            # また、元の数値を別の数値に変更することも原則としては行われませんが、
            # 例外的な処理については以下の通りです。

            # seq_extractor関数の動作の詳細
            # 文字列の整形と分割:

            # 入力された文字列から不要な括弧[]を削除し、カンマ,で分割してリストにします。
            # このプロセスでは元のデータの順序は保持されます。
            # 数値変換:

            # 分割された各要素を整数に変換します。
            # この際、正常に整数に変換できる場合はその値がそのままリストに追加されます。
            # 変換が不可能な要素（例えば、空の文字列や非数値の文字列が存在する場合）は
            # 特定のデフォルト値（例えばself.iid_all、通常はアイテムIDの最大値などを指定）に置き換えられます。
            # この点においてのみ、元の数値が異なる数値に変わる可能性があります。

            print("*********2********* ")
            print(data)

            print("data.pos_seq.map =")
            print(data.pos_seq)

            print("*********3********* ")
            print("pos_seq = ")
            print(pos_seq)

            # ポジティブシーケンスをtensorデータセット形式に変換
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)
            print("*********4********* ")
            print("pos_seq tensor = ")
            print(pos_seq)

            # data の x_col すなわち、uid, iid を id_fea として取り出している
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
            print("*********5********* ")
            print("id_fea tensor = ")
            print(id_fea)

            # uid, iid の id_fea と pos_seq を concat して、X を作成している！
            X = torch.cat([id_fea, pos_seq], dim=1)
            print("*********6********* ")
            print("X = ", X)

            # data の y_col から y を作成している！           
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            print("*********7********* ")
            print("y = ", y)

            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()

            # X, y からデータセット dataset を作っている
            dataset = TensorDataset(X, y)
            print("*********8********* ")
            print("dataset = ", dataset)

            # データローダーの生成:
            # TensorDataset(X, y)を使用して、特徴量と目的変数を組み合わせたデータセットを作成します。
            # DataLoader(dataset, batchsize, shuffle=True)を用いて、
            # データセットからデータをバッチサイズごとに取り出すデータローダーを生成します。
            # データのシャッフルもここで行われます。
            # dataset から DataLoader を使って、 data_iter を生成している

            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            print("*********9******** ")
            print("data_iter = ", data_iter)
            return data_iter

    def read_map_data(self):
        cols = ["uid", "iid", "y", "pos_seq"]
        # メタのCSVファイルを読み込んでメタネット用データフレームをつくる
        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols

        # メタネットの中で、ユニークなユーザーのtensor X を作る
        X = torch.tensor(data["uid"].unique(), dtype=torch.long)

        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
        # Xはユニークなユーザー。ユニークなユーザーIDの数だけからなる tensor y を作る
        #
        # range(X.shape[0])
        # Xテンソルの最初の次元の大きさ（shape[0]）を取得します。
        # これはXに含まれる要素の数、すなわちユニークなユーザーIDの数に相当します。
        #
        # range関数を用いて 0 から X.shape[0] - 1 までの整数のリスト（正確にはrangeオブジェクト）を生成します。
        # これにより、各ユーザーIDに一意のインデックスが割り当てられます。
        #
        # np.array(range(X.shape[0]))
        # 生成されたrangeオブジェクトをNumPyの配列に変換します。
        # この操作により、計算や変換が容易になる配列データ構造が得られます。
        #
        # torch.tensor(..., dtype=torch.long)
        # NumPy配列を引数として、dtype=torch.long（長整数型）のPyTorchテンソルに変換します。
        # このテンソルは後続の処理で目的変数やラベルとして使用されることが想定されます。

        # はい、その通りです。read_map_data関数において、Xはdata['uid'].unique()を用いて
        # データセットからユニークなユーザーIDを抽出して作成されますが、
        # yはそのデータセットから直接作成されているわけではありません。

        # yはnp.array(range(X.shape[0]))として生成されており、
        # これはXの要素数に基づいて0から始まる連番の整数配列を作成し、
        # それをテンソルに変換しています。つまり、yはXの各要素（ユニークなユーザーID）に
        # 対応する一意のインデックスを表しており、
        # モデルがIDを新しい表現にマッピングする際の"ターゲット"や"ラベル"として機能します。

        # このように、Xはデータセットから抽出されたユニークなIDを元に構築されているのに対して、
        # y は X の長さに基づく単なるインデックスの列であり、
        # データセットの具体的な内容から直接派生したものではありません。
        # これにより、各ユーザーIDに一意の整数ラベルを割り当てることで、
        # 教師あり学習での使用が可能になります。

        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter

    def read_aug_data(self):
        cols_train = ["uid", "iid", "y"]
        x_col = ["uid", "iid"]
        y_col = ["y"]

        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train

        tgt = pd.read_csv(self.tgt_path, header=None)
        tgt.columns = cols_train

        # ソースとターゲットからx_col = ["uid", "iid"] なので、
        # uidとiidの両方を value関数でNumpy形式で取り出してPytorchのテンソルへ変換
        X_src = torch.tensor(src[x_col].values, dtype=torch.long)
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)

        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)

        # ソースとターゲットの X を結合
        # ソースとターゲットの y を結合
        X = torch.cat([X_src, X_tgt])
        y = torch.cat([y_src, y_tgt])

        # X = torch.cat([X_src, X_tgt]) は、PyTorch の torch.cat() 関数を使用して、
        # 二つのテンソル X_src と X_tgt を連結する操作です。
        # ここでの目的は、ソースデータセット (src) からの特徴量テンソル X_src と、
        # ターゲットデータセット (tgt) からの特徴量テンソル X_tgt を一つの大きなテンソル X にまとめることです。

        # 処理の詳細と用途

        # データの結合:
        # torch.cat() 関数は、リスト内の複数のテンソルを連結するために使用されます。
        # デフォルトでは、dim=0（第一次元、つまり行方向）に沿ってテンソルを結合します。
        # これにより、X_src と X_tgt の全ての行が縦に積み重なり、
        # 結果としてより多くのデータサンプルを含む新しいテンソル X が形成されます。

        # 特徴量の拡張:
        # この操作により、異なるデータソース（ソースとターゲット）からの情報が統合され、
        # データの多様性が向上します。モデルがより広範なデータに基づいて訓練されるため、
        # 一般化能力の向上が期待されます。

        # データ拡張の目的:
        # 特にデータセットが小さい場合や多様な状況に対応する必要がある場合に、
        # 異なるデータソースを組み合わせることは有効な戦略です。これにより、
        # 過学習を防ぎ、未見のデータに対するモデルのパフォーマンスを向上させることができます。

        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)
        return data_iter

    def get_data(self):
        # CSVを読み込んでtensorデータセットを作成する
        print("========Reading data========")
        # ソースドメインのCSVをデータフレームとして読み込み、tensorデータセット形式に変換する
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print("src {} iter / batchsize = {} ".format(len(data_src), self.batchsize_src))

        # ターゲットドメインのCSVをデータフレームとして読み込み、tensorデータセット形式に変換する
        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print("tgt {} iter / batchsize = {} ".format(len(data_tgt), self.batchsize_tgt))

        # 共通ユーザー（ただしテストユーザーを除く）で、
        # 評価４以上のアイテムのリスト含む、メタネット用CSVをデータフレームとして読み込み、tensorデータセット形式に変換する
        data_meta = self.read_log_data(
            self.meta_path, self.batchsize_meta, history=True
        )
        print(
            "meta {} iter / batchsize = {} ".format(len(data_meta), self.batchsize_meta)
        )

        # メタデータからユニークなユーザーIDとアイテムIDのデータセットと
        # yが０からはじまる連番のtensorデータセット形式を作成している
        data_map = self.read_map_data()
        print("map {} iter / batchsize = {} ".format(len(data_map), self.batchsize_map))

        # ソースとターゲットを行方向（ユーザーの行が増える方向に積み重ねる）に
        # 結合したtensorデータセット形式を作成している
        data_aug = self.read_aug_data()
        print("aug {} iter / batchsize = {} ".format(len(data_aug), self.batchsize_aug))

        # テストユーザーで、評価４以上のアイテムのリスト含む、
        # テストユーザーをtensorデータセット形式を作成している
        data_test = self.read_log_data(
            self.test_path, self.batchsize_test, history=True
        )
        print(
            "test {} iter / batchsize = {} ".format(len(data_test), self.batchsize_test)
        )
        return data_src, data_tgt, data_meta, data_map, data_aug, data_test

    def get_model(self):
        # Modelを読み込む
        model = Model(
            self.uid_all,
            self.iid_all,
            self.num_fields,
            self.emb_dim,
            self.meta_dim
        )
        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model):
        
        # PTUPCDR は、Adam　を使っている。
        optimizer_src = torch.optim.Adam(
            params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        optimizer_tgt = torch.optim.Adam(
            params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        # DPTUPCDR は、SGD　を使っている。
        # GD by GD learning to_learn by gradient descent by gradient descent　を入れる場合は、ここを改造する
        optimizer_meta = torch.optim.SGD(  
            params=model.deep_meta_net.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
            momentum=0.9,
        )
        optimizer_aug = torch.optim.Adam(
            params=model.aug_model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        optimizer_map = torch.optim.Adam(
            params=model.mapping.parameters(), lr=self.lr, weight_decay=self.wd
        )
        return (
            optimizer_src,
            optimizer_tgt,
            optimizer_meta,
            optimizer_aug,
            optimizer_map,
        )

    def eval_mae(self, model, data_loader, stage):
        print("Evaluating MAE:")
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred = model(X, stage)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return (
            loss(targets, predicts).item(),
            torch.sqrt(mse_loss(targets, predicts)).item(),
        )

    def train(
        self,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        epoch,
        stage,
        mapping=False,
    ):
        print("Training Epoch {}:".format(epoch + 1))
        model.train()

        # model.train()は、PyTorchで定義されているモデルのメソッドで、モデルを訓練モードに設定します。
        # この設定は、特に訓練時に行われる特定の操作を有効にするために重要です。

        # 効果と目的

        # Dropoutの有効化:
        # 訓練モードでは、Dropout層が有効になります。
        # Dropoutは、訓練中にランダムにネットワークの一部のノード（ニューロン）を
        # 「ドロップアウト」させる（つまり一時的に無効にする）ことで、
        # モデルの過学習を防ぐために使用されます。
        # これにより、ネットワークが訓練データに過剰に適応するのを防ぎ、一般化能力を向上させる効果があります。

        # バッチ正規化の動作変更:
        # バッチ正規化（Batch Normalization）層も、訓練モードと評価モードで
        # 異なる動作をします。訓練モードでは、各バッチの平均と分散をリアルタイムで計算し、
        # これを使ってデータを正規化します。これにより、内部の共変量シフトを減少させ、
        # 訓練プロセスを安定させ、加速させることができます。

        # 使用例
        # train関数内でmodel.train()が呼び出されることにより、ネットワークはこれらの訓練固有の機能を
        # 適切に利用する準備が整います。これは、モデルが新しいデータに適切にフィットするように学習する際に重要です。
        # 以下は一般的な訓練ループの一部を示します：
        # 
        # def train(data_loader, model, criterion, optimizer, scheduler, epoch, stage):
        #     model.train()  # モデルを訓練モードに設定
        #     for X, y in data_loader:
        #         optimizer.zero_grad()  # 勾配をゼロで初期化
        #         output = model(X)      # モデルによる予測
        #         loss = criterion(output, y)  # 損失の計算
        #         loss.backward()       # バックプロパゲーション
        #         optimizer.step()      # パラメータの更新
        #         scheduler.step()      # 学習率スケジューラの更新
        # 
        # このコードでは、モデルが各イテレーションで訓練データに基づいてどのように学習していくかを制御しています。
        # model.train()はこの学習プロセスが正しく行われるための必要な設定を行います。

        for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            if mapping: 
                # mapping=True モデルの異なるコンポーネント間で特徴表現をマッピングするための学習に使用
                # EMCDR only
                src_emb, tgt_emb = model(X, stage)
                loss = criterion(src_emb, tgt_emb)
            else: # mapping=False 訓練プロセスで使用
                pred = model(X, stage)   # model の返り値が予測値 y !!
                loss = criterion(pred, y.squeeze().float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # train関数内でのmappingパラメータの使用は、異なる学習タスクを行うための条件分岐を提供しています。
            # このパラメータによって、モデルの動作や損失計算の方法が変わります。
            # 具体的には、mapping=Trueとmapping=Falseの場合で異なるタイプのデータ処理や学習が行われます。

            # mapping = False の場合
            # この設定は、通常の訓練プロセスで使用されます。モデルは入力データ X から予測 を生成し、
            # 生成された予測と実際のラベル y を比較して損失を計算します。
            # ここで使用される損失関数は、タスクに応じた適切なものを選択します
            # （例えば、回帰タスク（数値を予測するタスク）ではMSE、分類タスクではクロスエントロピーなど）。

            # mapping = True の場合
            # この設定は、特にモデルの異なるコンポーネント間で特徴表現をマッピングするための学習に使用されます。
            # 例えば、ユーザーIDを新しい特徴空間にマッピングするなどのタスクです。
            # この場合、model(X, stage)の呼び出しは、
            # ソース特徴表現 と ターゲット特徴表現 の両方を返します（src_embとtgt_emb）。
            # 損失関数はこれら二つの表現間の距離または類似度を測るものを使用し、
            # 例えばコサイン類似度損失やトリプレット損失などが考えられます。

            # このように異なる学習タスクに対応するため、
            # mappingパラメータを使用してモデルの動作を調整することが可能です。
            # これにより、一つの学習ループ内で多様な学習戦略を実装する柔軟性が得られ、
            # 効率的に複数のタスクを扱うことができます。

            # 目的と利点
            # 多様な学習タスクの統合: 
            # 一つのモデル内で異なるタイプの学習を行うことができ、
            # 訓練プロセスを柔軟に管理できます。

            # 特徴マッピングの強化: 
            # 異なる特徴空間間の関連を学習することで、
            # モデルがよりリッチな特徴表現を生成するのを助け、
            # 特定のタスクでのパフォーマンス向上が期待できます。

            # エンドツーエンドの最適化: 
            # 複数のモデルコンポーネントを同時に最適化することで、
            # 全体としてのモデルの一貫性と効率が向上します。
            # 
            # このような設計は、特に複雑なシステムやマルチタスク学習フレームワークにおいて有効です。

    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + "_mae"]:
            self.results[phase + "_mae"] = mae
        if rmse < self.results[phase + "_rmse"]:
            self.results[phase + "_rmse"] = rmse

    def TgtOnly(self, model, data_tgt, data_test, criterion, optimizer):
        scheduler_tgt = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.02,
            epochs=self.epoch,
            steps_per_epoch=len(data_tgt),
            pct_start=0.2,
        )
        print("=========TgtOnly========")
        for i in range(self.epoch):
            self.train(
                data_tgt,
                model,
                criterion,
                optimizer,
                scheduler_tgt,
                i,
                stage="train_tgt",
            )
            mae, rmse = self.eval_mae(model, data_test, stage="test_tgt")
            self.update_results(mae, rmse, "tgt")
            print("MAE: {} RMSE: {}".format(mae, rmse))

        # TgtOnly関数の処理の詳細
        # 訓練用スケジューラの設定:

        # torch.optim.lr_scheduler.OneCycleLRを使用して、学習率のスケジューリングを行います。
        # このスケジューラは、学習プロセスを通じて学習率を動的に調整し、最適な収束を促すように設計されています。
        # 訓練プロセスの実行:

        # 設定されたエポック数だけ訓練ループを実行します。各エポックで、ターゲットデータセット(data_tgt)を
        # 使用してモデルを訓練します。ここでの訓練は、train関数を呼び出して行われ、
        # モデルのパラメータ更新や損失の計算が含まれます。
        # 性能評価:

        # 訓練されたモデルを使用してテストデータセット(data_test)に対する性能を評価します。
        # eval_mae関数を使用して、モデルの平均絶対誤差(MAE)と平方根平均二乗誤差(RMSE)を計算し、
        # これらの指標を使用してモデルの予測精度を測定します。
        # 結果の更新:

        # 得られたMAEとRMSEの値を、クラス変数self.resultsに保存し、これにより訓練プロセスの進行状況と性能を追跡します。
        # 目的と利点
        # ターゲット特化の学習: TgtOnly関数は、ターゲットドメインのデータのみを使用してモデルを訓練することにより、
        # ターゲット特有の特徴やパターンをより効果的に捉えることができます。
        # 過学習のリスク軽減: ターゲットデータに特化して訓練することで、
        # ターゲットドメインにおける過学習のリスクを軽減し、
        # より堅牢なモデルを構築することが可能です。
        # 性能評価の明確化: テストデータセットに対する明確な性能評価を行うことで、
        # モデルの実際の適用能力を正確に把握し、
        # 必要に応じて改善策を講じることができます。
        # この関数は、特定のターゲットドメインへの適応性を高め、
        # そのドメインに最適化された予測モデルを開発するために特に有用です。

    def DataAug(self, model, data_aug, data_test, criterion, optimizer):
        print("=========DataAug========")
        scheduler_aug = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.02,
            epochs=self.epoch,
            steps_per_epoch=len(data_aug),
            pct_start=0.2,
        )
        for i in range(self.epoch):
            self.train(
                data_aug,
                model,
                criterion,
                optimizer,
                scheduler_aug,
                i,
                stage="train_aug",
            )
            mae, rmse = self.eval_mae(model, data_test, stage="test_aug")
            self.update_results(mae, rmse, "aug")
            print("MAE: {} RMSE: {}".format(mae, rmse))

        # DataAug関数は、データ拡張を用いてモデルのトレーニングを行うための関数です。
        # この関数の目的は、訓練データの多様性を高めることにより、モデルの一般化能力を向上させることにあります。
        # データ拡張は、既存のデータセットに対して様々な変換を適用し、
        # 実質的に新しいトレーニングサンプルを生成することによって行われます。
        # これにより、モデルが訓練中に見るデータのバリエーションが増え、
        # 未見のデータに対する予測能力が向上することが期待されます。

        # DataAug関数の処理の詳細
        # 訓練用スケジューラの設定:

        # torch.optim.lr_scheduler.OneCycleLRを使用して、学習率のスケジューリングを行います。
        # このスケジューラは、学習プロセスを通じて学習率を動的に調整し、モデルの収束を助けるように設計されています。
        # 訓練プロセスの実行:

        # 指定されたエポック数だけ訓練ループを実行します。各エポックで、
        # データ拡張を施したデータセット(data_aug)を使用してモデルを訓練します。
        # 訓練はtrain関数を呼び出して行われ、この中でモデルのパラメータ更新や損失の計算が含まれます。
        # 性能評価:

        # 訓練されたモデルを使用してテストデータセット(data_test)に対する性能を評価します。
        # eval_mae関数を使用して、モデルの平均絶対誤差(MAE)と平方根平均二乗誤差(RMSE)を計算し、
        # これらの指標を使用してモデルの予測精度を測定します。
        # 結果の更新:

        # 得られたMAEとRMSEの値を、クラス変数self.resultsに保存し、
        # 訓練プロセスの進行状況と性能を追跡します。
        # 用途と利点
        # 一般化能力の向上: データ拡張を通じて、訓練データの表現を豊かにすることで、
        # モデルが過学習を防ぎ、実際の運用環境でのパフォーマンスを向上させることができます。
        # 堅牢なモデルの構築: さまざまなバリエーションのデータで訓練されることで、
        # モデルはより堅牢で、異なる状況に対応できるようになります。
        # 性能評価の正確化: 実際にデプロイされる際の環境を模倣したデータセットでモデルを評価することで、
        # その実用性をより正確に評価することができます。
        # DataAug関数は、モデルが未知のデータや変化する環境条件に対しても頑健に機能するように設計されているため、
        # 特にデータセットのサイズが小さい場合や多様性に欠ける場合に特に有用です。

    def CDR(
        self,
        model,
        data_src,
        data_map,
        data_meta,
        data_test,
        criterion,
        optimizer_src,
        optimizer_map,
        optimizer_meta,
    ):

        print("=====CDR Pretraining=====")
        # ソースドメインの学習
        scheduler_CDR = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_src,
            max_lr=0.02,
            epochs=self.epoch,
            steps_per_epoch=len(data_src),
            pct_start=0.2,
        )
        for i in range(self.epoch):
            self.train(
                data_src,
                model,
                criterion,
                optimizer_src,
                scheduler_CDR,
                i,
                stage="train_src",
            )

        print("==========EMCDR==========")
        # メタのデータを使うが、yは、評価値ではなく、０からのシリアルな値を使って学習する？
        scheduler_EMCDR = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_map,
            max_lr=0.02,
            epochs=self.epoch,
            steps_per_epoch=len(data_map),
            pct_start=0.2,
        )
        for i in range(self.epoch):
            self.train(
                data_map,
                model,
                criterion,
                optimizer_map,
                scheduler_EMCDR,
                i,
                stage="train_map",
                mapping=True,
            )
            mae, rmse = self.eval_mae(model, data_test, stage="test_map")
            self.update_results(mae, rmse, "emcdr")
            print("MAE: {} RMSE: {}".format(mae, rmse))

        print("==========Deep PTUPCDR==========")
        scheduler_DPTUPCDR = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_meta,
            max_lr=0.02,
            epochs=self.epoch,
            steps_per_epoch=len(data_meta),
            pct_start=0.2,
        )
        for i in range(self.epoch):
            self.train(
                data_meta,
                model,
                criterion,
                optimizer_meta,
                scheduler_DPTUPCDR,
                i,
                stage="train_meta",
            )
            mae, rmse = self.eval_mae(model, data_test, stage="test_meta")
            self.update_results(mae, rmse, "dptupcdr")
            print("MAE: {} RMSE: {}".format(mae, rmse))

    def main(self):

        # modelを取得
        model = self.get_model()

        # dataこのデータは、データセット形式を取得
        data_src, data_tgt, data_meta, data_map, data_aug, data_test = self.get_data()

        # modelの最適化関数を取得
        optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map = (
            self.get_optimizer(model)
        )

        # 損失関数をセット
        criterion = torch.nn.MSELoss()

        # ターゲットのみで、学習とテストした場合
        self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)

        # ソースとターゲットを結合したデータで、学習とテストした場合
        self.DataAug(model, data_aug, data_test, criterion, optimizer_aug)

        # EMCDRとDPTUPCDRで、学習とテストした場合
        self.CDR(
            model,
            data_src,  # ソースドメインの学習
            data_map,  # EMCDRで学習
            data_meta,  # DPTUPCDRで学習
            data_test,  # testデータ
            criterion,
            optimizer_src,  # ソースドメインの最適化関数
            optimizer_map,  # EMCDRの最適化関数
            optimizer_meta,  # DPTUPCDRの最適化関数
        )
        print(self.results)
