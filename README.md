# DIT5411 繁體中文手寫辨識

用 TensorFlow 辨識繁體中文手寫字。資料集共 13,065 個字，模型最後一層是 13,065 個神經元。

繳交前請把 GitHub 專案名稱改成 `DIT5411-你的姓名`。這個資料夾裡的程式可以直接改名後上傳。

## 資料集

使用 [Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)。完整版在 [handwritting_data_all](https://github.com/chenkenanalytic/handwritting_data_all)，共 13,065 個字、約 684,677 張圖，原圖 300×300。

常用字子集只有 4,803 個字。那個子集不能拿來繳交，因為輸出層會少於 13,065。

資料集授權為 CC BY-NC-SA 4.0。使用時請保留下面這段出處：

> 本數據集由 AI . FREE Team 改作開發自 [STUST EECS_Chinese MNIST(總集)]。如有使用、改作、分享，請註明出處及此訊息。
>
> The dataset is AI . FREE Team development from [STUST EECS_Chinese MNIST(總集)]. If used, modified, or shared, please cite the source and the message.
>
> (source: https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)

## 訓練集與測試集

同一個字的檔名做自然排序（`字_2` 會排在 `字_10` 前面）。

- 該字至少有 41 張：前 40 張做訓練，剩下的做測試。
- 不足 41 張：大約 80% 做訓練、20% 做測試，並至少留 1 張測試。完整資料集平均每個字約 50 張，所以大多數是 40 張訓練、其餘測試。

測試圖片不做任何幾何變換。

## 資料增強

每個字只有約 40 張訓練原圖，直接訓練容易把筆跡背下來。這裡用 OpenCV 的 `warpAffine` 做旋轉、剪切、縮放，把每個字補到至少 200 張訓練圖。

40 張原圖各做 5 個版本，剛好 200 張：

| 版本 | 內容 |
| --- | --- |
| original | 原圖 |
| rotate_p12 | 旋轉 +12° |
| rotate_m12 | 旋轉 −12° |
| shear_x | 水平剪切 0.25 |
| scale_080 | 縮放 0.8 |

變換順序是先縮放、再剪切、再旋轉，都以圖片中心為準。邊界顏色用四個角的中位數，避免轉正之後出現一塊黑邊或白邊。

原圖先縮到 128×128 再做變換，最後縮成模型輸入 64×64。300×300 直接送進網路會很慢，64×64 在 Colab 上比較跑得完；筆畫複雜的字可以改 `--img-size 96`，但準備資料和訓練都要用同一個大小。

若某個字可用的訓練原圖少於 40 張，程式會再用更多旋轉、剪切、縮放組合，補到至少 200 張。增強是固定參數，不是每個 epoch 隨機重抽，所以結果可以重現。

預覽一張字的五個版本：

```powershell
python preview_augmentation.py --zip data/all_data.zip --out results/augmentation_preview.png
```

做法參考 [Introduction-to-OpenCV](https://github.com/darshitajain/Introduction-to-OpenCV) 裡的旋轉、仿射剪切與縮放。

## 模型

三個模型都是灰階 64×64 輸入、`sparse_categorical_crossentropy`、Adam（學習率 0.001）、softmax 寬度等於字數。訓練 epoch 數相同。測試集只在最後評估一次，不用來挑 epoch，避免把測試成績越調越高。

辨識率指測試集 Top-1 accuracy。因為有 13,065 個字，另外記錄 Top-5。

| 模型 | 架構 |
| --- | --- |
| `shallow_cnn` | Conv32 ×2、池化、Conv64 ×2、池化、Flatten、Dense256、Dropout、softmax |
| `deep_cnn` | 四段 Conv–BatchNorm–ReLU（通道 32/64/128/256）、全域平均池化、Dense512、Dropout、softmax |
| `residual_cnn` | 32 通道開頭，加上 stride 為 1 或 2 的殘差區塊（32/64/128/128）、全域平均池化、Dropout、softmax |

`deep_cnn` 與 `residual_cnn` 在分類層之前做全域平均池化，是為了不要在 13,065 類前面接一個超大的全連接層。

學習率若連續一個 epoch 沒有讓訓練 loss 下降，就減半，最低到 1e-5。

## 環境

- Python 3.9、3.10、3.11、3.12 或 3.13
- TensorFlow 2.16 以上（這份程式在 2.20 上寫的）

這台電腦預設的 Python 3.14 還沒有 TensorFlow 安裝包。請用下面的 3.9 虛擬環境，或在 Colab 上跑。

正式訓練請用有 GPU 的 Colab。13,065 個字、每個字 200 張，訓練集約 261 萬張。CPU 可以跑通程式，但三個模型各訓練數個 epoch 會非常久。

## 本機執行

在專案資料夾：

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py --variant full --output data
.\.venv\Scripts\python.exe prepare_records.py --zip data/all_data.zip --index-only
.\.venv\Scripts\python.exe prepare_records.py --zip data/all_data.zip --out data/records
.\.venv\Scripts\python.exe train.py --records data/records --epochs 5 --batch-size 128
```

`--index-only` 只看壓縮檔目錄，用來確認解出來的字數是不是 13,065、檔名有沒有亂碼。確認之後再寫 TFRecord。

準備資料會把增強後的圖寫進 `data/records`。磁碟請預留大約 15 GB（壓縮檔約 1.1 GB，TFRecord 數 GB）。

Colab 若中途斷線，用同一組參數加上 `--resume` 再跑，會從上次存下來的 epoch 繼續。

訓練結束後看 `results/comparison.md`。把那張表的數字貼回這份 README 的「結果」一節。權重在 `models/`，預測：

```powershell
.\.venv\Scripts\python.exe predict.py --image 某一張圖.png
```

## Colab

1. 執行階段改成 GPU。
2. 把這個專案推上 GitHub 後，在筆記本 `notebooks/DIT5411_colab.ipynb` 裡改成你的 repo 網址。
3. 依序執行下載、`--index-only`、準備 TFRecord、訓練。
4. 下載 `results/comparison.md` 與 `results/comparison.png`，數字貼回 README。

## 結果

下表要等完整 13,065 類訓練結束才填。只跑 `--max-classes` 的數字不能放進來。

| 模型 | 測試辨識率 Top-1 | Top-5 |
| --- | --- | --- |
| shallow_cnn | 訓練後填入 | 訓練後填入 |
| deep_cnn | 訓練後填入 | 訓練後填入 |
| residual_cnn | 訓練後填入 | 訓練後填入 |

最佳模型：（訓練後填入）

`results/comparison.md` 會由 `train.py` 自動寫好，直接把內容補到上面即可。

## 參考

- AI.FREE Team，Traditional Chinese Handwriting Dataset：https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset
- 完整資料集：https://github.com/chenkenanalytic/handwritting_data_all
- OpenCV 影像變換範例：https://github.com/darshitajain/Introduction-to-OpenCV
- README 寫法參考：https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow
- He 等人，Deep Residual Learning for Image Recognition（residual_cnn 的捷徑結構）
- TensorFlow：https://www.tensorflow.org/
