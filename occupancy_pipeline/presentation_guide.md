# NuScenes Occupancy 生成技術詳解 (Presentation Guide)

這份文件是為了幫助您製作簡報而寫的，詳細解釋了程式碼背後的**算法原理**，特別是**多幀堆疊 (Multi-frame Stacking)** 的部分。

---

## 1. 為什麼需要堆疊？ (The "Why")

### 原始 LiDAR 的問題
*   **稀疏 (Sparse)**：LiDAR 只有 32 或 64 條線，打在物體上只是幾個點。
*   **遮擋 (Occlusion)**：被前面的車擋住，後面的路就看不到了。
*   **盲區**：車頂 LiDAR 看不到極近處的地面。

### 解法：利用時間換取空間 (Time-to-Space)
*   車子在移動，上一秒看到的「左邊建築物」，這一秒可能到了「後方」。
*   如果我們把**過去 10 幀 (約 0.5~1.0 秒)** 的點雲全部搬到**現在**這一幀來，點雲就會變得很密。
*   這就是 **"Densification using Multi-frame Stacking"**。

---

## 2. 核心算法流程 (The "How")

### 步驟概覽
1.  **分離 (Separation)**：把點雲分成「靜態背景」與「動態物體」。
2.  **靜態對齊 (Static Alignment)**：利用車身姿態 (Ego Pose) 對齊背景。
3.  **動態對齊 (Dynamic Alignment)**：利用標註框 (Bounding Box) 對齊移動物體。
4.  **體素化 (Voxelization)**：降採樣成格子。

---

### 3. 詳細技術實作 (對應 `data_converter.py`)

#### A. 靜態場景堆疊 (Static Stacking)
*   **判斷標準**：利用 `lidarseg` 標籤。如果標籤屬於 `vegetation`, `driveable_surface`, `building` 等 (ID 24-30)，視為靜態。
*   **座標轉換**：
    我們需要將 `t-1` 時刻的點 $P_{t-1}$ 轉換到 `t` 時刻 $P_t$。
    
    公式：
    $$ P_{global} = R_{t-1} \cdot P_{t-1} + T_{t-1} $$  (先轉到世界座標)
    $$ P_{t} = R_{t}^{-1} \cdot (P_{global} - T_{t}) $$    (再轉回當前車身座標)
    
    *程式碼對應函數：`prev2ego`*

#### B. 動態物體對齊 (Dynamic Object Alignment) 🌟 *重點*
這是最難的部分。如果直接用上面的公式轉，移動中的車子會出現**殘影 (Ghosting)**（因為車子自己在動，不只受車身運動影響）。

*   **判斷標準**：`instance_token` 對應的物體，且在當前幀也存在。
*   **對齊邏輯**：
    1.  **以舊 Box 為中心**：把點雲從世界座標轉回「舊 Bounding Box」的局部座標 (Local Coordinates)。
        *   這時候，點雲是相對於「那台車」的位置。
    2.  **移動 Box**：把這個局部座標系，平移旋轉到「新 Bounding Box」的位置。
    3.  **還原**：再把點雲轉回車身座標。
    
    *程式碼對應函數：`align_dynamic_thing`*
    ```python
    # 1. 轉回舊 Box 的局部座標
    box_points = rotate(box_points, inv(prev_rot), center=prev_bbox_center)
    
    # 2. 跟隨 Box 移動到新位置 (Target Box)
    box_points = translate(box_points, ego_boxes_center - prev_bbox_center)
    box_points = rotate(box_points, ego_boxes_rot, center=ego_boxes_center)
    ```

---

## 4. 流程圖總結 (Workflow Summary)

```mermaid
graph TD
    A[輸入: 當前幀 t, 過去幀 t-1...t-N] --> B{點是靜態還是動態?}
    
    B -- 靜態 (路面/建築) --> C[使用 Ego Pose 轉換]
    C --> C1[轉到 Global] --> C2[轉回 Frame t]
    
    B -- 動態 (車/人) --> D[使用 Bounding Box 轉換]
    D --> D1[轉到舊 Box 局部空間] --> D2[跟隨 Box 移動到新 Box 位置]
    
    C2 --> E[合併所有點雲 (Stacking)]
    D2 --> E
    
    E --> F[Voxelization (0.2m 格子化)]
    F --> G[輸出 Occupancy Grid]
```

## 5. 簡報建議 (Tips)
*   **關鍵詞**：Ego-motion Compensation (自身運動補償), Object-centric Alignment (以物體為中心的對齊).
## 6. 為什麼會有殘影？(Ghosting Analysis) - 進階詳解 🕵️‍♂️

如果您自己實作堆疊時出現殘影，通常是因為**只做了「車身補償」，沒做「物體補償」**。

### 殘影成因範例
假設有一台前車 $O$，在 $t-1$ 和 $t$ 時刻的位置不同：
1.  **t-1 時刻**：前車在距離我們 10公尺處 ($P_{t-1} = [10, 0, 0]$)。
2.  **t   時刻**：前車往前開了，距離我們變 15公尺處 ($P_{t} = [15, 0, 0]$)。

如果您只用 `Ego Pose` 把 $t-1$ 的點轉過來：
*   算法會以為「那是靜止的牆」，所以把它轉到相對應的位置（可能還是算出約 10公尺處）。
*   結果：您現在的 LiDAR 看到車在 15m，但堆疊過來的舊點在 10m。
*   **視覺效果**：車子後面拖了一長串的尾巴（殘影）。

### 正確解法：Object-Centric Alignment (以物體為中心的對齊)

我們必須「把過去那台車上的點，剪下來，貼到現在這台車的位置上」。

#### 數學推導 (Step-by-Step Math)

假設我們有一個點 $P_{world}$ (世界座標)，它是屬於某個動態物體（如前車）的。

1.  **世界轉局部 (World to Local @ t-1)**:
    我們先算 $P$ 相對於「$t-1$ 時刻的 Box」在哪裡。
    $$ P_{local} = R_{box, t-1}^{-1} (P_{world} - T_{box, t-1}) $$
    *(這步算出：點在車子左前方 1公尺高的地方)*

2.  **跟隨物體移動 (Move with Object)**:
    因為點是黏在車上的，所以 $P_{local}$ 應該保持不變。我們把它套用到「$t$ 時刻的 Box」上。
    $$ P'_{world} = R_{box, t} \cdot P_{local} + T_{box, t} $$
    *(這步算出：現在這台車子左前方 1公尺高的地方，在世界座標哪裡)*

3.  **世界轉車身 (World to Ego @ t)**:
    最後轉回我們自己的 LiDAR 座標系。
    $$ P'_{ego} = R_{ego, t}^{-1} (P'_{world} - T_{ego, t}) $$

#### 程式碼對照 (`data_converter.py`: align_dynamic_thing)

```python
# 1. 轉回舊 Box 的局部座標 (Local Frame of Prev Box)
# 這裡的 box 是 t-1 時刻的 box
box_points = rotate(box_points, np.linalg.inv(prev_rotate_matrix), center=prev_bbox_center)

# 2. 跟隨 Box 移動 (Transform to Current Box Pose)
# target 是 t 時刻的 box
target = ego_frame_info['instance_tokens'].index(prev_instance_token)
ego_boxes_center = ego_frame_info['boxes'][target].center
ego_boxes_rot = ego_frame_info['boxes'][target].rotation_matrix

# 先移到當前 Box 的中心
box_points = translate(box_points, ego_boxes_center - prev_bbox_center)
# 再轉到當前 Box 的角度
box_points = rotate(box_points, ego_boxes_rot, center=ego_boxes_center)
```

**結論**：要消除殘影，您必須知道**每個點屬於哪個 Instance** (利用 `lidarseg`)，並對每個 Instance 分別做上述的 Box 變換。

## 7. 靜態場景的殘影與模糊 (Static Ghosting) 🏗️

如果您發現**連路邊的房子、紅綠燈**都有殘影或變厚，通常是以下原因：

#### A. 忽略了傳感器外參 (Sensor Extrinsics)
LiDAR 一般裝在車頂，不是在車身中心 (Ego Center)。
*   **錯誤**：直接拿 LiDAR 點 {lidar}$ 當作車身相對座標。
*   **正確**：必須先轉到車身。
    402119 P_{ego} = R_{sensor} \cdot P_{lidar} + T_{sensor} 402119
    *(如果這一步沒做，車子轉彎時，所有點雲都會「甩」出去，造成疊合失敗)*

#### B. 運動畸變 (Motion Distortion / Rolling Shutter Effect) 🌪️
這是最高階的誤差來源。
*   **現象**：LiDAR 掃描一圈需要 0.05 秒。這期間車子可能往前開了 1 公尺。
*   **問題**：LiDAR **開頭**打到的點 (0ms) 和 **結尾**打到的點 (50ms)，車子位置其實不一樣！但我們通常假設它們是「同一個時間」拍的。
*   **結果**：高速行駛或轉彎時，點雲會變形（房子變歪）。堆疊時就會對不準。
*   **解法 (Deskewing)**：
    使用 NuScenes SDK 的 `from_file(..., sweep_info)` 或自行根據每個點的 `time_lag` 進行微調補償。
    (註：本專案 `data_converter.py` 主要依賴 NuScenes 提供的 Pose 精度，對於極高速場景可能仍由原始數據決定，但 NuScenes 此數據集通常已做過品質控制)。

#### C. 座標轉換順序錯誤 (Transform Order) - 核心鍊條詳解 🔗
這就是您提到的「先轉到車身，再轉到車身」的意思。其實是從**舊車身**轉到**新車身**。

完整的轉換鍊條 (Chain) 如下：

1.  **Sensor to Ego ($t-1$)**:
    *   先修正安裝誤差 (Extrinsics)。
    *   $P_{ego1} = T_{lidar \to ego} \cdot P_{lidar1}$
    *   *此時點雲是相對於「上一秒的車子中心」。*

2.  **Ego to Global ($t-1$)**:
    *   把上一秒的車子放到世界地圖上 (Pose)。
    *   $P_{global} = T_{ego1 \to global} \cdot P_{ego1}$
    *   *此時點雲是絕對座標 (例如 GPS 座標)。*

3.  **Global to Ego ($t$)**:
    *   從世界地圖，轉回「現在這秒的車子中心」。
    *   $P_{ego2} = T_{global \to ego2} \cdot P_{global}$
    *   *此時點雲跟著現在的車子跑了！*

4.  **Ego to Sensor ($t$)** (選用):
    *   如果最後要在 LiDAR 座標系下看，再轉回傳感器。
    *   $P_{lidar2} = T_{ego \to lidar} \cdot P_{ego2}$

**口訣**：「先上車(舊)，出門(世界)，再上車(新)，回家(雷達)」。

$$ P_{final} = T_{Ego2 \to Lidar2} \cdot T_{Global \to Ego2} \cdot T_{Ego1 \to Global} \cdot T_{Lidar1 \to Ego1} \cdot P_{initial} $$

## 8. 直觀理解：為什麼可以疊得這麼好？ (Intuition: Why it works?) 💡

您可能會覺得神奇：車子一直在動，為什麼點雲疊在一起不會糊掉？

### A. 靜態場景：就像「3D 全景拼圖」 (3D Panorama)
想像您拿著攝影機走進一個房間錄影。
*   雖然每一幀畫面都只拍到房間的一部分。
*   但如果您**精確知道**自己每一秒站在哪裡 (Pose)。
*   您就可以把每一幀畫面，貼回到房間的正確位置上。
*   當您走了 10 秒鐘，您就貼滿了整個房間的模型。
*   **關鍵**：就在於 NuScenes 提供的 Ego Pose (定位) 非常準！所以拼圖拼得很完美。

### B. 動態物體：就像「便條紙/貼紙」 (Sticker Strategy)
 moving object 比較麻煩，因為它自己在動。
但我們的算法把它當作一張**貼紙**：
1.  **Peel (撕下來)**：我們知道哪裡有車 (Bounding Box)，先把那台車的點雲從背景「撕下來」(轉成局部座標)。
2.  **Move (移動手)**：把手移動到現在的位置。
3.  **Stick (貼上去)**：把這張貼紙貼在現在的位置上。

透過這種方式，不管那台車過去在哪裡，我們都把它強行「抓」到現在的位置上。所以看起來就像車子一直停在您面前讓您掃描一樣，點雲自然就變密了！

## 9. 實戰：程式碼怎麼寫？ (How to Code It?) 💻

既然理解了原理，我們來看程式碼具體要怎麼實作這個「轉換鍊」。

### A. 獲取 Transform Matrix (從 SDK)
在 NuScenes SDK 中，我們用  來拿 Pose。



### B. 執行轉換 (The "Chain")
數學上是矩陣相乘，但在 Python 裡我們通常分兩步做 (先旋轉再平移) 以節省運算。

**目標**：把點 {prev}$ (在 t-1 車身) 轉到 {curr}$ (在 t 車身)。



### C. 動態物體怎麼寫？ (Sticker Implementation)
動態物體需要 Bounding Box 的 Pose。



**小撇步**：這就是為什麼我們的  裡會有  (靜態用) 和  (動態用) 這兩個不同的函數。

## 10. 動態物件處理流程詳解 (Dynamic Object Pipeline Detail) 🚗

您問到了「動態物件具體怎麼處理」，除了上面的貼紙原理，整個系統的運作流程如下：

### Step 1: 辨識 (Identify) - 誰是動態的？
*   我們讀取 ，裡面每個點都有一個 。
*   如果這個 token 屬於 , ,  類別，我們就標記這些點是「動態候選人」。

### Step 2: 追蹤 (Track) - 它以前在哪？
*   對於每一個動態候選人 (Instance)，我們去查 NuScenes 的 。
*   找出它在 **上一幀 (t-1)** 的 Bounding Box {t-1}$。
*   找出它在 **現在 (t)** 的 Bounding Box {t}$。

### Step 3: 驗證 (Verify) - 它還在嗎？
*   如果在 -1$ 有這個物體，但在 $ 消失了（開走了/被擋住了）。
*   那我們就**直接丟棄**這些點，不進行堆疊，避免堆出一個「幽靈車」。

### Step 4: 轉換 (Transform) - 貼紙大法
*   確認兩幀都有它之後，就執行 **Object-centric Alignment**：
    1.  {world} \to P_{local\_prev}$ (相對於舊 Box)
    2.  {local\_prev} \to P_{world\_new}$ (貼到新 Box)
    3.  {world\_new} \to P_{ego}$ (轉回車身)

### Step 5: 融合 (Merge)
*   最後把這些「搬過來」的動態點，跟原本的靜態背景點 ({static}$) 合併在一起。
*   這就是為什麼最後輸出的點雲裡，車子會變得很密實。

## 10. 動態物件處理流程詳解 (Dynamic Object Pipeline Detail) 🚗

您問到了「動態物件具體怎麼處理」，除了上面的貼紙原理，整個系統的運作流程如下：

### Step 1: 辨識 (Identify) - 誰是動態的？
*   我們讀取 `lidarseg`，裡面每個點都有一個 `instance_token`。
*   如果這個 token 屬於 `vehicle`, `human`, `animal` 類別，我們就標記這些點是「動態候選人」。

### Step 2: 追蹤 (Track) - 它以前在哪？
*   對於每一個動態候選人 (Instance)，我們去查 NuScenes 的 `sample_annotation`。
*   找出它在 **上一幀 (t-1)** 的 Bounding Box $B_{t-1}$。
*   找出它在 **現在 (t)** 的 Bounding Box $B_{t}$。

### Step 3: 驗證 (Verify) - 它還在嗎？
*   如果在 $t-1$ 有這個物體，但在 $t$ 消失了（開走了/被擋住了）。
*   那我們就**直接丟棄**這些點，不進行堆疊，避免堆出一個「幽靈車」。

### Step 4: 轉換 (Transform) - 貼紙大法
*   確認兩幀都有它之後，就執行 **Object-centric Alignment**：
    1.  $P_{world} \to P_{local\_prev}$ (相對於舊 Box)
    2.  $P_{local\_prev} \to P_{world\_new}$ (貼到新 Box)
    3.  $P_{world\_new} \to P_{ego}$ (轉回車身)

### Step 5: 融合 (Merge)
*   最後把這些「搬過來」的動態點，跟原本的靜態背景點 ($P_{static}$) 合併在一起。
*   這就是為什麼最後輸出的點雲裡，車子會變得很密實。

## 11. 關於幀數與時間範圍 (Frame Strategy) ⏳

您提到：「動態只看前一幀，但靜態是疊幾幀？」
這是一個非常重要的誤解！其實 **靜態和動態都是疊很多幀 (Multi-frame)**。

### A. 我們疊了多少幀？
在 `data_converter.py` 預設的 `--num_sweeps 10` 意思是：**雙向堆疊 (Bidirectional)**。

如果您現在是 **第 10 幀 (Frame t=10)**，算法會看：
1.  **過去 (Past)**：第 0 ~ 9 幀。
2.  **未來 (Future)**：第 11 ~ 20 幀。

總共範圍是： `[ t-10, ..., t, ..., t+10 ]`
所以是 **21 幀** (自己1幀 + 前10 + 後10) 的資料量！

> **為什麼要偷看未來？**
> 因為這是 "Offline" (離線) 處理。反正資料都錄好了，把未來的點搬回來，可以補足你看不到的死角 (例如剛開過路口，回頭看才知道那裡有什麼)。

### B. 運作迴圈 (The Loop)
那個「貼紙大法」和「Ego 轉換」，是放在一個 `for` 迴圈裡執行的：

```python
# 虛擬碼概念
for k in range(1, 11):
    frame_prev = get_frame(t - k) # 拿前 k 幀
    
    # 1. 靜態處理
    static_points = align_static(frame_prev, frame_curr)
    all_points.add(static_points)
    
    # 2. 動態處理
    dynamic_points = align_dynamic(frame_prev, frame_curr) # 每一幀都做一次 Object alignment!
    all_points.add(dynamic_points)
```

所以：
*   **靜態**：每一幀的路面，都透過 Ego Pose 搬過來。
*   **動態**：每一幀的那台車，都透過 Object Pose 搬過來。

我們剛剛舉例用 "$t$ 和 $t-1$" 只是為了方便解釋「兩幀之間怎麼對齊」，但實際上這個動作被重複執行了 20 次 (前後各10幀)。這就是為什麼點雲會這麼密！

## 12. 進階答疑：物體會變長變短嗎？(Object Deformation) ��

您問得非常深！答案是：**會，是有可能的！**

這通常被稱為 **"Smearing" (拖影)** 或 **"Scaling Artifacts"**。主要原因有兩個：

### A. 插值誤差 (Interpolation Error)
*   **數據來源**：NuScenes 的人工標註框 (Bounding Box) 其實是 **2Hz** (每 0.5 秒標一次)。
*   **LiDAR 頻率**：但 LiDAR 是 **10Hz** (每 0.1 秒掃一次)。
*   **問題**：中間那 4 幀的 Box 位置，是用「線性插值 (Linear Interpolation)」算出來的。
    *   如果那台車正在 **急煞車** 或 **急轉彎**（非線性運動），我們算出來的 Box 就會稍微偏離它的真實位置。
    *   **結果**：點雲疊加上去時，可能會疊得太外面（車變長）或太裡面（車變短/糊掉）。

### B. 掃描畸變 (Rolling Shutter on Objects)
*   LiDAR 掃描一圈需要時間。當雷射光掃到「車尾」時，跟掃到「車頭」時，那台車其實已經往前移動了一點點。
*   這會導致那台車的點雲本身就是**歪的 (Sheared)**。
*   當我們硬要把這個歪的點雲塞進一個正的 Box 裡，就會產生形變。

**結論**：雖然我們的算法已經盡力了 (Object-centric Alignment)，但在高速運動或非線性運動下，**些微的變長、變胖或邊緣模糊是正常的物理現象**。這也是為什麼做 3D Occupancy 很難達到 100% 完美的像積木一樣整齊。

## 13. 體素化流程詳解 (Voxelization Process) 🧊

當我們有了「堆疊後的稠密點雲」之後，最後一步就是把它變成 Minecraft 一樣的格子 (Voxels)。

### A. 直觀想像：雞蛋盒 (The Egg Carton Analogy) 🥚
想像地板上有一萬顆散落的彈珠 (點雲)，而您手上有一個巨大的「雞蛋盒」 (Grid)。
*   **Voxelization (體素化)**：就是看「每一顆彈珠掉進哪一個蛋坑裡」。
*   **Occupancy (佔用)**：如果某個蛋坑裡有至少一顆彈珠，我們就在那裡做個記號 (1)；如果蛋坑是空的，就是 (0)。

### B. 定義範圍 (Grid Definition)
我們把世界劃分成無數個小格子：
*   **範圍 (Bounds)**：$X, Y \in [-60m, 60m]$， $Z \in [-5m, 11m]$ (總長 120m)
*   **格子大小 (Resolution)**：$0.2m$ (20公分)

這就像是我們的雞蛋盒有 $600 \times 600 \times 80 = 2880$ 萬個坑。

### C. 數學公式：怎麼算第幾格？ (Quantization Formula)
假設有一顆彈珠掉在 $P(1.5, -40.0, 0)$ 的位置，它屬於第幾號格子？

公式：$Index = \lfloor \frac{Coordinate - MinBound}{VoxelSize} \rfloor$

1.  **X軸計算**：
    *   座標 $x = 1.5$
    *   起點 $min\_x = -60$
    *   距離起點：$1.5 - (-60) = 61.5$ 公尺
    *   格子數：$61.5 / 0.2 = 307.5$
    *   取整數：**第 307 號格子**

2.  **Y軸計算**：
    *   座標 $y = -40.0$
    *   起點 $min\_y = -60$
    *   距離起點：$20.0$ 公尺
    *   格子數：$20.0 / 0.2 = 100$
    *   取整數：**第 100 號格子**

所以這顆彈珠的地址是 `(307, 100, ...)`。

### D. 衝突處理 (Collision Handling) 🤜🤛
如果一個格子裡掉進了兩顆不同顏色的彈珠（比如一顆是紅色車子，一顆是綠色樹葉），怎麼辦？
*   **Majority Vote (多數決)**：看紅色多還是綠色多。
*   **Priority (優先權)**：車子比較重要，優先標記為車子。
*   **Simple (簡化)**：隨機取一顆（本專案採用此法，速度最快）。

### E. 稀疏儲存 (Sparse Storage)
因為空氣佔了絕大多數，如果把 2800 萬個 0 都存下來太浪費了。
我們只存 **「有彈珠的座標」** (Indices) 和 **「它的顏色」** (Label)。
這就是為什麼最後輸出的 `.npz` 只有幾 MB，而不是幾百 MB。

## 14. 系統參數與資料總表 (System Parameters & Data Checklist) 📋

為了讓您的簡報更清楚，這裡列出了所有用到的資料與設定參數。

### A. 輸入資料 (Input Data)
我們從 NuScenes 資料集中讀取了以下內容：

| 資料類型 (Data Type) | 具體項目 (Specific Items) | 用途 (Purpose) |
| :--- | :--- | :--- |
| **感測器數據** | `LIDAR_TOP` | 主要的點雲來源 (Point Cloud Basis)。 |
| | `CAM_FRONT`, `CAM_BACK`... (共6顆) | 用於視覺化展示或多模態融合 (本專案僅作輸出)。 |
| **標註數據** | `lidarseg` | 點雲語意分割標籤 (辨識道路、車輛)。 |
| | `sample_annotation` | 3D Bounding Boxes (用於動態物件追蹤)。 |
| | `instance_token` | 用於識別每一台獨一無二的車。 |
| **定位與校正** | `ego_pose` | 車輛在世界座標系的精確位置 (用於靜態堆疊)。 |
| | `calibrated_sensor` | 感測器相對於車身的安裝位置 (Extrinsics)。 |

### B. 核心參數 (Key Parameters)
這是我們生成 Occupancy Grid 的具體設定：

| 參數 (Parameter) | 設定值 (Value) | 說明 (Description) |
| :--- | :--- | :--- |
| **Voxel Size** | `0.2m` | 體素解析度 (CVPR 2023 標準)。 |
| **Scene Bounds (X/Y)** | `[-60.0m, 60.0m]` | 前後左右各 60公尺 (總長 120m)。 |
| **Scene Bounds (Z)** | `[-5.0m, 11.0m]` | 高度範圍 (含地面與高樓)。 |
| **Num Sweeps** | `10` | 堆疊幀數 (前後各 10 幀，共約 21 幀)。 |
| **Stacking Range** | `~ 1.0 Sec` | 因 LiDAR 10Hz，前後10幀約等於 1 秒鐘的歷史。 |

這張表可以直接放在簡報的 "Experimental Setup" 或 "Methodology" 頁面！

## 15. Ego Pose 與 Extrinsics 詳解 (Pose & Extrinsics Deep Dive) 📐

針對您最後提到的問題，這裡做深入拆解：

### A. Ego Pose 怎麼取得？
Ego Pose 代表「車身中心 (Ego)」在「世界地圖 (Global)」上的位置。
*   **來源**：NuScenes 已經幫我們算好了（融合了 GPS + IMU + 里程計）。
*   **程式碼取得方式**：
    ```python
    # 透過 sample_data 裡的 ego_pose_token 查表
    sd_rec = nusc.get('sample_data', lidar_token)
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    # 內容物
    trans = pose_record['translation'] # [x, y, z] (如: [300.5, 120.3, 0.5])
    rot   = pose_record['rotation']    # [w, x, y, z] (Quaternion)
    ```

### B. 外參 (Extrinsics) 用在哪裡？
外參代表「LiDAR 感測器」安裝在「車身 (Ego)」的哪個位置。
*   **為什麼重要？**：LiDAR 裝在車頂，距離地面可能有 1.8 公尺。如果不扣掉這個高度，我們建出來的地板就會浮在半空中！
*   **程式碼取得方式**：
    ```python
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    ```
*   **用在哪一步？**：用在轉換鍊的 **第一步** 和 **最後一步**。
    1.  $P_{lidar} \to P_{ego}$：加上外參 (把點雲從頭頂移到車中心)。
    2.  $P_{ego} \to P_{lidar}$ (選用)：減去外參 (如果要轉回頭頂視角)。

    $$ P_{ego} = R_{ext} \cdot P_{lidar} + T_{ext} $$
    *(這個 $T_{ext}$ 通常是 `[0.9, 0, 1.8]` 這樣的數值，代表 LiDAR 裝在前軸上方 1.8 米處)*

## 16. 終極圖解：什麼是 Ego 與 Extrinsics？ (The Ultimate Guide) 🧠

這兩個概念是所有自駕車座標系的基石，用一個生活化的比喻最容易懂。

### A. 外參 (Extrinsics) - 就像「戴帽子」 🧢
想像您是一台車 (Ego)。您的「中心」在您的**雙腳** (後軸中心)。
但您的眼睛 (LiDAR) 長在**頭頂**上。

*   **外參 (Extrinsics)**：就是從「雙腳」到「頭頂」的距離。
    *   不管您走到哪裡，您的頭永遠在腳上面 1.8 公尺處。
    *   這是一個**固定不變**的關係 (Rigid Transform)。
    *   **用途**：LiDAR 看到的數據是在「頭頂」的，我們要把它扣掉 1.8 公尺，才能換算成「雙腳」的位置。

### B. Ego Pose - 就像「走路的軌跡」 👣
想像您在一個巨大的操場 (Global World) 上走路。

*   **Ego Pose**：就是記錄您在每一秒鐘，雙腳站在操場的**哪個座標 (x, y)** 以及 **面朝哪裡 (heading)**。
*   **算法 (Algorithm)**：雖然我們直接查表，但 NuScenes 背後是怎麼算出來的？
    *   **SLAM (Simultaneous Localization and Mapping)**：結合了 GPS (大概位置)、IMU (加速度/轉向)、Wheel Odometry (輪子轉幾圈) 和 LiDAR Matching (比對四周特徵)。
    *   利用 **Kalman Filter (卡爾曼濾波)** 把這些數據融合，算出最可能的軌跡。
    *   這是一個**隨時間變動**的數值。

### C. 總串聯 (The Full Chain)
所以，當我們要把一個點雲放到地圖上：

```text
[點雲 Point] --(往下看)--> [外參 Extrinsics] --(你的身體)--> [車身 Ego] --(走路)--> [Ego Pose] --(操場)--> [世界 Global]
```

1.  **LiDAR 看到前面有一棵樹** (相對於頭頂)。
2.  **外參**：樹在頭頂前方 10米 = 樹在腳底前方 10米 + 高度修正 (相對於車身)。
3.  **Ego Pose**：腳底在操場的 (100, 200) 位置，面朝北。
4.  **Global**：所以樹在操場的 (100, 210) 位置。

### D. 如果外參錯了會怎樣？ (Consequences of Wrong Extrinsics) ⚠️

最直觀的比喻：**自拍棒效應 (The Selfie Stick Analogy)** 🤳

想像您手拿一支 2公尺長的自拍棒，原地旋轉一圈錄影。

1.  **如果有加外參 (Correct)**：
    *   電腦知道鏡頭離您 2公尺。
    *   計算時會把鏡頭的移動「扣掉」。
    *   **結果**：背景房子是**紋絲不動**的，只有您的臉在轉。

2.  **如果沒加外參 (Wrong)**：
    *   電腦以為鏡頭就是您的眼睛 (距離 0)。
    *   當您旋轉時，鏡頭實際上畫了一個大圓圈。
    *   電腦會以為「您真的位移了這麼遠」。
    *   **結果**：背景的房子會跟著鏡頭**一起平移 (Drift)**。
    *   **在點雲上的現象**：原本一面平平的牆，因為車子轉彎，疊成了**兩層**甚至是**模糊的一團**。

**結論**：外參就是為了消除這種「因為安裝位置不同」而產生的**假位移**。沒有它，車子一轉彎，地圖就糊了。

### E. 外參是一幀一組還是一景一組？ (Constant vs Variable?)
*   **答案**：其實是 **「一趟行程 (Log) 一組」**。
    *   只要感測器沒被拆下來重裝，它的位置就是固定的。
    *   所以同一個 Scene (20秒) 裡面，外參**絕對是固定的**。
*   **為什麼程式要每幀都讀？**
    *   為了程式寫法統一。NuScenes 的資料結構是 `SampleData` -> `CalibratedSensor`。
    *   雖然每一幀都去查表，但查出來的通常都是**同一個 Token** (同一組參數)。

### F. 車子行駛震動怎麼辦？ (Vibration Handling) 📉
這是一個很棒的問題！車子過減速帶或坑洞時會跳動，這會影響外參嗎？

*   **答案**：**不會影響外參，但會影響 Ego Pose。**
*   **原理**：
    1.  **外參 (Extrinsics)**：假設 LiDAR 是「焊死」在車頂上的 (剛性連接)。把車子想像成一個大鐵塊，不管鐵塊怎麼跳，頭頂跟腳底的距離是不會變的。
    2.  **Ego Pose**：當車子跳動 (Pitch/Roll 改變) 時，車上的 **IMU (慣性測量單元)** 會感測到這個傾斜。
    3.  **補償**：NuScenes 的算法會把這個「傾斜」更新到 **Ego Pose** 的 `rotation` 裡。
*   **結論**：震動被「吸收」進了 Ego Pose 裡。所以當我們做轉換時，點雲會跟著車身一起「歪」一下，剛好抵消掉路面的起伏，讓建出來的地圖保持水平。

## 17. 靜態堆疊詳解：透過「世界」來傳遞 (Static Stacking Revisited) 🌏

您問到「靜態物怎麼疊」，其核心觀念就是：**利用不動的世界 (Global) 當作橋樑**。

### A. 直觀比喻：釘圖釘 📌
想像一個巨大的軟木塞板 (世界座標 Global)，它是絕對不動的。
1.  **Frame 0 (過去)**：
    *   您看到了一棵樹。您根據當時站的位置 (Frame 0 Pose)，算出了那棵樹在軟木塞板上的位置。
    *   **動作**：把這棵樹「釘」在軟木塞板上。 (Point -> Global)
2.  **Frame 10 (現在)**：
    *   您往前走了 10 公尺 (Ego Moved)。
    *   現在您想知道那棵樹在哪裡？
    *   **動作**：回頭去看軟木塞板上的那個釘子。 (Global -> Point)

因為軟木塞板 (世界) 不會動，所以不管您怎麼走，只要透過這個板子轉換，樹的位置永遠是準的。

### B. 數學步驟 (The Math Flow)
程式碼 (`prev2ego`) 其實就是走了這條路：

1.  **脫掉帽子 (Extrinsics)**：
    $P_{ego\_old} = R_{ext} \cdot P_{lidar\_old} + T_{ext}$
    *(先把以前看到的點，換算成以前的車身座標)*

2.  **釘在牆上 (Ego to Global)**：
    $P_{global} = R_{pose\_old} \cdot P_{ego\_old} + T_{pose\_old}$
    *(利用以前的 Pose，算出點在世界的絕對座標)*

3.  **回頭看牆 (Global to Ego)**：
    $P_{ego\_new} = R_{pose\_new}^{-1} (P_{global} - T_{pose\_new})$
    *(利用現在的 Pose，算出那個絕對座標離我現在多遠)*

4.  **戴上帽子 (Extrinsics)**：
    $P_{lidar\_new} = R_{ext}^{-1} (P_{ego\_new} - T_{ext})$
    *(最後換算成現在頭頂上 LiDAR 的視角)*

通過這四步，我們就把「過去看到的樹」，成功搬到了「現在的眼前」。
