# Auto Push 設定說明

## 設定日期
2026-04-13

## 做了什麼

1. 在 `/data2/t113c52027/occ_gt_v2` 初始化 git repository
2. 建立 `.gitignore`，排除以下大型檔案：
   - `data/`（原始資料）
   - `*/output/`（程式輸出結果）
   - `*.mp4`（影片檔）
   - `*.ipynb`、`__pycache__/` 等
3. 將程式碼（745 個檔案）推送至 GitHub：
   - Repo：https://github.com/AmazingWilson-hub/occ_gt_v2
   - Branch：master
4. 在 `.claude/settings.json` 設定 Claude Code Hook：
   - 觸發時機：每次 Claude 完成回答後（Stop 事件）
   - 動作：自動 `git add -A` → `git commit` → `git push`
   - Commit 訊息格式：`Auto backup YYYY-MM-DD HH:MM`

---

## 日後注意事項

### Git 指令需加 GIT_DIR 參數
這個目錄的 git 指令需要明確指定路徑，否則會報錯：
```bash
GIT_DIR=/data2/t113c52027/occ_gt_v2/.git \
GIT_WORK_TREE=/data2/t113c52027/occ_gt_v2 \
git <指令>
```

### 新增要排除的檔案
如果有新的大型檔案不想上傳，編輯 `.gitignore` 加入對應規則即可。

### Token 過期處理
GitHub Personal Access Token 若過期或被撤銷，自動 push 會失敗。
處理方式：
1. 至 GitHub 產生新 token（需勾選 `repo` 權限）
2. 執行以下指令更新 credential：
```bash
echo "https://AmazingWilson-hub:<新token>@github.com" >> ~/.git-credentials
```

### 手動觸發 push
如果想手動 push，執行：
```bash
GIT_DIR=/data2/t113c52027/occ_gt_v2/.git GIT_WORK_TREE=/data2/t113c52027/occ_gt_v2 git add -A
GIT_DIR=/data2/t113c52027/occ_gt_v2/.git GIT_WORK_TREE=/data2/t113c52027/occ_gt_v2 git commit -m "手動備份"
GIT_DIR=/data2/t113c52027/occ_gt_v2/.git GIT_WORK_TREE=/data2-t113c52027/occ_gt_v2 git push origin clean-main:master
```

### 關閉自動 push
編輯 `.claude/settings.json`，將 `hooks` 區塊刪除即可。
