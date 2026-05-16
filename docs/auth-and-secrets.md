# 認証 & シークレット (B-1 解決方針)

## 結論
**コンテナへの token 注入は不要**。本プロジェクトは **ローカル macOS** 上で tmux + `claude -p --worktree` で実行するため、Claude プロセスは起動シェルの環境を継承する。

## 前提環境 (2026-05-16 確認済)
| 認証 | 状態 | 場所 |
|---|---|---|
| Kaggle API | ✓ 認証済 | `~/.kaggle/kaggle.json` (perms 600) |
| GitHub gh CLI | ✓ Y-Kanekoo として認証済 (`gho_***`, scopes: gist/read:org/repo/workflow) | keyring (macOS Keychain) |
| Git push | ✓ origin = `https://github.com/Y-Kanekoo/orbit-wars.git` 設定済 | `.git/config` |

## 自律実行時の継承経路
```
ユーザー shell (zsh)
  └─ env: KAGGLE_CONFIG_DIR=~/.kaggle (default)
  └─ keychain access (gh CLI)
  └─ git credentials (gh が helper として設定)
       │
       ▼
  tmux session
       │
       ▼
  claude -p --worktree --max-budget-usd <N>
       │  (env を inherit)
       │
       ▼
  Bash tool 経由で kaggle / gh / git コマンド実行
       └─ ~/.kaggle/kaggle.json を読む
       └─ keychain から GitHub token を取得
       └─ git push が origin に通る
```

## ローカル実行で追加対応が必要なもの

### Kaggle Notebooks (kernel) を push する場合
`kaggle kernels push` は `~/.kaggle/kaggle.json` を読むため、追加設定不要。

### Kaggle Notebooks の中で kaggle API を呼ぶ場合
Notebook 自体に Kaggle credentials を埋め込むのは漏洩リスク。
- **Kaggle Secrets** を Notebook 設定で有効化
- Notebook 内で `from kaggle_secrets import UserSecretsClient` → `UserSecretsClient().get_secret("KAGGLE_API_TOKEN")` 等

### GitHub Actions を併用する場合 (将来)
- repo settings → Secrets and variables → Actions に `KAGGLE_USERNAME`, `KAGGLE_KEY` を登録
- Workflow から `${{ secrets.KAGGLE_KEY }}` 経由で参照

## 禁止事項
- `.kaggle/kaggle.json` を repo に commit (`.gitignore` で除外済)
- ハードコード (`KAGGLE_KEY = "abc123"` 等)
- ログ・PR description への token 漏洩
- ストレージへの平文保存

## チェック
セットアップ完了の検証コマンド:
```bash
# Kaggle
kaggle competitions list --search orbit | head -3

# GitHub
gh auth status

# Git push 経路
git ls-remote origin HEAD
```
すべて成功すれば B-1 は解決済。
