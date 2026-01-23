# Git Push Guide - How to Push to Multiple Repositories

This guide explains how to push code changes to multiple Git repositories.

## Current Repository Setup

### VirtualTryOn Repository
- **Remote Name**: `origin`
- **URL**: `https://github.com/royalpandit/VirtualTryOn.git`
- **Location**: `D:\vton demo\VirtualTryOn`

### CatVTON Repository  
- **Remote Name**: `origin`
- **URL**: `https://github.com/yashvardhansharma111/catvton.git`
- **Location**: `D:\vton demo\CatVTON`

## Method 1: Push to Single Repository (Current Setup)

### For VirtualTryOn Changes (Frontend)
```powershell
# Navigate to VirtualTryOn folder
cd "D:\vton demo\VirtualTryOn"

# Check what files changed
git status

# Add changed files
git add app.json
# Or add all changes: git add .

# Commit with message
git commit -m "Your commit message here"

# Push to origin (VirtualTryOn repo)
git push origin main
```

### For CatVTON Changes (Backend)
```powershell
# Navigate to CatVTON folder
cd "D:\vton demo\CatVTON"

# Check what files changed
git status

# Add changed files
git add app_fastapi.py
# Or add all changes: git add .

# Commit with message
git commit -m "Your commit message here"

# Push to origin (CatVTON repo)
git push origin main
```

## Method 2: Push Same Code to Multiple Repositories

### Step 1: Add Multiple Remotes

If you want to push the same code to multiple repositories from one folder:

```powershell
# Navigate to your project folder
cd "D:\vton demo\VirtualTryOn"

# Check current remotes
git remote -v

# Add additional remote (if not already added)
git remote add virtualtryon https://github.com/royalpandit/VirtualTryOn.git
git remote add catvton https://github.com/yashvardhansharma111/catvton.git

# Verify remotes
git remote -v
```

### Step 2: Push to All Remotes

```powershell
# Push to first remote
git push virtualtryon main

# Push to second remote
git push catvton main

# Or push to all remotes at once (if using git 2.4+)
git remote set-url --add --push origin https://github.com/royalpandit/VirtualTryOn.git
git remote set-url --add --push origin https://github.com/yashvardhansharma111/catvton.git
git push origin main
```

## Method 3: Push Different Folders to Different Repos

### When You Have Separate Folders

```powershell
# Push VirtualTryOn folder to VirtualTryOn repo
cd "D:\vton demo\VirtualTryOn"
git add .
git commit -m "Your message"
git push origin main

# Push CatVTON folder to CatVTON repo
cd "D:\vton demo\CatVTON"
git add .
git commit -m "Your message"
git push origin main
```

## Common Workflow

### 1. Make Changes
Edit files in your project

### 2. Check Status
```powershell
git status
```

### 3. Stage Changes
```powershell
# Stage specific file
git add filename.ext

# Stage all changes
git add .

# Stage all changes including deletions
git add -A
```

### 4. Commit Changes
```powershell
git commit -m "Descriptive commit message"
```

### 5. Push to Remote
```powershell
# Push to default remote (origin)
git push origin main

# Or just
git push
```

## Quick Reference Commands

```powershell
# Check current remotes
git remote -v

# Add new remote
git remote add <name> <url>

# Remove remote
git remote remove <name>

# Rename remote
git remote rename <old-name> <new-name>

# Change remote URL
git remote set-url <remote-name> <new-url>

# Push to specific remote
git push <remote-name> <branch-name>

# Pull from specific remote
git pull <remote-name> <branch-name>

# Fetch from all remotes
git fetch --all
```

## Example: Updating app.json and Pushing

```powershell
# 1. Navigate to VirtualTryOn
cd "D:\vton demo\VirtualTryOn"

# 2. Edit app.json (change API_URL or other settings)

# 3. Check what changed
git diff app.json

# 4. Stage the file
git add app.json

# 5. Commit
git commit -m "Update API URL in app.json"

# 6. Push to VirtualTryOn repo
git push origin main

# 7. If you also need to update CatVTON repo (for shared files)
cd "D:\vton demo\CatVTON"
git add app_fastapi.py  # or whatever file changed
git commit -m "Update shared file"
git push origin main
```

## Troubleshooting

### Error: "remote origin already exists"
```powershell
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin <new-url>
```

### Error: "failed to push some refs"
```powershell
# Pull latest changes first
git pull origin main

# Then push
git push origin main
```

### Push to Different Branch
```powershell
# Create and switch to new branch
git checkout -b feature-branch

# Make changes and commit
git add .
git commit -m "Feature update"

# Push to new branch
git push origin feature-branch
```

## Best Practices

1. **Always check status** before committing: `git status`
2. **Use descriptive commit messages**: Explain what and why
3. **Pull before push** if working with others: `git pull origin main`
4. **Test before pushing** to avoid breaking production
5. **Push frequently** to avoid large commits
6. **Use branches** for features: `git checkout -b feature-name`

## Current Setup Summary

- **VirtualTryOn** → Pushes to `https://github.com/royalpandit/VirtualTryOn.git`
- **CatVTON** → Pushes to `https://github.com/yashvardhansharma111/catvton.git`

Both repositories are independent. Changes in one don't automatically sync to the other unless you manually push to both.
