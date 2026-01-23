# Git Push Guide - Pushing to Multiple Repositories

This guide explains how to push code to both repositories from the `d:\vton demo` root directory.

## Repository Structure

- **Root**: `d:\vton demo\`
  - Contains both `CatVTON/` and `VirtualTryOn/` folders
  - Pushes to: `https://github.com/royalpandit/VirtualTryOn.git`

- **CatVTON Only**: `d:\vton demo\CatVTON\`
  - Pushes to: `https://github.com/yashvardhansharma111/catvton.git`

## Pushing to VirtualTryOn Repository (Both Folders)

From `d:\vton demo\`:

```bash
cd "d:\vton demo"
git add .
git commit -m "Your meaningful commit message"
git push origin main
```

This pushes both `CatVTON/` and `VirtualTryOn/` folders to the VirtualTryOn repo.

## Pushing to CatVTON Repository (Backend Only)

From `d:\vton demo\CatVTON\`:

```bash
cd "d:\vton demo\CatVTON"
git add .
git commit -m "Your meaningful commit message"
git push origin main
```

This pushes only the CatVTON backend code to the CatVTON repo.

## Quick Push to Both Repos

To push to both repositories:

```bash
# Push to VirtualTryOn (both folders)
cd "d:\vton demo"
git add .
git commit -m "Your meaningful commit message"
git push origin main

# Push to CatVTON (backend only)
cd "d:\vton demo\CatVTON"
git add .
git commit -m "Your meaningful commit message"
git push origin main
```

## Important Notes

1. **Root-level README files** in `d:\vton demo\` are excluded via `.gitignore`
2. **Always commit with meaningful messages** describing what changed
3. **Pull before pushing** if working with others: `git pull origin main`
4. **Check status first**: `git status` to see what will be committed

## Troubleshooting

### Merge Conflicts
If you get merge conflicts:
```bash
git pull origin main --allow-unrelated-histories
# Resolve conflicts in files
git add .
git commit -m "Resolve merge conflicts"
git push origin main
```

### Unrelated Histories
If you see "refusing to merge unrelated histories":
```bash
git pull origin main --allow-unrelated-histories --no-rebase
```
