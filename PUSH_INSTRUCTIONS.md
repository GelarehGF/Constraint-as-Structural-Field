# How to Push This Repository to GitHub

This file walks you through replacing the existing single-notebook repository
at https://github.com/GelarehGF/Constraint-as-Structural-Field with the full
reproducibility package.

**IMPORTANT**: The steps below WILL overwrite the current remote history.
Since your remote has only one file and one commit, there is nothing valuable
to lose. If you are unsure, follow the "Safe Alternative" section instead.

---

## Prerequisites

- Git installed locally (`git --version` should show 2.30 or newer)
- A GitHub personal access token with `repo` scope (fine-grained tokens have
  caused 403 errors; use a classic PAT)
- Terminal access on your MacBook

---

## Method 1: Force-push over the existing repo (RECOMMENDED)

This is the fastest path. Run these commands in your terminal:

```bash
# 1. Extract the zip to a working directory
cd ~/Documents
unzip Constraint-as-Structural-Field-repo.zip
cd Constraint-as-Structural-Field

# 2. Initialize git and add everything
git init
git add .
git commit -m "Full reproducibility package for SCEN-EC paper"

# 3. Configure your identity (if not already set globally)
git config user.name "GelarehGF"
git config user.email "farhadiangelareh@gmail.com"

# 4. Add the remote
git remote add origin https://github.com/GelarehGF/Constraint-as-Structural-Field.git

# 5. Rename the local branch to match GitHub's default
git branch -M main

# 6. Force-push, overwriting the existing single-notebook history
git push --force origin main
```

When prompted:
- Username: `GelarehGF`
- Password: paste your personal access token (NOT your GitHub password)

**Expected output**: `Writing objects: 100% ... To github.com/GelarehGF/... + forced update`

---

## Method 2: Safe alternative (create a fresh branch)

If you want to preserve the current single-file state (for whatever reason):

```bash
cd ~/Documents
unzip Constraint-as-Structural-Field-repo.zip
cd Constraint-as-Structural-Field

git init
git remote add origin https://github.com/GelarehGF/Constraint-as-Structural-Field.git
git fetch origin

# Create a new branch instead of overwriting main
git checkout -b full-reproducibility-package
git add .
git commit -m "Full reproducibility package for SCEN-EC paper"
git push -u origin full-reproducibility-package
```

Then on GitHub, open a pull request from `full-reproducibility-package` -> `main`
and merge it manually.

---

## After pushing

1. Go to https://github.com/GelarehGF/Constraint-as-Structural-Field
2. Verify the README renders correctly (it should show the project header,
   badges, structure tree, and usage instructions)
3. On the repository's main page, click the gear icon next to "About" and add:
   - Description: "Replication materials for 'Constraint as Structural Field: A Computational Analysis of Entrepreneurial Discourse'"
   - Topics: `entrepreneurship`, `computational-discourse-analysis`, `nlp`, `field-theory`, `reproducibility`
   - Uncheck "Releases" and "Packages" if you don't plan to use them
4. Consider creating a release tag `v1.0.0` for the submission:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
5. **Get a Zenodo DOI**: Log into https://zenodo.org, link your GitHub account,
   enable the repository for Zenodo. Publish a release; Zenodo will mint a DOI.
   Add the DOI badge to the top of README.md. Reviewers at top-tier journals
   often require this for computational reproducibility.

---

## Adding your notebook

The current `SCEN-EC-02-Github.ipynb` on the remote should be preserved in the
new repo under `notebooks/`. Before running the push commands, copy it in:

```bash
# From the extracted repo folder
cp ~/Downloads/SCEN-EC-02-Github.ipynb notebooks/
```

Or, if the notebook already exists in your local Colab folder, adjust the path.

---

## Common problems

**403 error on push**:
- Your PAT probably has fine-grained permissions. Use a classic PAT with `repo`
  scope (Settings -> Developer settings -> Personal access tokens -> Tokens (classic))

**"refusing to merge unrelated histories"**:
- Add `--allow-unrelated-histories` to the force-push, or use Method 2

**"Support for password authentication was removed"**:
- You're passing your GitHub password instead of a PAT. Generate a token and
  paste that when prompted for the password

**Large file warnings**:
- The zip should not contain files over 100 MB. If it does, add them to
  `.gitignore` before committing. Check with:
  ```bash
  find . -type f -size +50M
  ```
