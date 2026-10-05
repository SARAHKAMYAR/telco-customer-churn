# Publish this project on GitHub

## 1. Review the project locally

Extract the ZIP into a permanent project folder. Read `README.md`, open `notebooks/telco_churn.ipynb`, and look through the figures. The notebook includes saved outputs so you can read it before installing anything.

The raw dataset is excluded from the ZIP. Copy your `Telco_customer_churn.xlsx` into `data/raw/` when you want to rerun the project. Follow the setup commands in the README. Do not upload the old PDF alongside the revised report as if they describe the same experiment.

## 2. Create a repository

Sign in to GitHub. Use the **+** menu and choose **New repository**. Enter:

| GitHub field | Text to use |
| --- | --- |
| Repository name | `telco-customer-churn` |
| Description | `Customer churn analysis with reproducible ML pipelines, model comparison, and a validation-based retention threshold.` |
| Visibility | Public, when you are ready for people to view it |
| Website | Leave blank until you have a live project page |

For the browser upload route below, leave **Add a README**, **Add .gitignore** and **Choose a license** unchecked. The prepared folder already includes the first two. A license grants reuse rights, so choose one deliberately after reviewing the code and checking any course rules. Dataset rights are separate.

Click **Create repository**. GitHub's official instructions are here: [Creating a new repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository).

## 3. Upload the prepared files

On the empty repository page, use **uploading an existing file**. On an existing repository, use **Add file → Upload files**. Open your extracted project folder in Finder and drag its contents into the upload area. Upload `README.md`, `requirements.txt`, `.gitignore`, and the folders `src`, `notebooks`, `figures`, `reports`, `tests`, `docs`, and `data`.

Keep the folder structure. Do not upload the ZIP itself, `.venv`, cache folders, or the raw Excel workbook. On macOS, **Command + Shift + .** shows hidden files such as `.gitignore`.

Use this first commit message:

> Add churn analysis, evaluated models, and project documentation

Optional commit description:

> Includes reproducible preprocessing, baseline comparisons, validation threshold selection, and test-set figures.

Commit the files to the main branch. Then open the README and notebook on GitHub and check that figures and saved outputs render. See [GitHub's file upload guide](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

## 4. Fill in the About section

On the repository home page, click the gear beside **About**. Use the description from step 2. Add these topics:

`customer-churn`, `machine-learning`, `python`, `scikit-learn`, `data-analysis`, `classification`, `random-forest`, `logistic-regression`

The README is already written. Keep it at the root of the repository and edit any first-person wording so it matches how you would explain the work. Make sure you can explain the split, the feature exclusions and the threshold choice before sharing it as your portfolio.

## 5. Pin it to your profile

Open your GitHub profile. In the pinned repositories area, choose **Customize your pins**, select this repository, and save. Copy the repository URL for your CV or portfolio.

Suggested portfolio entry:

> **Telco Customer Churn** — Extended a class project into a reproducible analysis of customer churn. Compared logistic regression and random forest with a baseline, kept model and threshold selection separate from test evaluation, and explored the trade-off between catching churners and contacting too many customers.

This describes the work without claiming deployment or measured revenue savings.

## 6. Make later changes with Git

For the first upload, the browser route is enough. For ongoing development, clone the published repository and work inside that clone:

```bash
git clone https://github.com/YOUR-USERNAME/telco-customer-churn.git
cd telco-customer-churn
```

Replace `YOUR-USERNAME` with your GitHub username. After making a change:

```bash
git status
git add README.md src notebooks figures reports docs tests requirements.txt .gitignore
git commit -m "Explain threshold choice and update analysis"
git push
```

Use a commit message that describes the actual change. Authenticate using GitHub's supported Git login flow when prompted. [GitHub's local project guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=mac) covers the command-line route.

## Which GitHub files do I need?

| File or section | Purpose | Included? |
| --- | --- | --- |
| README | Explain the question, results, methods and how to run the project | Yes |
| About description and topics | Help visitors understand and find the repository | Copy the text above |
| Notebook | Show the analysis with saved outputs | Yes |
| Source code | Keep the experiment reusable outside the notebook | Yes |
| Requirements | Record the analysis environment | Yes |
| Figures | Display evidence in the README and report | PNG and SVG |
| Tests | Check feature safety, split separation and preprocessing behavior | Yes |
| Data instructions | Explain which workbook is needed | Yes |
| License | State permission to reuse your code | Choose separately |
| Releases, wiki, contributing guide | Useful once a project needs them | Not needed for this first version |
