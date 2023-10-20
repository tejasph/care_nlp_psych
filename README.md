# care_nlp_psych
Cancer Ambulatory Resource Enhancement through early prediction of ER visits


### Updating Environment.yml

To maintain consistency in the repo, will always update the dependency list if new packages are installed. 

If it's your first time loading the .yml , then use `conda env create -f environment.yml`. 

If you add a package, use `conda install ____ --freeze-installed`

If you submit a pull request and you added a new package, then update the .yml with `conda env export --from-history > environment.yml`. 

If someone updated the yml and you pulled the changes then update your active local env with `conda env update --prefix ./env --file environment.yml  --prune`. 
