--- Draft to be cleaned and consolidated, and to be made into a proper PRODUCT_SPEC.md for agents to build under my guidance. ---

Final product vision: an OH webapp that allows the user to install a specific release of the SNT Stratification suite (SNT pipelines collection + associated webapps), and/or to check what is the statis of the workspace content relative to a specific release (check that all expected files are there and if they are of the correct version for a given release, else flag anything that is off) and then decided to either fix things (if files are missing or are of the wrong version: install and update to match a target release) or leave as is (it is possible that some changes were made manually and the user wants keep them).

This webapp is build as a nice UI/UX layer on top of an OH pipeline. This is because the pipeline is needed to do things that are a bit too much for a web app: namely pulling files (saving to OH ws file system) and deploying pipelines in OH, from the GitHub repo.

So for now I want to focus on the pipeline. The web app will come later and it is not a concern for now. I first need the pipeline.

What I want the pipeline to do:
* Check the status of the workspace: look at all files (or maybe just a set list of files such as the ones belonging to the pipelines and webapps) present in the workspace and extract their SHA. Compare this SHA against the release_manifest.json for each release tag of the reference GitHub repo, and derive to which release each file belongs to.
* Here the options are:
	* File belong to target release = correct (green light)
	* File does NOT belong to target release, so it could be: a) "behind" (older release), b) "ahead" (newer release), c) "unknown" (does not belong to any release, probably edited manually on purpose or corrupted)
	* also, mark files that are missing: defined in the release manifest but missing from file system or from pipelines database
* I think it could be handy of the pipeline could output a summary file with the status of all relevant files (as listed in the release manifest). This could be a json, already conceived to be readable by the (future) webapp.
* the GitHub repo will be public. The user eventually will not be able to chose the repo (so it should be hard coded, so we are in control), but for initial stages let's make it a parameter for ease of testing (I'm working with a test repo at the moment). 
* the user should be allowed to chose the release version thou, for reproducibility (it is possible that a country runs a certain analysis now, and in a year will wont to run the exact same analysis just with newer data). 
* the pipeline should be very verbose, explicit and clear, to avoid any "blackbox" feeling. Also, older or stale files should never be deleted, but moved to an "archive" location or somewhere findable by the user (then they can delete whatever they want, but manually) 

Open questions:
* should it be a single pipeline for all (check and fix, based on pipeline run parameters) or better to split into "check" (could be scheduled to run daily, no need for params, always checks against "latest" release) + "fix" (brings everything to a specific release, so overwrites and re-deploys wrong versions of files and pipelines).
* I think there should be a mechanism to import the release_manifest.json for all releases of the reference GitHub repo. This import should be performed by the pipeline.
* what to do with "other" files: things not listed in the release_manifest.json ? I'd say ignore as these will not affect the SNT stratification process
how to handle cases in which differnt releases have a different list of files (e.g., a pipeline is remove or added)

Also consider that to test these, I need an appropriate test repo. Currently snt_development_sandbox has 2 releases ("v0.0.1-test" and "v0.0.2-test"). Maybe I need to add a "latest" release (remember I never really worked with releases before).
