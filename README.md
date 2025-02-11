# BINF 6310: GAN-WGCNA
Group Members: Jane Adams, Madison Nguyen, Nicole Leon Vargas, Tzu-Tung (Bella) Chang

## Project Summary

The purpose of this project is to reproduce a published work in bioinformatics. we selected ["GAN-WGCNA: Calculating gene modules to identify key intermediate regulators in cocaine addiction"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0311164#sec019). This study stood out for its innovative integration of Weighted Gene Co-expression Network Analysis (WGCNA) and Generative Adversarial Networks (GANs)—two well-established but distinct computational techniques. The combination of these methods offers a powerful approach to uncovering gene expression patterns linked to phenotypic traits, making it an exciting case study in applying machine learning to biological data. The study’s focus on cocaine addiction was particularly compelling, as it demonstrates how computational methods can enhance our understanding of complex disorders with genetic components. A key factor in our decision was the study’s transparency and accessibility. The authors provided both the dataset and analysis scripts, which are well-structured and documented. This ensures a clearer path for our reproducibility study and allows us to critically examine and validate their findings without unnecessary roadblocks related to missing or proprietary data.

The original code for the project is on Github [here](https://github.com/baicalin/GAN-WGCNA).

## Project Structure

Each analysis notebook should have a data directory listed at the top; if we use these, then each analysis step will have its own folder, in order, in the data directory. For example, `analysis/00_data-gathering` deposits all output files to `data/00_data-gathering`. 

> [!TIP]  
>  Note that the data directory is in the .gitignore, which means that you will need to run each analysis step to generate the data for your local use. We do this so we don't need to use `git-lfs`, the large file system for git.

> [!WARNING]  
> Be careful not to commit any large files, because they are hard to retroactively remove from tracking. Most data file extensions should already be in the .gitignore, so this should be handled automatically.

### Dependencies
> [!WARNING]  
> We are using uv. DO NOT use `pip` directly or create a requirements.txt file. For example, instead of `pip install tqdm`, you should use `uv add tqdm`, or prepend all pip commands, e.g. `uv pip install tqdm`.

This project uses `uv` for dependency management. [Read more about `uv` here.](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) To set up your environment on first clone, run `uv sync`. Learn more about uv project structure [here](https://docs.astral.sh/uv/guides/projects/#project-structure).

To activate the virtual environment after it is created, you can call `source .venv/bin/activate` on Mac/Linux or `.venv\Scripts\activate` on Windows. You might want to just add an alias to your bash config, like `vim .zshrc` > `alias activate="source .venv/bin/activate"` so you can just type `activate` to activate.

## Contributing

Use branching to make commits. For example, to create a new branch:

`git checkout -b your_branch`

To push your new branch:

`git push -u origin your_branch`

Create a pull request in the repository with an explanation of your changes, and then email the group.

You may find it helpful to use [Github Desktop](https://github.com/apps/desktop) to review your changes and any merge conflicts.

Once your PR is merged, make sure to **delete your old branch** so it doesn't cause confusion. Also, make sure to always `git pull` main before you branch, so you know you're working with the most up-to-date code.